"""
The five agents of the multi-agent system.

  supervisor  (LLM)  reads the question into a typed Plan, then after every phase decides which agent
                     works next (Route). Its choice is checked by code guards ("LLM proposes, code validates").
  research    (LLM ReAct agent, one per company, run in parallel) fetches the figures with tools.
  quant       (LLM ReAct agent) writes Python for the calculation and runs it in the MCP sandbox.
  auditor     (deterministic, no LLM) checks completeness, grounding, recomputes every result, checks the winner.
  writer      (LLM) writes the final answer from the audited table; optional human approval first.
"""
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, Send, interrupt

from config.companies import COMPANIES
from config.settings import settings
from src.mcp.facts_server import _manifest
from src.multiagent.state import (FORMULAS, Findings, MAState, Plan, QuantResult, Route,
                                  required_metrics, unit_of)
from src.multiagent.tools import research_tools, sandbox_tool
from src.retrieval.query_filters import detect_tickers

logger = logging.getLogger(__name__)

NAMES = {c["ticker"]: c["name"] for c in COMPANIES}
TOLERANCE = {"percent": 0.05, "ratio": 0.005}   # absolute; amounts use 0.5% relative


def get_agent_llm() -> ChatOllama:
    """The tool-calling model every agent uses (settings.AGENT_MODEL), at temperature 0 for repeatable runs."""
    # num_ctx: Ollama's default window is small; research prompts carry 5 passages of filing text
    return ChatOllama(model=settings.AGENT_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0, num_ctx=8192)


def _text(content: Any) -> str:
    """Tool results from MCP arrive as content blocks; plain tools return strings."""
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _numbers(text: str) -> List[float]:
    """Every number in a text ("391,035" -> 391035.0), used by the grounding check."""
    return [float(m.replace(",", "")) for m in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def _close(a: float, b: float, unit: str) -> bool:
    """Equal within tolerance: 0.05 points for percentages, 0.005 for ratios, 0.5% for amounts."""
    tol = TOLERANCE.get(unit, 0.005 * max(abs(a), abs(b)))
    return abs(a - b) <= tol


def _event(agent: str, action: str, **detail) -> Dict[str, Any]:
    """One entry of the run's trace: the per-run record returned to the user (logs are for operators)."""
    return {"agent": agent, "action": action, **detail}


# ======================================================================================
# Supervisor
# ======================================================================================
PLAN_PROMPT = """You plan the work for a team of financial analyst agents.
Companies you can use (name = ticker): {companies}

Read the question and fill the plan:
- tickers: every company the question names (tickers from the list above)
- fiscal_year: the fiscal year asked about
- calculation: gross_margin, operating_margin, net_margin, rd_intensity (R&D / revenue),
  revenue_growth, net_income_growth, current_ratio, or "compare" when the question ranks one
  reported figure (then set compare_metric: revenue, net_income, operating_income, gross_profit,
  rd, cfo (operating cash flow), total_assets, equity, eps_diluted)
- direction: "highest" unless the question asks for the lowest/smallest

Question: {question}"""

ROUTE_PROMPT = """You supervise analyst agents answering: "{question}"
Status:
{status}

Choose the next agent:
- research: fetch missing figures (list the tickers that need it)
- quant: all figures are present and the calculation has not been done (or was wrong)
- write: the audit passed (or no more retries are possible), write the final answer"""


# Phrases that name a calculation unambiguously. In testing, Qwen 2.5 7B read "operating margin
# (operating income / total revenue)" as "compare operating income" in 3 of 8 three-company questions,
# so code checks the planner's calculation against the question's own words.
CALC_PHRASES = [
    (r"gross margin", "gross_margin"), (r"operating margin", "operating_margin"),
    (r"net (profit )?margin|profit margin", "net_margin"),
    (r"r&d intensity|research and development (expense )?(/|as a (percentage|share) of|to) (total )?revenue",
     "rd_intensity"),
    (r"revenue growth|growth in (total )?revenue", "revenue_growth"),
    (r"net income growth|growth in net income", "net_income_growth"), (r"current ratio", "current_ratio"),
]


def _calculation_from_words(question: str) -> Optional[str]:
    low = question.lower()
    for pattern, calculation in CALC_PHRASES:
        if re.search(pattern, low):
            return calculation
    return None


def _validate_plan(plan: Plan, question: str) -> Dict[str, Any]:
    """Code checks the LLM's plan: real tickers, an indexed fiscal year, the inputs each calculation needs."""
    available = {t for (t, _) in _manifest()}
    llm_tickers = [t.upper() for t in plan.tickers if t.upper() in available]
    # Company names are matched deterministically (config/companies.py aliases); the LLM's list is used
    # only when the question names no known company (e.g. it typed a ticker the alias list lacks)
    named = [t for t in detect_tickers(question) if t in available]
    tickers = named or llm_tickers
    llm_calculation = plan.calculation
    worded = _calculation_from_words(question)
    if worded and worded != plan.calculation:
        plan = plan.model_copy(update={"calculation": worded, "compare_metric": None})
    if re.search(r"\b(lowest|smallest|least|weakest)\b", question.lower()):
        plan = plan.model_copy(update={"direction": "lowest"})
    metrics = required_metrics(plan.calculation, plan.compare_metric)
    return {**plan.model_dump(), "tickers": tickers, "llm_tickers": llm_tickers, "llm_calculation": llm_calculation,
            "metrics": metrics,
            "unit": unit_of(plan.calculation, plan.compare_metric),
            "tasks": [{"ticker": t, "fiscal_year": plan.fiscal_year, "metrics": metrics} for t in tickers]}


def _latest_findings(state: MAState) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """ticker -> metric -> newest finding (retries append; the newest wins)."""
    latest: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for f in state.get("findings", []):
        latest.setdefault(f["ticker"], {})[f["metric"]] = f
    return latest


def _missing(state: MAState) -> Dict[str, List[str]]:
    """ticker -> metrics the plan needs that no research agent has found yet."""
    latest = _latest_findings(state)
    return {t["ticker"]: [m for m in t["metrics"] if m not in latest.get(t["ticker"], {})]
            for t in state["plan"]["tasks"]
            if any(m not in latest.get(t["ticker"], {}) for m in t["metrics"])}


def _status(state: MAState) -> str:
    """A short text summary of progress, shown to the supervisor's LLM when it has a routing choice."""
    latest = _latest_findings(state)
    lines = []
    for t in state["plan"]["tasks"]:
        got = {m: latest.get(t["ticker"], {}).get(m, {}).get("value") for m in t["metrics"]}
        lines.append(f"- {t['ticker']}: " + ", ".join(f"{m}={v if v is not None else 'MISSING'}" for m, v in got.items())
                     + f" (research rounds used: {state.get('research_rounds', {}).get(t['ticker'], 0)}"
                       f"/{settings.AGENT_MAX_RESEARCH_ROUNDS + 1})")
    quant = state.get("quant")
    lines.append(f"- calculation ({state['plan']['calculation']}): " + ("done" if quant else "not done"))
    audit = state.get("audit")
    if audit:
        lines.append("- audit: " + ("PASSED" if audit["passed"] else "FAILED: " + "; ".join(audit["problems"])))
    return "\n".join(lines)


def _research_sends(state: MAState, tickers: List[str], feedback: Dict[str, str]) -> List[Send]:
    """One Send per company: LangGraph runs these research agents in parallel, each with its own input."""
    tasks = {t["ticker"]: t for t in state["plan"]["tasks"]}
    return [Send("research", {"task": tasks[t], "question": state["question"], "mode": state.get("mode", "full"),
                              "feedback": feedback.get(t, "")}) for t in tickers if t in tasks]


async def supervisor_node(state: MAState) -> Command:
    """Plans the work on the first visit, then chooses the next agent after every phase.

    Reads:  question, plan, findings, quant, audit, research_rounds, hops
    Writes: plan (first visit), research_rounds, hops, guard_overrides, llm_decisions, trace
    Routes: Command(goto=[Send("research", ...)...]) | "quant" | "writer"
    """
    hops = state.get("hops", 0) + 1
    llm = get_agent_llm()

    # 1. First visit: plan the work and fan out one research agent per company, in parallel
    if not state.get("plan"):
        companies = ", ".join(f"{c['name']} = {c['ticker']}" for c in COMPANIES)
        plan = await llm.with_structured_output(Plan).ainvoke(
            PLAN_PROMPT.format(companies=companies, question=state["question"]))
        validated = _validate_plan(plan, state["question"])
        rounds = {t: 1 for t in validated["tickers"]}      # first research attempt per company
        logger.info("plan: %s fiscal %s, calculation=%s%s", validated["tickers"], validated["fiscal_year"],
                    validated["calculation"],
                    f" (LLM said {validated['llm_calculation']}, corrected by code)"
                    if validated["llm_calculation"] != validated["calculation"] else "")
        trace = [_event("supervisor", "plan", tickers=validated["tickers"], llm_tickers=validated["llm_tickers"],
                        llm_calculation=validated["llm_calculation"],
                        fiscal_year=validated["fiscal_year"],
                        calculation=validated["calculation"], compare_metric=validated.get("compare_metric"))]
        if not validated["tickers"]:
            logger.warning("no known company in the question; going straight to the writer")
            return Command(goto="writer", update={"plan": validated, "hops": hops, "trace": trace,
                                                  "audit": {"passed": False, "problems": ["no known company in the question"]}})
        sends = [Send("research", {"task": t, "question": state["question"], "mode": state.get("mode", "full"),
                                   "feedback": ""}) for t in validated["tasks"]]
        return Command(goto=sends, update={"plan": validated, "research_rounds": rounds, "hops": hops, "trace": trace})

    # 2. Later visits: the LLM proposes the next agent, code validates it
    missing = _missing(state)
    audit = state.get("audit")
    rounds = dict(state.get("research_rounds", {}))
    # rounds counts attempts so far (first attempt = 1), so <= MAX_RESEARCH_ROUNDS leaves room for one more
    can_retry = {t for t in state["plan"]["tickers"] if rounds.get(t, 0) <= settings.AGENT_MAX_RESEARCH_ROUNDS}

    # Guard: the moves that are valid in the current state (code decides what is ALLOWED)
    redo = sorted(set((audit or {}).get("redo_research", [])) & can_retry)
    retry_missing = sorted(set(missing) & can_retry)
    if hops >= settings.AGENT_MAX_HOPS:
        allowed, default = ["write"], "write"
    elif missing:
        # research again, or accept that a figure isn't reported and answer with what we have
        allowed, default = (["research", "write"], "research") if retry_missing else (["write"], "write")
    elif not state.get("quant"):
        allowed, default = ["quant"], "quant"
    elif audit and not audit["passed"]:
        allowed = (["research"] if redo else []) + \
                  (["quant"] if audit.get("redo_quant") and not state["quant"].get("retried") else []) + ["write"]
        default = allowed[0]
    else:
        allowed, default = ["write"], "write"

    # The LLM decides only when there is a real choice; a forced move costs no LLM call
    if len(allowed) == 1:
        final, proposed, reason = allowed[0], None, "only valid move"
    else:
        try:
            route = await llm.with_structured_output(Route).ainvoke(
                ROUTE_PROMPT.format(question=state["question"], status=_status(state))
                + f"\nValid choices now: {', '.join(allowed)}")
            proposed, reason = route.next, route.reason
        except Exception as err:   # an unparsable decision falls back to the guard's default
            logger.warning("router output invalid, using default '%s'", default, exc_info=True)
            proposed, reason = None, f"router output invalid: {type(err).__name__}"
        final = proposed if proposed in allowed else default
    overridden = proposed is not None and final != proposed
    overrides = state.get("guard_overrides", 0) + overridden
    logger.info("route #%d: %s (valid: %s%s)", hops, final, ", ".join(allowed),
                f"; LLM proposed {proposed}, OVERRIDDEN" if overridden else
                (f"; LLM chose it: {reason}" if len(allowed) > 1 else "; forced"))
    trace = [_event("supervisor", "route", allowed=allowed, proposed=proposed, chosen=final, reason=reason,
                    overridden=overridden, llm_decided=len(allowed) > 1)]
    update = {"hops": hops, "guard_overrides": overrides, "trace": trace,
              "llm_decisions": state.get("llm_decisions", 0) + (len(allowed) > 1)}

    if final == "research":
        targets = retry_missing if missing else redo
        feedback = {t: "; ".join(p for p in (audit or {}).get("problems", []) if t in p) for t in targets}
        for t in targets:
            rounds[t] = rounds.get(t, 0) + 1
        logger.info("re-research %s (attempt %s)", targets, {t: rounds[t] for t in targets})
        # a re-research makes the old calculation and audit stale
        return Command(goto=_research_sends(state, targets, feedback),
                       update={**update, "research_rounds": rounds, "quant": {}, "audit": {}})
    if final == "quant":
        retried = bool(state.get("quant"))
        return Command(goto="quant", update={**update, "quant": {"retried": retried} if retried else {}})
    return Command(goto="writer", update=update)


# ======================================================================================
# Research agent (one per company, run in parallel with Send)
# ======================================================================================
RESEARCH_PROMPT = """You are a financial research analyst. Get these figures for {name} ({ticker}), fiscal {year}:
{metrics}
Rules:
- First call get_financial_metric for each figure (if available), always with fiscal_year={year}, using the
  metric names exactly as listed (revenue_prior / net_income_prior are their own metric names: the previous
  year's figure as reported in the fiscal {year} 10-K). Use exactly the value it returns.
- If it returns an error or is not available, call search_filings for {ticker} {year} and read the figure
  from the passages (table rows look like 'Gross margin — 2024: 180,683; 2023: 169,148', in $ millions).
- Never estimate or calculate a figure yourself. Report the source id the tool gave.
{feedback}"""


EXTRACT_PROMPT = """From these tool results for {ticker} fiscal {year}, report the figures below.
Use only numbers that appear in the tool results, exactly as printed (values in $ millions), with their source id.
Skip a figure if it is not there. Figures: {metrics}

Tool results:
{results}"""


def _xbrl_findings(tool_log: List[Dict[str, Any]], ticker: str, year: int, metrics: List[str]) -> List[Dict[str, Any]]:
    """Figures from get_financial_metric results are read straight from its JSON: no LLM needed."""
    found = {}
    for entry in tool_log:
        if entry["tool"] != "get_financial_metric":
            continue
        try:
            data = json.loads(entry["content"])
        except ValueError:
            continue
        # Only the exact filing asked for: a value fetched for another fiscal year is not this year's figure
        if data.get("metric") in metrics and "value" in data and data.get("ticker") == ticker \
                and data.get("fiscal_year") == year:
            found[data["metric"]] = {"ticker": ticker, "fiscal_year": year, "metric": data["metric"],
                                     "value": float(data["value"].replace(",", "")), "source": data["source"],
                                     **({"derived": data["derived"]} if data.get("derived") else {})}
    return list(found.values())


async def research_node(payload: Dict[str, Any]) -> Dict[str, Any]:
    """One company's figures. Runs once per company, in parallel (the payload comes from Send, not the full state).

    Reads:  payload = {task: {ticker, fiscal_year, metrics}, question, mode, feedback}
    Writes: findings, tool_log (both appended through reducers), trace
    """
    started = time.perf_counter()
    task, mode = payload["task"], payload.get("mode", "full")
    ticker, year, metrics = task["ticker"], task["fiscal_year"], task["metrics"]
    tools = await research_tools(mode)
    rules = "Use search_filings only." if mode == "text" else ""
    prompt = RESEARCH_PROMPT.format(
        name=NAMES.get(ticker, ticker), ticker=ticker, year=year,
        metrics="\n".join(f"- {m}" for m in metrics),
        feedback=(f"A previous attempt was rejected by the auditor: {payload['feedback']}" if payload.get("feedback") else "")
        + ("\n" + rules if rules else ""))
    # A standard ReAct loop (model <-> tools until the model stops calling tools); no response_format,
    # because XBRL figures are read from the tool JSON below and only text findings need an LLM extraction
    agent = create_react_agent(get_agent_llm(), tools, name=f"research_{ticker}")
    tool_log, findings, steps = [], [], []
    try:
        result = await agent.ainvoke({"messages": [("user", prompt)]}, config={"recursion_limit": 14})
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                steps += [f"{c['name']}({json.dumps(c['args'])})" for c in msg.tool_calls]
            if isinstance(msg, ToolMessage):
                tool_log.append({"ticker": ticker, "agent": "research", "tool": msg.name, "content": _text(msg.content)})
        findings = _xbrl_findings(tool_log, ticker, year, metrics)
        still_missing = [m for m in metrics if m not in {f["metric"] for f in findings}]
        text_results = [e["content"] for e in tool_log if e["tool"] == "search_filings"]
        if still_missing and text_results:
            # Figures read from filing text need the LLM to pick them out: one structured call
            structured = await get_agent_llm().with_structured_output(Findings).ainvoke(EXTRACT_PROMPT.format(
                ticker=ticker, year=year, metrics=", ".join(still_missing), results="\n\n".join(text_results)[:6000]))
            for f in structured.findings:
                if f.metric in still_missing:
                    findings.append({"ticker": ticker, "fiscal_year": year, **f.model_dump(),
                                     "source": f.source.strip("[] ")})
    except GraphRecursionError:
        logger.warning("research[%s] stopped at the step limit", ticker)
        steps.append("stopped: step limit")
    except Exception as err:     # e.g. a structured answer that fails validation: report, don't crash
        logger.warning("research[%s] failed", ticker, exc_info=True)
        steps.append(f"failed: {type(err).__name__}: {str(err)[:120]}")
    logger.info("research[%s] %.1fs: %d tool calls, found %s", ticker, time.perf_counter() - started, len(steps),
                {f["metric"]: f["value"] for f in findings} or "nothing")
    return {"findings": findings, "tool_log": tool_log,
            "trace": [_event(f"research[{ticker}]", "fetched", tool_calls=steps,
                             found={f["metric"]: f["value"] for f in findings})]}


# ======================================================================================
# Quant agent
# ======================================================================================
QUANT_PROMPT = """You are a quantitative analyst. Compute {calculation} for each company from these audited inputs
($ millions unless stated):
{table}
Formula: {formula}
Call execute_python_calc ONCE with Python code that sets `result` to a dict of ticker -> value rounded
to 2 decimals, e.g. result = {{"AAPL": round(180683 / 391035 * 100, 2), ...}}.
Then report every company's value from the sandbox output and the winner (the company with the
{direction} value)."""

FORMULA_TEXT = {
    "compare": "the reported figure itself (no arithmetic)",
    "gross_margin": "gross_profit / revenue * 100", "operating_margin": "operating_income / revenue * 100",
    "net_margin": "net_income / revenue * 100", "rd_intensity": "rd / revenue * 100",
    "revenue_growth": "(revenue / revenue_prior - 1) * 100",
    "net_income_growth": "(net_income / net_income_prior - 1) * 100",
    "current_ratio": "current_assets / current_liabilities",
}


async def quant_node(state: MAState) -> Dict[str, Any]:
    """Computes the calculation for every company by writing Python and running it in the MCP sandbox.

    Reads:  plan, findings (newest per company and metric)
    Writes: quant = {values, winner, code, sandbox_output, retried}, tool_log, trace
    """
    started = time.perf_counter()
    plan = state["plan"]
    latest = _latest_findings(state)
    table = "\n".join(f"- {t}: " + ", ".join(f"{m} = {latest[t][m]['value']:g}" for m in plan["metrics"])
                      for t in plan["tickers"] if t in latest)
    prompt = QUANT_PROMPT.format(calculation=plan["calculation"], table=table, direction=plan["direction"],
                                 formula=FORMULA_TEXT[plan["calculation"]])
    agent = create_react_agent(get_agent_llm(), [await sandbox_tool()], response_format=QuantResult, name="quant")
    sandbox_out, code = "", ""
    quant: Dict[str, Any] = {"retried": bool(state.get("quant", {}).get("retried"))}
    try:
        result = await agent.ainvoke({"messages": [("user", prompt)]}, config={"recursion_limit": 10})
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                for c in msg.tool_calls:
                    code = c["args"].get("code", code)
            if isinstance(msg, ToolMessage):
                sandbox_out = _text(msg.content)
        structured: Optional[QuantResult] = result.get("structured_response")
        if structured:
            quant.update({"values": {v.ticker.upper(): v.value for v in structured.values},
                          "winner": structured.winner.upper()})
    except GraphRecursionError:
        logger.warning("quant stopped at the step limit")
        quant["error"] = "step limit"
    except Exception as err:
        logger.warning("quant failed", exc_info=True)
        quant["error"] = f"{type(err).__name__}: {str(err)[:120]}"
    logger.info("quant %.1fs: values=%s winner=%s sandbox=%s", time.perf_counter() - started,
                quant.get("values"), quant.get("winner"), (sandbox_out or "not run")[:80])
    quant.update({"code": code, "sandbox_output": sandbox_out})
    return {"quant": quant,
            "tool_log": [{"ticker": "*", "agent": "quant", "tool": "execute_python_calc", "content": sandbox_out}],
            "trace": [_event("quant", "computed", code=code, sandbox_output=sandbox_out,
                             values=quant.get("values"), winner=quant.get("winner"))]}


# ======================================================================================
# Auditor (deterministic: these checks don't need an LLM's opinion)
# ======================================================================================
def audit(state: MAState) -> Dict[str, Any]:
    """Four exact checks; returns what failed and which agent must redo its work.

    Returns {passed, problems, redo_research: [tickers], redo_quant: bool, expected: {ticker: value}}.
    "expected" (recomputed from grounded inputs) is what the writer reports, not the quant agent's numbers.
    """
    plan, quant = state["plan"], state.get("quant") or {}
    latest = _latest_findings(state)
    problems, redo_research, redo_quant = [], set(), False

    # 1. Completeness + 2. grounding: every figure must appear in a tool result for that company
    for t in plan["tickers"]:
        observed = [n for e in state.get("tool_log", []) if e["ticker"] == t for n in _numbers(e["content"])]
        for m in plan["metrics"]:
            f = latest.get(t, {}).get(m)
            if f is None:
                problems.append(f"{t}: {m} missing")
                redo_research.add(t)
            # 0.5 absorbs rounding ("391,035" vs 391035.4); a figure no tool returned was invented
            elif not any(abs(f["value"] - n) <= 0.5 for n in observed):
                problems.append(f"{t}: {m}={f['value']:g} does not appear in any tool result (ungrounded)")
                redo_research.add(t)

    # 3. Recompute every company's result from the grounded inputs and compare with the quant agent
    expected: Dict[str, float] = {}
    for t in plan["tickers"]:
        inputs = {m: latest.get(t, {}).get(m, {}).get("value") for m in plan["metrics"]}
        if None in inputs.values():
            continue
        try:
            expected[t] = inputs[plan["metrics"][0]] if plan["calculation"] == "compare" \
                else FORMULAS[plan["calculation"]][1](inputs)
        except ZeroDivisionError:
            problems.append(f"{t}: division by zero in {plan['calculation']}")
    got = quant.get("values", {})
    if not quant.get("sandbox_output", "").startswith("SUCCESS"):
        problems.append("quant: the sandbox was not run successfully")
        redo_quant = True
    for t, v in expected.items():
        if t not in got:
            problems.append(f"quant: no value for {t}")
            redo_quant = True
        elif not _close(got[t], v, plan["unit"]):
            problems.append(f"quant: {t} = {got[t]:g}, but the inputs give {v:.2f}")
            redo_quant = True

    # 4. The winner must follow from the recomputed values
    if expected:
        pick = max if plan["direction"] == "highest" else min
        right = pick(expected, key=expected.get)
        if quant.get("winner") != right:
            problems.append(f"quant: winner {quant.get('winner')} but the {plan['direction']} is {right}")
            redo_quant = True
    return {"passed": not problems, "problems": problems, "redo_research": sorted(redo_research),
            "redo_quant": redo_quant, "expected": expected}


async def auditor_node(state: MAState) -> Dict[str, Any]:
    """Graph node around audit(): writes the verdict for the supervisor to act on."""
    verdict = audit(state)
    if verdict["passed"]:
        logger.info("audit passed")
    else:
        logger.warning("audit failed: %s", "; ".join(verdict["problems"]))
    return {"audit": verdict, "trace": [_event("auditor", "passed" if verdict["passed"] else "failed",
                                               problems=verdict["problems"])]}


# ======================================================================================
# Writer (+ optional human approval)
# ======================================================================================
WRITER_PROMPT = """Write the answer to: "{question}"
Use ONLY these audited figures ({unit_text}):
{table}
Winner ({direction}): {winner}
Sources: {sources}
Write 2-4 sentences: state each company's figure with its value, name the winner, and cite sources as [Source: id].
{caveat}"""


def _fmt(value: float, unit: str) -> str:
    return f"{value:.2f}%" if unit == "percent" else (f"{value:.2f}" if unit in ("ratio", "usd") else f"${value:,.0f} million")


async def writer_node(state: MAState) -> Dict[str, Any]:
    """Writes the final answer from the AUDITED values (recomputed by code), optionally after human approval.

    Reads:  plan, audit (expected values, passed, problems), findings (for sources), require_approval
    Writes: answer (structured), report (text), trace
    """
    plan, audit_result = state.get("plan", {}), state.get("audit") or {}
    values = audit_result.get("expected") or {}
    if not plan.get("tickers") or not values:
        report = "I could not answer this: " + "; ".join(audit_result.get("problems", ["no figures found"]))
        logger.warning("writer gave up: %s", report)
        return {"report": report, "answer": {"values": {}, "winner": None},
                "trace": [_event("writer", "gave up", report=report)]}
    unit = plan["unit"]
    pick = max if plan["direction"] == "highest" else min
    winner = pick(values, key=values.get)
    latest = _latest_findings(state)
    sources = sorted({f["source"].strip("[] ") for t in values for f in latest.get(t, {}).values()})
    answer = {"calculation": plan["calculation"], "unit": unit, "direction": plan["direction"],
              "values": {t: round(v, 2) for t, v in values.items()}, "winner": winner,
              "audit_passed": audit_result.get("passed", False), "sources": sources}

    if state.get("require_approval"):
        # Pauses the run here; the checkpointer saves the state and the API resumes it with Command(resume=...)
        logger.info("waiting for human approval of %s", answer["values"])
        decision = interrupt({"question": state["question"], "draft_answer": answer})
        if not (isinstance(decision, dict) and decision.get("approved")):
            note = decision.get("note", "") if isinstance(decision, dict) else ""
            logger.info("answer rejected by reviewer: %s", note or "(no note)")
            return {"answer": {**answer, "approved": False}, "report": f"Rejected by reviewer. {note}".strip(),
                    "trace": [_event("human", "rejected", note=note)]}

    table = "\n".join(f"- {NAMES.get(t, t)} ({t}): {_fmt(v, unit)}" for t, v in values.items())
    caveat = "" if audit_result.get("passed") else \
        "Say clearly that the audit did not fully pass: " + "; ".join(audit_result.get("problems", []))
    unit_text = {"percent": "percent", "ratio": "ratio", "usd": "dollars per share"}.get(unit, "$ millions")
    msg = await get_agent_llm().ainvoke(WRITER_PROMPT.format(
        question=state["question"], unit_text=unit_text, table=table, direction=plan["direction"],
        winner=f"{NAMES.get(winner, winner)} ({winner})", sources=", ".join(sources), caveat=caveat))
    report = str(msg.content).strip()
    # Deterministic safety net: every audited figure must be in the text the user reads
    if any(_fmt(v, unit).replace("$", "").replace(" million", "") not in report.replace("$", "") for v in values.values()):
        report += "\n\nAudited figures:\n" + table + f"\nWinner ({plan['direction']}): {NAMES.get(winner, winner)}"
    logger.info("answer: winner %s, values %s, audit %s", winner, answer["values"],
                "passed" if answer["audit_passed"] else "NOT passed")
    return {"answer": answer, "report": report, "trace": [_event("writer", "wrote", winner=winner)]}
