"""
Offline tests for the multi-agent system (no Ollama, no Pinecone): fake LLMs and fake tools.

Covers the XBRL tool, formulas, plan validation, the deterministic auditor, the supervisor's guards,
and full graph runs (parallel research fan-out, audit-driven retry, human approval with a checkpointer).
"""
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import src.multiagent.agents as agents
import src.multiagent.graph as graph_module
from src.eval.compare_systems import named_winner, score
from src.mcp.facts_server import lookup
from src.multiagent.state import FORMULAS, Plan, Route, required_metrics, unit_of


# ---------------------------------------------------------------- XBRL facts tool
def test_facts_lookup_reads_the_filing_value():
    row = lookup("AAPL", 2024, "revenue")
    assert row["value"] == "391,035" and row["source"] == "xbrl:0000320193-24-000123"
    assert lookup("AAPL", 2024, "revenue_prior")["value"] == "383,285"


def test_facts_lookup_derives_gross_profit_when_untagged():
    row = lookup("GOOGL", 2024, "gross_profit")          # Alphabet tags no GrossProfit
    assert row["value"] == "203,712" and row["derived"].startswith("revenue - CostOfRevenue")


def test_facts_lookup_errors_are_actionable():
    assert "available_years" in lookup("AAPL", 2019, "revenue")
    assert "valid_metrics" in lookup("AAPL", 2024, "ebitda")


# ---------------------------------------------------------------- formulas and plan
def test_code_decides_the_inputs_of_each_calculation():
    assert required_metrics("gross_margin", None) == ["gross_profit", "revenue"]
    assert required_metrics("compare", "cfo") == ["cfo"]
    assert unit_of("compare", "revenue") == "usd_millions" and unit_of("current_ratio", None) == "ratio"
    assert round(FORMULAS["revenue_growth"][1]({"revenue": 391035, "revenue_prior": 383285}), 2) == 2.02


def test_plan_tickers_come_from_the_question_not_the_llm():
    plan = Plan(tickers=["AAPL", "XXXX"], fiscal_year=2024, calculation="gross_margin")
    validated = agents._validate_plan(plan, "Compare the gross margins of Apple and Microsoft in fiscal 2024")
    assert validated["tickers"] == ["AAPL", "MSFT"]                 # MSFT added, unknown XXXX dropped
    assert [t["metrics"] for t in validated["tasks"]] == [["gross_profit", "revenue"]] * 2


# ---------------------------------------------------------------- auditor
def _state(findings, quant, tool_values, calculation="gross_margin", direction="highest"):
    metrics = required_metrics(calculation, None)
    tickers = sorted({f["ticker"] for f in findings} | set(tool_values))
    return {
        "plan": {"tickers": tickers, "metrics": metrics, "calculation": calculation, "direction": direction,
                 "unit": unit_of(calculation, None),
                 "tasks": [{"ticker": t, "fiscal_year": 2024, "metrics": metrics} for t in tickers]},
        "findings": findings, "quant": quant,
        "tool_log": [{"ticker": t, "tool": "get_financial_metric", "content": " ".join(f"{v:,.0f}" for v in vals)}
                     for t, vals in tool_values.items()],
    }


def _f(ticker, metric, value):
    return {"ticker": ticker, "fiscal_year": 2024, "metric": metric, "value": value, "source": "xbrl:x"}


GOOD_FINDINGS = [_f("AAPL", "gross_profit", 180683), _f("AAPL", "revenue", 391035),
                 _f("MSFT", "gross_profit", 171008), _f("MSFT", "revenue", 245122)]
TOOLS = {"AAPL": [180683, 391035], "MSFT": [171008, 245122]}
GOOD_QUANT = {"values": {"AAPL": 46.21, "MSFT": 69.76}, "winner": "MSFT", "sandbox_output": "SUCCESS: {...}"}


def test_audit_passes_grounded_correct_work():
    verdict = agents.audit(_state(GOOD_FINDINGS, GOOD_QUANT, TOOLS))
    assert verdict["passed"], verdict["problems"]
    assert round(verdict["expected"]["MSFT"], 2) == 69.76


def test_audit_catches_a_figure_no_tool_returned():
    findings = GOOD_FINDINGS[:2] + [_f("MSFT", "gross_profit", 999999), _f("MSFT", "revenue", 245122)]
    verdict = agents.audit(_state(findings, GOOD_QUANT, TOOLS))
    assert not verdict["passed"] and verdict["redo_research"] == ["MSFT"]
    assert any("ungrounded" in p for p in verdict["problems"])


def test_audit_recomputes_and_catches_wrong_maths_and_wrong_winner():
    bad = {"values": {"AAPL": 46.21, "MSFT": 50.0}, "winner": "AAPL", "sandbox_output": "SUCCESS: {...}"}
    verdict = agents.audit(_state(GOOD_FINDINGS, bad, TOOLS))
    assert verdict["redo_quant"] and not verdict["redo_research"]
    assert any("inputs give 69.76" in p for p in verdict["problems"])
    assert any("winner AAPL" in p for p in verdict["problems"])


def test_audit_requires_the_sandbox_to_have_run():
    verdict = agents.audit(_state(GOOD_FINDINGS, {**GOOD_QUANT, "sandbox_output": ""}, TOOLS))
    assert verdict["redo_quant"] and any("sandbox" in p for p in verdict["problems"])


def test_xbrl_findings_are_read_without_an_llm():
    log = [{"tool": "get_financial_metric", "content": json.dumps(lookup("AAPL", 2024, "revenue"))},
           {"tool": "get_financial_metric", "content": json.dumps(lookup("AAPL", 2024, "ebitda"))}]
    found = agents._xbrl_findings(log, "AAPL", 2024, ["revenue", "gross_profit"])
    assert found == [{"ticker": "AAPL", "fiscal_year": 2024, "metric": "revenue", "value": 391035.0,
                      "source": "xbrl:0000320193-24-000123"}]


# ---------------------------------------------------------------- scorer
def test_named_winner_reads_the_winner_sentence():
    report = ("Apple reported 46.21%, Microsoft had a higher gross margin at 69.76%, and Alphabet 58.20%. "
              "Microsoft emerged as the winner with the highest gross margin.")
    assert named_winner(report, ["AAPL", "MSFT", "GOOGL"]) == "MSFT"
    assert named_winner("Oracle had higher R&D (9,860) than Procter & Gamble (2,100).", ["ORCL", "PG"]) == "ORCL"
    # the winner is named last in the list: nearest company BEFORE "highest", not the first one in the sentence
    listed = "In fiscal 2025, Oracle had 17.18%, Cisco had 16.42%, and Qualcomm had the highest at 20.42%."
    assert named_winner(listed, ["ORCL", "CSCO", "QCOM"]) == "QCOM"
    assert named_winner("The highest was Qualcomm at 20.42%, ahead of Oracle.", ["ORCL", "QCOM"]) == "QCOM"
    item = {"unit": "usd_millions", "answers": {"ORCL": 9860, "PG": 2100}, "tickers": ["ORCL", "PG"], "winner": "ORCL"}
    assert score(item, "Oracle had higher R&D (9,860) than Procter & Gamble (2,100).")["correct"]


# ---------------------------------------------------------------- full graph with fakes
class FakeStructured:
    def __init__(self, schema, owner):
        self.schema, self.owner = schema, owner

    async def ainvoke(self, prompt):
        self.owner.calls.append(self.schema.__name__)
        if self.schema is Plan:
            return Plan(tickers=["AAPL", "MSFT"], fiscal_year=2024, calculation="gross_margin")
        if self.schema is Route:
            return Route(next="research", tickers=["MSFT"], reason="re-check the figure the audit rejected")
        raise AssertionError(f"unexpected schema {self.schema}")


class FakeLLM:
    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema):
        return FakeStructured(schema, self)

    async def ainvoke(self, prompt):
        self.calls.append("writer")

        class Msg:
            content = "Microsoft had the highest gross margin at 69.76%, ahead of Apple at 46.21%."
        return Msg()


def _fake_graph(monkeypatch, msft_first_value, checkpointer=None):
    llm = FakeLLM()
    attempts = {"MSFT": 0}

    async def fake_research(payload):
        t = payload["task"]["ticker"]
        good = {"AAPL": TOOLS["AAPL"], "MSFT": TOOLS["MSFT"]}[t]
        values = list(good)
        if t == "MSFT":
            attempts["MSFT"] += 1
            if attempts["MSFT"] == 1:
                values[0] = msft_first_value
        return {"findings": [_f(t, "gross_profit", values[0]), _f(t, "revenue", values[1])],
                "tool_log": [{"ticker": t, "tool": "get_financial_metric", "content": " ".join(f"{v:,.0f}" for v in good)}],
                "trace": [{"agent": f"research[{t}]", "action": "fetched"}]}

    async def fake_quant(state):
        latest = agents._latest_findings(state)
        values = {t: round(latest[t]["gross_profit"]["value"] / latest[t]["revenue"]["value"] * 100, 2)
                  for t in state["plan"]["tickers"]}
        return {"quant": {"values": values, "winner": max(values, key=values.get), "sandbox_output": "SUCCESS: ok"},
                "trace": [{"agent": "quant", "action": "computed"}]}

    monkeypatch.setattr(agents, "get_agent_llm", lambda: llm)
    monkeypatch.setattr(graph_module, "research_node", fake_research)
    monkeypatch.setattr(graph_module, "quant_node", fake_quant)
    return graph_module.build_multiagent_graph(checkpointer=checkpointer), llm, attempts


@pytest.mark.asyncio
async def test_graph_fans_out_in_parallel_and_answers(monkeypatch):
    graph, llm, attempts = _fake_graph(monkeypatch, msft_first_value=171008)
    final = await graph.ainvoke(graph_module.initial_state("Compare the gross margins of Apple and Microsoft in fiscal 2024"),
                                config=graph_module.run_config())
    agents_seen = [s["agent"] for s in final["trace"]]
    assert {"research[AAPL]", "research[MSFT]"} <= set(agents_seen[1:3])      # same super-step, both researchers
    assert final["answer"]["winner"] == "MSFT" and final["answer"]["audit_passed"]
    assert llm.calls == ["Plan", "writer"]          # every routing move was forced: no routing LLM call
    assert final["llm_decisions"] == 0 and attempts["MSFT"] == 1


@pytest.mark.asyncio
async def test_failed_audit_sends_only_the_bad_company_back_to_research(monkeypatch):
    graph, llm, attempts = _fake_graph(monkeypatch, msft_first_value=999999)    # ungrounded on the first try
    final = await graph.ainvoke(graph_module.initial_state("Compare the gross margins of Apple and Microsoft in fiscal 2024"),
                                config=graph_module.run_config())
    assert attempts["MSFT"] == 2                    # re-researched once
    assert final["research_rounds"] == {"AAPL": 1, "MSFT": 2}
    assert "Route" in llm.calls and final["llm_decisions"] == 1   # a real choice (research / quant / write) went to the LLM
    assert final["answer"]["audit_passed"] and final["answer"]["values"]["MSFT"] == 69.76


@pytest.mark.asyncio
async def test_human_approval_pauses_and_resumes_with_a_checkpointer(monkeypatch):
    graph, _, _ = _fake_graph(monkeypatch, msft_first_value=171008, checkpointer=InMemorySaver())
    config = graph_module.run_config("thread-1")
    await graph.ainvoke(graph_module.initial_state("Compare the gross margins of Apple and Microsoft in fiscal 2024",
                                                   require_approval=True), config=config)
    paused = await graph.aget_state(config)
    assert paused.next == ("writer",) and paused.tasks[0].interrupts[0].value["draft_answer"]["winner"] == "MSFT"
    final = await graph.ainvoke(Command(resume={"approved": True}), config=config)
    assert final["answer"]["winner"] == "MSFT" and "69.76" in final["report"]


@pytest.mark.asyncio
async def test_rejected_answer_is_not_published(monkeypatch):
    graph, _, _ = _fake_graph(monkeypatch, msft_first_value=171008, checkpointer=InMemorySaver())
    config = graph_module.run_config("thread-2")
    await graph.ainvoke(graph_module.initial_state("Compare Apple and Microsoft gross margins in fiscal 2024",
                                                   require_approval=True), config=config)
    final = await graph.ainvoke(Command(resume={"approved": False, "note": "check segment data"}), config=config)
    assert final["answer"]["approved"] is False and final["report"].startswith("Rejected by reviewer")


def test_calculation_named_in_the_question_overrides_the_planner():
    plan = Plan(tickers=["AAPL"], fiscal_year=2024, calculation="compare", compare_metric="operating_income")
    q = "Compare the operating margin (operating income / total revenue) of Apple, Oracle and Walmart in fiscal 2024."
    validated = agents._validate_plan(plan, q)
    assert validated["calculation"] == "operating_margin" and validated["llm_calculation"] == "compare"
    assert validated["metrics"] == ["operating_income", "revenue"]
    low = agents._validate_plan(Plan(tickers=[], fiscal_year=2024, calculation="gross_margin"),
                                "Which of Apple and Microsoft had the lowest gross margin in fiscal 2024?")
    assert low["direction"] == "lowest"


def test_a_value_from_another_fiscal_year_is_not_accepted():
    log = [{"tool": "get_financial_metric", "content": json.dumps(lookup("ADBE", 2023, "revenue"))}]
    assert agents._xbrl_findings(log, "ADBE", 2024, ["revenue", "revenue_prior"]) == []
