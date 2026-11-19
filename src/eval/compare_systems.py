"""
Head-to-head on the comparison set: the original workflow vs the multi-agent system.

Every system is scored by the SAME text scorer on its final report:
  values_correct  every company's expected figure appears in the report (golden_eval.is_correct per company)
  winner_correct  the first sentence that compares ("higher", "highest", "lowest", ...) names the right company first
  correct         both
The multi-agent system also returns a structured answer; its accuracy is reported separately.

    python -m src.eval.compare_systems --system multiagent          # XBRL facts + text search
    python -m src.eval.compare_systems --system multiagent_text     # research agents may only read filing text
    python -m src.eval.compare_systems --system workflow            # the original 3-node workflow (Mistral)
"""
import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from config.companies import COMPANIES  # noqa: E402
from config.logging_config import run_id_var, setup_logging  # noqa: E402
from src.eval.golden_eval import is_correct  # noqa: E402

logger = logging.getLogger(__name__)

QA_PATH = Path("data/eval/comparison_qa.jsonl")
OUT_DIR = Path("data/eval")
TIMEOUT_SEC = 900
ALIASES = {c["ticker"]: [c["name"].lower()] + c["aliases"] for c in COMPANIES}
SUPERLATIVE = re.compile(r"\b(winner|wins|highest|lowest|largest|smallest|greatest|least|most|top|leads?|led)\b", re.I)
COMPARATIVE = re.compile(r"\b(higher|lower|larger|smaller|more|less|greater|exceed\w*|outperform\w*)\b", re.I)


def named_winner(report: str, tickers: List[str]) -> str:
    """The company the report names as the winner.

    Takes the first sentence with a winner word ("winner", "highest", "lowest", ...), or if none, the first
    comparative sentence ("higher than", ...), and returns the company named NEAREST BEFORE that word:
    "Oracle had 17.18%, Cisco 16.42%, and Qualcomm had the highest" -> Qualcomm (not Oracle, named first).
    If no company comes before the word ("The highest was Qualcomm"), the nearest one after it.
    """
    sentences = re.split(r"(?<=[.!?])\s+", report)
    for pattern in (SUPERLATIVE, COMPARATIVE):
        for sentence in sentences:
            keyword = pattern.search(sentence)
            if not keyword:
                continue
            low = sentence.lower()
            mentions = []   # (position, ticker) of every company mention in the sentence
            for t in tickers:
                mentions += [(m.start(), t) for a in ALIASES[t] for m in re.finditer(rf"(?<!\w){re.escape(a)}(?!\w)", low)]
                mentions += [(m.start(), t) for m in re.finditer(rf"\b{t}\b", sentence)]
            if not mentions:
                continue
            before = [m for m in mentions if m[0] < keyword.start()]
            return max(before)[1] if before else min(mentions)[1]
    return ""


def rescore(result_file: Path, qa_path: Path) -> None:
    """Re-applies the current scorer to saved reports (no agent re-run), then rewrites the summary."""
    items = {json.loads(line)["id"]: json.loads(line) for line in qa_path.read_text().splitlines() if line.strip()}
    data = json.loads(result_file.read_text())
    for row in data["results"]:
        row.update(score(items[row["id"]], row["report"]))
    data["summary"].update(summarize(data["results"]))
    result_file.write_text(json.dumps(data, indent=2))
    logger.info("rescored %s: %s", result_file.name, f"{data['summary']['accuracy']:.0%}")


def score(item: Dict[str, Any], report: str) -> Dict[str, Any]:
    values_ok = all(is_correct(report, v, item["unit"]) for v in item["answers"].values())
    winner_ok = named_winner(report, item["tickers"]) == item["winner"]
    return {"values_correct": values_ok, "winner_correct": winner_ok, "correct": values_ok and winner_ok}


async def run_workflow(question: str) -> Dict[str, Any]:
    from src.agents.graph import build_financial_agent_graph
    from src.agents.state import initial_agent_state
    state = await build_financial_agent_graph().ainvoke(initial_agent_state(question))
    return {"report": state.get("draft_report", ""), "extra": {"iterations": state.get("iteration_count", 0),
                                                              "audit_passed": state.get("audit_passed", False)}}


async def run_multiagent(question: str, mode: str) -> Dict[str, Any]:
    from src.multiagent.graph import build_multiagent_graph, initial_state, run_config
    state = await build_multiagent_graph().ainvoke(initial_state(question, mode=mode), config=run_config())
    return {"report": state.get("report", ""), "answer": state.get("answer", {}),
            "extra": {"hops": state.get("hops"), "llm_decisions": state.get("llm_decisions"),
                      "guard_overrides": state.get("guard_overrides"),
                      "audit_passed": (state.get("answer") or {}).get("audit_passed", False),
                      "research_rounds": state.get("research_rounds"),
                      "agents": [f"{s['agent']}:{s['action']}" for s in state.get("trace", [])]}}


def structured_ok(item: Dict[str, Any], answer: Dict[str, Any]) -> bool:
    values = (answer or {}).get("values", {})
    tol = 0.1 if item["unit"] == "percent" else None
    for t, expected in item["answers"].items():
        got = values.get(t)
        if got is None or abs(got - expected) > (tol if tol is not None else 0.005 * abs(expected)):
            return False
    return (answer or {}).get("winner") == item["winner"]


async def main(system: str, ids: List[str] = None, qa_path: Path = QA_PATH, tag: str = ""):
    items = [json.loads(line) for line in qa_path.read_text().splitlines() if line.strip()]
    if ids:
        items = [i for i in items if i["id"] in ids]
    results = []
    for item in items:
        run_id_var.set(item["id"])      # log lines of this question carry its id
        start = time.perf_counter()
        try:
            if system == "workflow":
                out = await asyncio.wait_for(run_workflow(item["question"]), TIMEOUT_SEC)
            else:
                mode = "text" if system == "multiagent_text" else "full"
                out = await asyncio.wait_for(run_multiagent(item["question"], mode), TIMEOUT_SEC)
        except Exception as err:   # a crash or timeout counts as a wrong answer; the run continues
            logger.warning("%s failed", item["id"], exc_info=True)
            out = {"report": f"ERROR: {type(err).__name__}: {err}", "extra": {}}
        row = {"id": item["id"], "type": item["type"], "question": item["question"], **score(item, out["report"]),
               "latency_sec": round(time.perf_counter() - start, 1), "report": out["report"], **out.get("extra", {})}
        if "answer" in out:
            row["structured_correct"] = structured_ok(item, out["answer"])
            row["answer"] = out["answer"]
        results.append(row)
        logger.info(f"{item['id']}: {'correct' if row['correct'] else 'WRONG  '} | values {'✅' if row['values_correct'] else '❌'} "
              f"winner {'✅' if row['winner_correct'] else '❌'}"
              + (f" | structured {'✅' if row.get('structured_correct') else '❌'}" if "structured_correct" in row else "")
              + f" | {row['latency_sec']}s")

    summary = {"system": system, **summarize(results)}
    if any("structured_correct" in r for r in results):
        summary["structured_accuracy"] = sum(r.get("structured_correct", False) for r in results) / len(results)
        summary["avg_supervisor_decisions"] = sum(r.get("hops") or 0 for r in results) / len(results)
        summary["llm_routing_decisions"] = sum(r.get("llm_decisions") or 0 for r in results)
        summary["guard_overrides"] = sum(r.get("guard_overrides") or 0 for r in results)
    (OUT_DIR / f"comparison_{system}{tag}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    logger.info("summary:\n%s", json.dumps(summary, indent=2))


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    return {"questions": n,
               "accuracy": sum(r["correct"] for r in results) / n,
               "values_correct": sum(r["values_correct"] for r in results) / n,
               "winner_correct": sum(r["winner_correct"] for r in results) / n,
               "accuracy_lookup": sum(r["correct"] for r in results if r["type"] == "compare_lookup") /
                                  max(1, sum(r["type"] == "compare_lookup" for r in results)),
               "accuracy_computed": sum(r["correct"] for r in results if r["type"] == "compare_computed") /
                                    max(1, sum(r["type"] == "compare_computed" for r in results)),
               "avg_latency_sec": sum(r["latency_sec"] for r in results) / n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["workflow", "multiagent", "multiagent_text"])
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--qa", type=Path, default=QA_PATH, help="question set (default: the tuning set)")
    parser.add_argument("--tag", default="", help="suffix for the result file, e.g. _holdout")
    parser.add_argument("--rescore", type=Path, help="re-score a saved result file with the current scorer")
    args = parser.parse_args()
    setup_logging()
    if args.rescore:
        rescore(args.rescore, args.qa)
    else:
        asyncio.run(main(args.system, args.ids, args.qa, args.tag))
