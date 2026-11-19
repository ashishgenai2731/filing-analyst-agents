"""
Builds the README results table from the saved comparison runs (data/eval/comparison_*.json),
so the numbers in the docs always come from the result files.

    python -m scripts.results_table
"""
import json
import logging
from pathlib import Path

from config.logging_config import setup_logging

logger = logging.getLogger("results_table")

RUNS = [  # (result file, label in the table)
    ("comparison_workflow.json", "v1 workflow (3 nodes, Mistral-7B)"),
    ("comparison_multiagent_text.json", "Multi-agent, filing text only (ablation: no XBRL tool)"),
    ("comparison_multiagent_v1.json", "Multi-agent, first version (before planner guard)"),
    ("comparison_multiagent.json", "**Multi-agent (final)**"),
    ("comparison_multiagent_holdout.json", "**Multi-agent (final), held-out set**: 16 new questions"),
]


def table() -> str:
    lines = ["| System | Fully correct | Values right | Winner right | 2-company lookups | 3-company calculations | Avg time / question |",
             "|---|---|---|---|---|---|---|"]
    for name, label in RUNS:
        path = Path("data/eval") / name
        if not path.exists():
            continue
        s = json.loads(path.read_text())["summary"]
        n = s["questions"]
        lines.append(f"| {label} | **{round(s['accuracy'] * n)}/{n} ({s['accuracy']:.0%})** | {s['values_correct']:.0%} | "
                     f"{s['winner_correct']:.0%} | {s['accuracy_lookup']:.0%} | {s['accuracy_computed']:.0%} | "
                     f"{s['avg_latency_sec']:.0f} s |")
    return "\n".join(lines)


if __name__ == "__main__":
    setup_logging()
    logger.info("\n%s", table())
