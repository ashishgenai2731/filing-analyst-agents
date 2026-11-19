"""
Ask the multi-agent system one question from the terminal and log what every agent did.

    python -m scripts.ask_agents "Compare the gross margins of Apple, Microsoft and Alphabet in fiscal 2024"
    python -m scripts.ask_agents --mode text "..."     # research agents may only read the filing text
    LOG_LEVEL=DEBUG python -m scripts.ask_agents "..." # also every tool call
"""
import argparse
import asyncio
import json
import logging
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

from config.logging_config import run_id_var, setup_logging  # noqa: E402
from src.multiagent.graph import build_multiagent_graph, initial_state, run_config  # noqa: E402

logger = logging.getLogger("ask_agents")


async def main(question: str, mode: str):
    run_id_var.set(uuid.uuid4().hex[:8])
    graph = build_multiagent_graph()
    started = time.perf_counter()
    logger.info("question (mode=%s): %s", mode, question)
    final = await graph.ainvoke(initial_state(question, mode=mode), config=run_config())

    # The trace is the per-run record of every agent step; log it as one readable block at the end
    for step in final["trace"]:
        agent, action = step.pop("agent"), step.pop("action")
        logger.info("trace [%s] %s: %s", agent, action, json.dumps(step, default=str)[:400])
    logger.info("answer: %s", json.dumps(final.get("answer"), default=str))
    logger.info("report:\n%s", final.get("report", ""))
    logger.info("%.0fs | supervisor decisions: %s | needed the LLM: %s | guard overrides: %s",
                time.perf_counter() - started, final.get("hops"), final.get("llm_decisions"), final.get("guard_overrides"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["full", "text"], default="full")
    args = parser.parse_args()
    setup_logging()
    asyncio.run(main(args.question, args.mode))
