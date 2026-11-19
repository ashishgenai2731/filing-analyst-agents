"""
Builds cross-company comparison questions from the single-company XBRL golden set.

Every expected value is an answer already in data/eval/edgar_golden_qa.jsonl (read from the companies'
own XBRL facts and confirmed to appear in the parsed filing text), so no answer is written by hand.

    python -m src.eval.comparison_golden                      # tuning set -> data/eval/comparison_qa.jsonl
    python -m src.eval.comparison_golden --seed 23 --out data/eval/comparison_holdout_qa.jsonl
                                       # held-out set: new questions the fixes were never tuned on
"""
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from config.companies import COMPANIES
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)

GOLDEN = Path("data/eval/edgar_golden_qa.jsonl")
OUT_PATH = Path("data/eval/comparison_qa.jsonl")
NAMES = {c["ticker"]: c["name"] for c in COMPANIES}

# metric id in the golden set -> (label, unit, companies per question)
LOOKUPS = {"revenue": "total revenue", "net_income": "net income", "operating_income": "operating income",
           "rd": "research and development expense", "cfo": "net cash provided by operating activities"}
COMPUTED = {"gross_margin": "gross margin (gross profit / total revenue)",
            "operating_margin": "operating margin (operating income / total revenue)",
            "net_margin": "net margin (net income / total revenue)",
            "rd_intensity": "R&D intensity (research and development expense / total revenue)",
            "revenue_growth": "revenue growth versus the prior fiscal year"}


def _join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def build(per_kind: int = 8, seed: int = 11, out_path: Path = OUT_PATH):
    rng = random.Random(seed)
    golden = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    by_key = defaultdict(dict)   # (metric, fiscal_year) -> ticker -> answer
    for q in golden:
        metric = q["id"].split("-", 2)[2]
        by_key[(metric, q["fiscal_year"])][q["ticker"]] = q["answer"]

    questions = []
    for kind, metrics, size in (("compare_lookup", LOOKUPS, 2), ("compare_computed", COMPUTED, 3)):
        pool = [(m, y) for (m, y), answers in sorted(by_key.items()) if m in metrics and y in (2024, 2025)
                and len(answers) >= size]
        for i in range(per_kind):
            metric, year = pool[rng.randrange(len(pool))]
            tickers = sorted(rng.sample(sorted(by_key[(metric, year)]), size))
            answers = {t: by_key[(metric, year)][t] for t in tickers}
            direction = rng.choice(["highest", "lowest"]) if kind == "compare_computed" else "highest"
            pick = max if direction == "highest" else min
            names = [NAMES[t] for t in tickers]
            if kind == "compare_lookup":
                text = (f"Which company had the higher {metrics[metric]} in fiscal {year}: {names[0]} or {names[1]}? "
                        f"Give both figures in $ millions.")
                unit = "usd_millions"
            else:
                text = (f"Compare the {metrics[metric]} of {_join(names)} in fiscal {year}. "
                        f"Which company had the {direction}? Give each in percent.")
                unit = "percent"
            questions.append({"id": f"CMP-{kind.split('_')[1][:4].upper()}-{i + 1:02d}", "type": kind,
                              "question": text, "metric": metric, "fiscal_year": year, "tickers": tickers,
                              "direction": direction, "unit": unit, "answers": answers,
                              "winner": pick(answers, key=answers.get)})
    out_path.write_text("".join(json.dumps(q) + "\n" for q in questions))
    logger.info("%d comparison questions -> %s", len(questions), out_path)
    return questions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    setup_logging()
    for q in build(seed=args.seed, out_path=args.out):
        logger.info("%s | %s | winner: %s %s", q["id"], q["question"], q["winner"], q["answers"])
