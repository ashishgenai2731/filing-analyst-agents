"""
MCP server: a company's reported figures from its own XBRL data (SEC companyfacts).

Every 10-K is filed with machine-readable XBRL facts, so "Apple's revenue in fiscal 2024" can be read
exactly instead of being parsed out of a table. Values are taken from the one filing that reports that
fiscal year (data/edgar/filings.jsonl), the same rule the golden set uses (src/eval/edgar_golden.py).

Run standalone:  python -m src.mcp.facts_server     (stdio transport; agents start it as a child process)
"""
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from mcp.server import FastMCP

from config.settings import settings
from src.eval.edgar_golden import CONCEPTS, _pick, as_filed, filing_values

# WARNING: this runs as a child process whose stderr shows in the agent's output; INFO would log every request
mcp = FastMCP("SEC-XBRL-Facts", log_level="WARNING")
logger = logging.getLogger(__name__)

# revenue, net_income, ... plus the prior-year figures the same 10-K reports (revenue_prior, net_income_prior)
METRICS = sorted(CONCEPTS) + ["net_income_prior", "revenue_prior"]


@lru_cache(maxsize=1)      # read once per process: (ticker, fiscal_year) -> filing row
def _manifest() -> Dict[tuple, Dict[str, Any]]:
    rows = [json.loads(line) for line in Path(settings.FILINGS_MANIFEST).read_text().splitlines() if line.strip()]
    return {(r["ticker"], int(r["fiscal_year"])): r for r in rows}


@lru_cache(maxsize=40)     # one company's facts JSON is ~4 MB; cache the 30 companies once read
def _facts(ticker: str) -> Dict[str, Any]:
    return json.loads((Path(settings.EDGAR_DIR) / "facts" / f"{ticker}.json").read_text())


# 41 of the 90 filings (Alphabet, Amazon, Meta, Netflix, pharma, ...) tag no GrossProfit. Their income
# statement still tags revenue and cost of revenue, so gross profit = revenue - cost of revenue.
COST_OF_REVENUE = ["CostOfRevenue", "CostOfGoodsAndServicesSold",
                   "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"]


def _derived_gross_profit(facts: Dict[str, Any], filing: Dict[str, Any], revenue: float):
    gaap = facts["facts"].get("us-gaap", {})
    for concept in COST_OF_REVENUE:
        rows = [r for r in gaap.get(concept, {}).get("units", {}).get("USD", []) if r.get("accn") == filing["accession"]]
        cost = _pick(rows, filing["report_date"], False)
        if cost is not None:
            return revenue - cost, f"revenue - {concept} ({as_filed('revenue', cost)})"
    return None, None


def lookup(ticker: str, fiscal_year: int, metric: str) -> Dict[str, Any]:
    """Plain-Python core of the tool, so tests and other code can call it without MCP."""
    ticker, metric = ticker.upper().strip(), metric.strip()
    if metric not in METRICS:
        return {"error": f"unknown metric '{metric}'", "valid_metrics": METRICS}
    filing = _manifest().get((ticker, int(fiscal_year)))
    if filing is None:
        years = sorted(y for (t, y) in _manifest() if t == ticker)
        return {"error": f"no indexed 10-K for {ticker} fiscal {fiscal_year}", "available_years": years}
    values = filing_values(_facts(ticker), filing["accession"], filing["report_date"])
    derived = None
    if metric == "gross_profit" and metric not in values and "revenue" in values:
        value, derived = _derived_gross_profit(_facts(ticker), filing, values["revenue"])
        if value is not None:
            values[metric] = value
            logger.debug("%s FY%s gross_profit derived: %s", ticker, fiscal_year, derived)
    if metric not in values:
        return {"error": f"{metric} not reported in {ticker}'s fiscal {fiscal_year} XBRL facts"}
    unit = "usd_per_share" if metric == "eps_diluted" else "usd_millions"
    return {
        "ticker": ticker, "fiscal_year": int(fiscal_year), "metric": metric,
        "value": as_filed(metric.replace("_prior", ""), values[metric]),   # as printed in the filing, e.g. "391,035"
        "unit": unit, "period_end": filing["report_date"],
        "source": f"xbrl:{filing['accession']}",
        **({"derived": derived} if derived else {}),
    }


@mcp.tool()
def get_financial_metric(ticker: str, fiscal_year: int, metric: str) -> str:
    """Exact full-year figure a company reported in its 10-K XBRL data.

    metric: one of revenue, net_income, gross_profit, operating_income, rd (R&D expense),
    cfo (operating cash flow), total_assets, equity, current_assets, current_liabilities, eps_diluted,
    revenue_prior / net_income_prior (the previous fiscal year, as reported in this year's 10-K).
    Returns JSON with the value in $ millions (EPS in dollars per share) and its source filing.
    """
    return json.dumps(lookup(ticker, fiscal_year, metric))


@mcp.tool()
def list_metrics() -> List[str]:
    """Names of the metrics get_financial_metric can return."""
    return METRICS


if __name__ == "__main__":
    mcp.run(transport="stdio")
