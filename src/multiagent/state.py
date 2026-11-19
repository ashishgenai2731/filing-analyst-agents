"""
Shared state, typed messages between agents, and the formulas the quant agent and auditor use.

Agents never pass free text to each other: the supervisor's plan, each researcher's findings and the
quant agent's results are typed objects in this state, so the next agent (and the auditor) can check them.
"""
import operator
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict

from pydantic import BaseModel, Field

# ---- Metrics the research agents can fetch (src/mcp/facts_server.py) ----
Metric = Literal["revenue", "revenue_prior", "net_income", "net_income_prior", "gross_profit",
                 "operating_income", "rd", "cfo", "total_assets", "equity",
                 "current_assets", "current_liabilities", "eps_diluted"]

# ---- What the question asks to compute, per company ----
# calculation -> (input metrics, formula, unit). "compare" = rank one reported metric as-is.
FORMULAS: Dict[str, Tuple[List[str], Callable[[Dict[str, float]], float], str]] = {
    "gross_margin": (["gross_profit", "revenue"], lambda v: v["gross_profit"] / v["revenue"] * 100, "percent"),
    "operating_margin": (["operating_income", "revenue"], lambda v: v["operating_income"] / v["revenue"] * 100, "percent"),
    "net_margin": (["net_income", "revenue"], lambda v: v["net_income"] / v["revenue"] * 100, "percent"),
    "rd_intensity": (["rd", "revenue"], lambda v: v["rd"] / v["revenue"] * 100, "percent"),
    "revenue_growth": (["revenue", "revenue_prior"], lambda v: (v["revenue"] / v["revenue_prior"] - 1) * 100, "percent"),
    "net_income_growth": (["net_income", "net_income_prior"],
                          lambda v: (v["net_income"] / v["net_income_prior"] - 1) * 100, "percent"),
    "current_ratio": (["current_assets", "current_liabilities"],
                      lambda v: v["current_assets"] / v["current_liabilities"], "ratio"),
}
Calculation = Literal["compare", "gross_margin", "operating_margin", "net_margin", "rd_intensity",
                      "revenue_growth", "net_income_growth", "current_ratio"]


def required_metrics(calculation: str, compare_metric: Optional[str]) -> List[str]:
    """The code, not the LLM, decides which inputs a calculation needs."""
    if calculation == "compare":
        return [compare_metric or "revenue"]
    return list(FORMULAS[calculation][0])


def unit_of(calculation: str, compare_metric: Optional[str]) -> str:
    if calculation == "compare":
        return "usd" if compare_metric == "eps_diluted" else "usd_millions"
    return FORMULAS[calculation][2]


# ---- Typed messages (LLM structured outputs) ----
class Plan(BaseModel):
    """Supervisor's reading of the question."""
    tickers: List[str] = Field(description="Ticker symbols of every company the question asks about")
    fiscal_year: int = Field(description="The fiscal year asked about, e.g. 2024")
    calculation: Calculation = Field(description="What to compute per company; 'compare' = rank one reported figure")
    compare_metric: Optional[Metric] = Field(default=None, description="Only for calculation='compare': which figure")
    direction: Literal["highest", "lowest"] = Field(default="highest", description="Which company the question wants")


class Route(BaseModel):
    """Supervisor's decision about which agent works next."""
    next: Literal["research", "quant", "write"] = Field(description="The agent that should work next")
    tickers: List[str] = Field(default_factory=list, description="For research: which companies to (re)research")
    reason: str = Field(description="One short sentence explaining the choice")


class Finding(BaseModel):
    metric: Metric
    value: float = Field(description="The number exactly as a tool returned it, in $ millions (EPS in dollars)")
    source: str = Field(description="The source id the tool gave, e.g. xbrl:0000320193-24-000123 or AAPL-FY2024-chunk-12")


class Findings(BaseModel):
    findings: List[Finding]


class TickerValue(BaseModel):
    ticker: str
    value: float


class QuantResult(BaseModel):
    values: List[TickerValue] = Field(description="The computed value for every company, from the sandbox output")
    winner: str = Field(description="Ticker of the company the question asks for (highest or lowest)")


# ---- Graph state ----
class MAState(TypedDict, total=False):
    question: str
    mode: str                                   # "full" (XBRL + text tools) or "text" (text search only)
    require_approval: bool                      # pause for a human before the final answer
    plan: Dict[str, Any]                        # Plan + "tasks": [{ticker, fiscal_year, metrics}]
    findings: Annotated[List[Dict[str, Any]], operator.add]   # appended by parallel research agents
    tool_log: Annotated[List[Dict[str, Any]], operator.add]   # every tool result, for grounding checks
    trace: Annotated[List[Dict[str, Any]], operator.add]      # who did what, in order
    quant: Dict[str, Any]                       # QuantResult + the sandbox code/output
    audit: Dict[str, Any]                       # passed, problems, redo_research, redo_quant
    research_rounds: Dict[str, int]             # ticker -> research attempts so far
    hops: int                                   # supervisor decisions so far
    guard_overrides: int                        # times code overrode the supervisor's LLM choice
    llm_decisions: int                          # supervisor routing decisions that needed the LLM
    answer: Dict[str, Any]                      # final structured answer
    report: str                                 # final text report
