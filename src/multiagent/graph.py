"""
The multi-agent graph.

    START → supervisor ──Send×N──► research (one per company, in parallel) ──► supervisor
                      ├──────────► quant ──► auditor ──► supervisor
                      └──────────► writer ──► END

The supervisor routes with Command(goto=...): edges out of it are decided at run time, by its LLM
choice checked by code guards. Every other edge is fixed. Hard caps: AGENT_MAX_HOPS supervisor
decisions, AGENT_MAX_RESEARCH_ROUNDS retries per company, and LangGraph's recursion limit.
"""
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from config.settings import settings
from src.multiagent.agents import auditor_node, quant_node, research_node, supervisor_node, writer_node
from src.multiagent.state import MAState


def build_multiagent_graph(checkpointer=None):
    """Compiles the graph. Pass a checkpointer (SQLite in the API, in-memory in tests) to save every step
    to a thread, which is what makes human approval (pause + resume) and resuming after a crash possible."""
    builder = StateGraph(MAState)
    # The supervisor routes with Command(goto=...), so its edges aren't declared below;
    # `destinations` only tells LangGraph (and graph drawings) where it can go
    builder.add_node("supervisor", supervisor_node, destinations=("research", "quant", "writer"))
    builder.add_node("research", research_node)
    builder.add_node("quant", quant_node)
    builder.add_node("auditor", auditor_node)
    builder.add_node("writer", writer_node)

    # Fixed edges: every worker reports back; parallel research branches all finish before the
    # supervisor runs again (they share one LangGraph super-step)
    builder.add_edge(START, "supervisor")
    builder.add_edge("research", "supervisor")
    builder.add_edge("quant", "auditor")        # every calculation is audited, never skipped
    builder.add_edge("auditor", "supervisor")
    builder.add_edge("writer", END)
    return builder.compile(checkpointer=checkpointer)


def initial_state(question: str, mode: str = "full", require_approval: bool = False) -> Dict[str, Any]:
    """A fresh run. mode="text" removes the XBRL tool (the ablation); require_approval pauses before the answer."""
    return {"question": question, "mode": mode, "require_approval": require_approval,
            "findings": [], "tool_log": [], "trace": [], "hops": 0, "guard_overrides": 0, "llm_decisions": 0}


def run_config(thread_id: Optional[str] = None) -> Dict[str, Any]:
    """LangGraph run config: the super-step cap, plus the thread id when a checkpointer is used."""
    config: Dict[str, Any] = {"recursion_limit": settings.AGENT_RECURSION_LIMIT}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    return config
