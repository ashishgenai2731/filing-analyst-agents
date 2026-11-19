"""
Tools the agents can use.

- get_financial_metric (MCP server src/mcp/facts_server.py): exact XBRL figures. Light, so it runs as
  an MCP server in its own process, started per call over stdio.
- execute_python_calc (MCP server src/mcp/server.py): the sandbox, unchanged from the workflow version.
- search_filings (in-process): hybrid search over the 10-K text. It stays in the API process because it
  needs the embedding model, reranker and BM25 index in memory (~2 GB, ~20 s to load); a per-call
  process would reload them every time.
"""
import logging
import sys
import threading
import time
from typing import Dict, List

from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.tools import get_retriever

logger = logging.getLogger(__name__)

# Each MCP server is started as a child process with the same Python interpreter (sys.executable),
# so it sees the same venv and the project root on its import path (python -m src.mcp...)
MCP_SERVERS = {
    "xbrl_facts": {"command": sys.executable, "args": ["-m", "src.mcp.facts_server"], "transport": "stdio"},
    "python_sandbox": {"command": sys.executable, "args": ["-m", "src.mcp.server"], "transport": "stdio"},
}

_mcp_tools: Dict[str, BaseTool] = {}   # tool name -> LangChain tool, filled once by mcp_tools()
_retrieval_lock = threading.Lock()   # parallel research agents share one retriever (models + BM25 cache)


async def mcp_tools() -> Dict[str, BaseTool]:
    """Discovers the tools of every MCP server once (tools/list), keyed by tool name."""
    if not _mcp_tools:
        # get_tools() runs initialize + tools/list against each server; every later tool call opens a
        # fresh session (a new child process), which keeps sandboxed code from sharing state across calls
        client = MultiServerMCPClient(MCP_SERVERS)
        for t in await client.get_tools():
            _mcp_tools[t.name] = t
        logger.info("MCP tools discovered: %s", sorted(_mcp_tools))
    return _mcp_tools


@tool
def search_filings(query: str, ticker: str, fiscal_year: int) -> str:
    """Searches the text of ONE company's 10-K for a fiscal year and returns the 5 best passages.

    Use it when get_financial_metric returns an error, or to read a figure from the filing itself.
    Table rows look like 'Total net sales — 2024: 391,035; 2023: 383,285' (values in $ millions).
    Each passage starts with its source id in square brackets.
    """
    # Always one filing: the agent already knows the company and year, so search only inside it
    filters = {"ticker": {"$in": [ticker.upper()]}, "fiscal_year": {"$in": [int(fiscal_year)]}}
    started = time.perf_counter()
    # The retriever's models and BM25 cache are shared; parallel research agents call this from threads
    with _retrieval_lock:
        chunks = get_retriever().retrieve(query=query, top_k=5, filters=filters)
    logger.debug("search_filings(%r, %s, %s): %d passages in %.1fs", query, ticker, fiscal_year, len(chunks),
                 time.perf_counter() - started)
    if not chunks:
        return f"No passages found for {ticker} fiscal {fiscal_year}."
    # 700 characters per passage keeps 5 passages well inside the agent's context window
    return "\n\n".join(f"[{c['id']}]\n{c['text'][:700]}" for c in chunks)


async def research_tools(mode: str) -> List[BaseTool]:
    """'full' = XBRL facts + text search; 'text' = text search only (the ablation)."""
    if mode == "text":
        return [search_filings]
    return [(await mcp_tools())["get_financial_metric"], search_filings]


async def sandbox_tool() -> BaseTool:
    return (await mcp_tools())["execute_python_calc"]
