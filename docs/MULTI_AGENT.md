# Multi-Agent Design

How the multi-agent system works, why each piece exists, and what went wrong on the way. Every number
here comes from a run on an Apple M4 (24 GB) with Qwen 2.5 7B served by Ollama.

---

## 1. Why multi-agent at all

The original system (`src/agents/`, still served at `/api/v1/agent/analyze`) is a **workflow**: a fixed
LangGraph path (retrieve → write code → audit → retry). It answers one company at a time well, but a
question like *"Compare the gross margins of Apple, Microsoft and Alphabet in fiscal 2024"* needs a
plan (which companies, which figures, which formula), work per company, and a check across all of them.

Before building agents, the agent loop itself was tested on one question (Apple revenue growth FY2023 →
FY2024, truth 2.02%), with the same two tools each time:

| Run | Model / method | Result |
|---|---|---|
| A | Mistral-7B, native tool calling | Wrote the tool calls as plain text, so nothing ran |
| B | Mistral-7B, ReAct text format | 38.33% ❌ (fetched only 2023, repeated an action) |
| C | B + one worked example | 2.02 ✅, then never stopped (hit the step cap) |
| D | **Qwen 2.5 7B, native tool calling** (same code as A) | **2.02 ✅ in 3 steps**, two parallel calls, stopped itself |

So the agents run on Qwen 2.5 (`AGENT_MODEL`); the workflow keeps Mistral (`LLM_MODEL`).

---

## 2. The agents

```
START → supervisor ──Send×N──► research (one per company, in parallel) ──► supervisor
                  ├──────────► quant ──► auditor ──► supervisor
                  └──────────► writer (+ optional human approval) ──► END
```

| Agent | Kind | Tools | Job |
|---|---|---|---|
| supervisor | LLM (structured output) | none | Reads the question into a typed `Plan`; after each phase chooses the next agent (`Route`) |
| research ×N | LLM ReAct agent, one per company, in parallel (`Send`) | `get_financial_metric` (MCP), `search_filings` | Fetches the figures; never calculates |
| quant | LLM ReAct agent | `execute_python_calc` (MCP sandbox) | Writes and runs the Python for the calculation, names the winner |
| auditor | **Deterministic** | none | Completeness, grounding, recomputation, winner check |
| writer | LLM | none | Writes the answer from the audited table; optional `interrupt()` for human approval |

**How they talk:** typed objects in one shared LangGraph state, never free text: `Plan`, `Finding`,
`QuantResult`, the audit verdict. Parallel researchers append to `findings` and `tool_log` through
`operator.add` reducers. The supervisor routes with `Command(goto=...)`; fan-out uses `Send`.

**Tools over MCP:** `src/mcp/facts_server.py` (XBRL figures) and `src/mcp/server.py` (sandbox) are MCP
servers discovered with `tools/list` through `langchain-mcp-adapters`. `search_filings` stays in-process
because it needs the embedding model, reranker and BM25 index in memory (~2 GB, ~20 s to load).

---

## 3. Guardrails: "the LLM proposes, code validates"

| Risk | Guard |
|---|---|
| The planner misreads a company | Company names are matched by code (`detect_tickers`); the LLM's tickers are used only if none match |
| The planner picks wrong inputs | Code derives the inputs from the calculation (`required_metrics`), the LLM only names the calculation |
| The supervisor picks an invalid next step | Code computes the valid moves; the LLM is asked only when there is more than one, and an invalid choice is overridden (counted as `guard_overrides`) |
| A researcher invents a figure | Auditor: every figure must appear in a tool result for that company |
| Wrong arithmetic or winner | Auditor recomputes every value from the grounded inputs and re-derives the winner |
| Loops | `AGENT_MAX_HOPS` = 12 supervisor decisions, `AGENT_MAX_RESEARCH_ROUNDS` = 2 retries per company, LangGraph `recursion_limit` = 60, 14 steps per research agent |
| Publishing a bad answer | `require_approval=True` pauses before the writer (`interrupt`), resumable by `thread_id` from a SQLite checkpointer |
| LLM-written code | Runs in the MCP sandbox: own process, 8 builtins, 10 s timeout |

The auditor is deliberately not an LLM: every check it makes has an exact answer.

---

## 4. Case study: the first real run

Question: *Compare the gross margins of Apple, Microsoft and Alphabet in fiscal 2024. Which was highest?*

**First version (519 s, one wrong value):**
1. Apple and Microsoft: figures from XBRL. Alphabet: `get_financial_metric` returned no gross profit,
   because **Alphabet doesn't tag GrossProfit** in XBRL, so the research agent calculated 203,712 itself.
2. The auditor rejected it: *"GOOGL: gross_profit=203712 does not appear in any tool result (ungrounded)"*.
3. The supervisor sent Alphabet back to research. This time the agent read **146,306** from the filing
   text: Alphabet's *cost of revenues*, not gross profit. It is a real number from a tool, so the
   grounding check passed it.
4. Final answer: Alphabet 41.80% ❌ (truth 58.20%); the winner (Microsoft) was still right.

**Fixes:**
- **At the tool:** 41 of the 90 filings tag no GrossProfit (Alphabet, Amazon, Meta, Netflix, Oracle,
  Qualcomm, the pharma companies). `get_financial_metric` now derives it as revenue − cost of revenue
  (both XBRL-tagged) and says so (`"derived": "revenue - CostOfRevenue (146,306)"`).
- **Speed:** figures from the XBRL tool are read straight from its JSON (no extra LLM call), and the
  supervisor's LLM is asked only when there is a real choice.

**After (166 s, all correct):** Apple 46.21%, Microsoft 69.76%, Alphabet 58.20%, winner Microsoft;
every figure sourced from XBRL; 3 supervisor decisions, none needing the LLM.

**Lesson:** the hardest error is a *real number on the wrong line*. A grounding check cannot catch it;
better tools (an exact, labelled figure) can.

---

## 5. Results

| System | Fully correct | Values right | Winner right | 2-company lookups | 3-company calculations | Avg time / question |
|---|---|---|---|---|---|---|
| v1 workflow (3 nodes, Mistral-7B) | **1/16 (6%)** | 12% | 31% | 12% | 0% | 209 s |
| Multi-agent, filing text only (ablation: no XBRL tool) | **2/16 (12%)** | 12% | 50% | 12% | 12% | 359 s |
| Multi-agent, first version (before planner guard) | **12/16 (75%)** | 75% | 81% | 100% | 50% | 151 s |
| **Multi-agent (final)** | **16/16 (100%)** | 100% | 100% | 100% | 100% | 78 s |
| **Multi-agent (final), held-out set**: 16 new questions | **15/16 (94%)** | 100% | 94% | 88% | 100% | 86 s |

* First run (75%): 3 of the 4 misses were the planner reading "operating/gross margin (… / total revenue)" as
  "compare operating income / gross profit"; 1 was a research agent fetching revenue for the wrong fiscal year.
* Fixes (code guards, not prompt tweaks): calculation phrases in the question override the planner
  (`CALC_PHRASES`), and a tool result is accepted only for the exact fiscal year asked.
* Final: 16/16 on the tuning set, 15/16 on a held-out set of 16 new questions. The scorer was also corrected
  (it took the first company in the winner sentence instead of the one nearest the winner word) and every saved
  run was re-scored with it; only the held-out result changed (14 → 15).

