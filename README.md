# Filing Analyst Agents

A **multi-agent system** that answers cross-company financial questions over SEC 10-K filings with exact,
sourced numbers: *"Compare the gross margins of Apple, Microsoft and Alphabet in fiscal 2024. Which was highest?"*

A **supervisor** plans the work and routes it; **research agents** (one per company, in parallel) fetch the
figures through **MCP** tools; a **quant agent** computes in a sandbox; a deterministic **auditor** checks every
number; a **writer** answers, optionally after **human approval**. Built with **LangGraph**, **MCP**,
**Qwen 2.5 7B** on **Ollama**, **Pinecone** and **FastAPI**, over 90 10-K filings from 30 companies (98,355 chunks).

It extends [enterprise_doc_intelligence](https://github.com/ashishgenai2731/enterprise_doc_intelligence)
(a single-company RAG workflow), which is still included and served at `/api/v1`, so both can be compared on the same questions.

## Results

Same comparison questions for every system, generated from the companies' own XBRL data (no answer typed by hand),
scored by the same text scorer: **fully correct** = every company's figure and the winner are right in the answer text.

| System | Fully correct | Values right | Winner right | 2-company lookups | 3-company calculations | Avg time / question |
|---|---|---|---|---|---|---|
| v1 workflow (3 nodes, Mistral-7B) | **1/16 (6%)** | 12% | 31% | 12% | 0% | 209 s |
| Multi-agent, filing text only (ablation: no XBRL tool) | **2/16 (12%)** | 12% | 50% | 12% | 12% | 359 s |
| Multi-agent, first version (before planner guard) | **12/16 (75%)** | 75% | 81% | 100% | 50% | 151 s |
| **Multi-agent (final)** | **16/16 (100%)** | 100% | 100% | 100% | 100% | 78 s |
| **Multi-agent (final), held-out set**: 16 new questions | **15/16 (94%)** | 100% | 94% | 88% | 100% | 86 s |

**What each column means:** *System* = the setup; *Fully correct* = questions with every value and the winner right;
*Values right* / *Winner right* = each part on its own; *2-company lookups* / *3-company calculations* = accuracy by question
type (8 each); *Avg time* = wall time per question on an M4 (Ollama answers one request at a time).

* **The multi-agent system beats the workflow 16/16 vs 1/16** on the tuning set. The workflow searches once for all
  companies together and usually misses one company's figure.
* **Ablation: the XBRL tool matters most.** The same multi-agent system restricted to filing text scores 2/16 (12%).
  The architecture alone doesn't fix it; exact, labelled figures do.
* **Held-out check.** The planner guard was added after the first run (75%) showed the planner reading "operating margin"
  as "compare operating income". To check it isn't tuned to these 16 questions, 16 new questions (seed 23, no overlap,
  including net margin, which the tuning set never asks) scored **15/16 (94%)**. The one miss: the quant agent twice
  failed to run the sandbox; the auditor caught it, the structured answer was still right (values are recomputed by the
  auditor), but the written report hedged on the winner.
* Supervisor routing: about 3 decisions per question; the LLM was needed only when more than one move was valid
  (1 time in the final tuning run, 2 in the held-out run), and the guard overrode an invalid choice 1–2 times.

Results: `data/eval/comparison_*.json`, logs: `data/eval/logs/`, table: `python -m scripts.results_table`.


---

## Architecture

```
            question
               │
               ▼
        ┌─────────────┐   typed Plan: companies, fiscal year, calculation
        │ Supervisor  │◄────────────────────────────────────────────┐
        └──────┬──────┘   Route: LLM proposes, code validates        │
     Send × N  │  Command(goto=...)                                   │
   ┌───────────┼───────────┬──────────────┐                          │
   ▼           ▼           ▼              │                          │
┌────────┐ ┌────────┐ ┌────────┐          │                          │
│Research│ │Research│ │Research│  parallel, one per company          │
│ AAPL   │ │ MSFT   │ │ GOOGL  │──────────┴──────────────────────────┤
└───┬────┘ └───┬────┘ └───┬────┘                                     │
    │ MCP: get_financial_metric (XBRL)   in-process: search_filings  │
    ▼                                                                │
┌────────┐   MCP: execute_python_calc (sandbox)   ┌──────────┐       │
│ Quant  │───────────────────────────────────────►│ Auditor  │───────┘
└────────┘                                        │ (no LLM) │  pass / redo research / redo quant
                                                  └──────────┘
        Supervisor ──► Writer (+ optional human approval: interrupt / resume) ──► answer
```

| Agent | Kind | Tools | Job |
|---|---|---|---|
| Supervisor | LLM, structured output | – | Reads the question into a typed `Plan`; after every phase decides which agent works next |
| Research ×N | LLM ReAct agent, parallel (`Send`) | `get_financial_metric` (MCP), `search_filings` | Fetches each company's figures; never calculates |
| Quant | LLM ReAct agent | `execute_python_calc` (MCP sandbox) | Writes and runs the Python for the calculation |
| Auditor | Deterministic | – | Completeness, grounding (every figure in a tool result), recomputation, winner check |
| Writer | LLM | – | Writes the answer from the audited table; optional `interrupt()` for approval |

**How the agents talk:** typed objects in one shared LangGraph state (`Plan`, `Finding`, `QuantResult`, audit
verdict), never free text. Parallel researchers append through `operator.add` reducers; the supervisor routes with
`Command(goto=...)` and fans out with `Send`. Every run is a thread saved by a SQLite checkpointer.

**Guardrails ("the LLM proposes, code validates"):** company names and named calculations are matched by code;
code derives each calculation's inputs; the supervisor's LLM is asked only when more than one move is valid, and an
invalid choice is overridden; hard caps on supervisor decisions (12), research retries (2 per company), agent
steps (14) and graph steps (60); LLM-written code runs in a separate MCP sandbox process (8 builtins, 10 s timeout).

Design notes, the model choice and a full case study: [`docs/MULTI_AGENT.md`](docs/MULTI_AGENT.md).

---

## Why these choices

* **Qwen 2.5 7B for the agents.** On one test question (Apple revenue growth, truth 2.02%), Mistral-7B wrote its
  tool calls as plain text, then answered 38.33%, then got 2.02 but never stopped. Qwen 2.5 7B, same code, answered
  2.02% in 3 steps with parallel tool calls and stopped by itself. The workflow keeps Mistral (`LLM_MODEL`).
* **Exact figures from XBRL, over MCP.** Every 10-K carries machine-readable XBRL facts, so `get_financial_metric`
  returns the exact reported value and its filing. 41 of the 90 filings tag no GrossProfit; the tool derives it as
  revenue − cost of revenue and says so.
* **A deterministic auditor.** Every check has an exact answer (is the figure in a tool result? does the
  arithmetic hold? is the winner right?), so it doesn't need an LLM's opinion.
* **Retrieval stays in-process.** `search_filings` needs the embedding model, reranker and BM25 index in memory
  (~2 GB, ~20 s to load); the light tools are MCP servers in their own processes.

---

## Quickstart

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct        # agents (make model)
ollama pull mistral                    # the v1 workflow
cp .env.example .env                   # PINECONE_API_KEY, SEC_USER_AGENT, LangSmith keys
python -m scripts.edgar_download       # 10-Ks + XBRL facts for config/companies.py
python -m scripts.ingest_edgar         # parse, chunk, embed -> Pinecone + data/edgar/corpus.jsonl
```

Ask from the terminal (prints what every agent did):

```bash
python -m scripts.ask_agents "Compare the gross margins of Apple, Microsoft and Alphabet in fiscal 2024. Which was highest?"
```

API (`uvicorn src.api.main:app`):

```bash
# run a question; the full agent trace comes back with the answer
curl -X POST localhost:8000/api/v2/agents/ask -H "Content-Type: application/json" \
     -d '{"question": "Which company had the higher net income in fiscal 2024: Adobe or Oracle?"}'

# pause for human approval before the answer is written, then approve by thread_id
curl -X POST localhost:8000/api/v2/agents/ask -H "Content-Type: application/json" \
     -d '{"question": "...", "require_approval": true}'          # -> status: awaiting_approval, thread_id
curl -X POST localhost:8000/api/v2/agents/resume -H "Content-Type: application/json" \
     -d '{"thread_id": "<id>", "approved": true}'

# stream one Server-Sent Event per agent step
curl -N -X POST localhost:8000/api/v2/agents/ask/stream -H "Content-Type: application/json" -d '{"question": "..."}'
```

The single-company workflow is still at `/api/v1/agent/analyze` (and its stream / semantic cache endpoints).

Tests and evaluation:

```bash
python -m pytest tests/unit -q          # offline, fake LLMs: tools, guards, auditor, parallel fan-out, approval
make eval-compare                       # builds the comparison set and runs all three systems
```

**Logging:** one setup in `config/logging_config.py` (used by the API, scripts and evals). Every line carries its run
id (the API's thread id), so one question can be followed across parallel agents:

```
12:04:31 INFO    [3f9c2a1e] src.multiagent.agents: route #3: research (valid: research, write; LLM chose it: ...)
12:04:31 WARNING [3f9c2a1e] src.multiagent.agents: audit failed: MSFT: gross_profit=999999 does not appear in any tool result
```

`LOG_LEVEL=DEBUG` adds every tool call; noisy libraries (httpx, MCP, Pinecone) are kept at WARNING, and the MCP servers
log at WARNING so their per-request lines don't drown the agents' output. Errors an agent recovers from are logged with
their traceback.

`make up` runs the API and Redis in Docker Compose; the API reaches Ollama on the host through
`OLLAMA_BASE_URL=http://host.docker.internal:11434`.

---

## Repository map

| Path | What |
|---|---|
| `src/multiagent/state.py` | Shared state, typed messages, formulas |
| `src/multiagent/agents.py` | Supervisor, research, quant, auditor, writer |
| `src/multiagent/graph.py` | The LangGraph graph (Send fan-out, Command routing, caps) |
| `src/multiagent/tools.py` | MCP tool discovery + in-process `search_filings` |
| `src/mcp/facts_server.py` | MCP server: exact XBRL figures (derives gross profit when untagged) |
| `src/mcp/server.py` | MCP server: Python sandbox |
| `src/api/agents_v2.py` | `/api/v2/agents`: ask, resume, stream, threads (SQLite checkpointer) |
| `src/eval/comparison_golden.py`, `compare_systems.py` | Comparison question set + head-to-head runner |
| `src/agents/`, `src/retrieval/`, `src/ingestion/`, `src/cache/` | The v1 workflow, hybrid retrieval, EDGAR ingestion, semantic cache |
| `docs/MULTI_AGENT.md` | Design, guardrails, case study |
| `docs/PROJECT_METRICS.md` | Ingestion, retrieval and v1 workflow metrics |

---

## Known limitations

* **Small local models.** Planning and routing run on a 7B model; the code guards exist because it misreads some
  questions. A larger model would need fewer guards, at higher cost and latency.
* **Ollama serves one request at a time** on this machine, so the "parallel" research agents overlap their tool
  calls but queue their LLM calls; latency is ~1.5–5 min per question on an M4.
* **The question sets are generated.** 2 × 16 templated comparison questions with XBRL answers; real users phrase
  questions differently, and 16 questions is a small sample.
* **Simple comparisons still run the quant agent.** For "which is higher?" questions the sandbox step adds a call and a
  failure point (the one held-out miss); a code path that ranks reported figures directly would remove it.
* **A real number on the wrong line passes grounding.** The auditor checks that a figure came from a tool, not that
  it is the right line item; exact XBRL figures are the main defence.
