"""
/api/v2/agents: the multi-agent system over HTTP.

Every run is a LangGraph thread saved by a SQLite checkpointer (settings.CHECKPOINT_DB), so a run that
pauses for human approval can be resumed later, even after an API restart, with the same thread_id.

  POST /api/v2/agents/ask          run a question (optionally pausing for approval before the answer)
  POST /api/v2/agents/resume       approve or reject a paused answer
  POST /api/v2/agents/ask/stream   Server-Sent Events: one event per agent step
  GET  /api/v2/agents/threads/{id} current state of a thread
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

import aiosqlite
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from config.logging_config import run_id_var
from config.settings import settings
from src.multiagent.graph import build_multiagent_graph, initial_state, run_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v2/agents", tags=["multi-agent"])
_graph = None
_lock = asyncio.Lock()      # two first requests at once must not open two checkpointer connections


async def get_graph():
    """One compiled graph with one SQLite checkpointer, created on first use."""
    global _graph
    async with _lock:
        if _graph is None:
            # One long-lived connection for the API process; the saver creates its tables on first use
            conn = await aiosqlite.connect(settings.CHECKPOINT_DB)
            _graph = build_multiagent_graph(checkpointer=AsyncSqliteSaver(conn))
            logger.info("multi-agent graph ready, checkpoints in %s", settings.CHECKPOINT_DB)
    return _graph


class AskRequest(BaseModel):
    question: str = Field(..., examples=["Compare the gross margins of Apple, Microsoft and Alphabet in fiscal 2024."])
    mode: Literal["full", "text"] = "full"
    require_approval: bool = False
    thread_id: Optional[str] = None


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool
    note: str = ""


class AgentResponse(BaseModel):
    thread_id: str
    status: Literal["done", "awaiting_approval"]
    answer: Dict[str, Any] = Field(default_factory=dict)
    report: str = ""
    draft_answer: Optional[Dict[str, Any]] = None
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    supervisor_decisions: int = 0
    llm_routing_decisions: int = 0
    guard_overrides: int = 0


async def _respond(graph, thread_id: str) -> AgentResponse:
    """Builds the response from the thread's saved state, so ask, resume and threads all answer the same way."""
    snapshot = await graph.aget_state(run_config(thread_id))
    values = snapshot.values
    # A paused run has a pending task carrying the interrupt() payload (the draft answer to approve)
    pending = [i.value for task in snapshot.tasks for i in task.interrupts]
    return AgentResponse(
        thread_id=thread_id, status="awaiting_approval" if pending else "done",
        answer=values.get("answer", {}), report=values.get("report", ""),
        draft_answer=pending[0].get("draft_answer") if pending else None,
        trace=values.get("trace", []), supervisor_decisions=values.get("hops", 0),
        llm_routing_decisions=values.get("llm_decisions", 0), guard_overrides=values.get("guard_overrides", 0))


@router.post("/ask", response_model=AgentResponse)
async def ask(request: AskRequest):
    graph = await get_graph()
    thread_id = request.thread_id or str(uuid.uuid4())
    run_id_var.set(thread_id[:8])      # every log line of this run carries its thread id
    logger.info("ask (mode=%s, approval=%s): %s", request.mode, request.require_approval, request.question)
    started = time.perf_counter()
    try:
        await graph.ainvoke(initial_state(request.question, request.mode, request.require_approval),
                            config=run_config(thread_id))
    except Exception as err:
        logger.exception("multi-agent run failed")
        raise HTTPException(status_code=500, detail=f"multi-agent run failed: {err}")
    response = await _respond(graph, thread_id)
    logger.info("ask finished in %.0fs: status=%s", time.perf_counter() - started, response.status)
    return response


@router.post("/resume", response_model=AgentResponse)
async def resume(request: ResumeRequest):
    graph = await get_graph()
    run_id_var.set(request.thread_id[:8])
    snapshot = await graph.aget_state(run_config(request.thread_id))
    if not any(task.interrupts for task in snapshot.tasks):
        logger.warning("resume refused: thread %s is not waiting for approval", request.thread_id)
        raise HTTPException(status_code=409, detail="this thread is not waiting for approval")
    logger.info("resume: approved=%s note=%r", request.approved, request.note)
    # Command(resume=...) becomes the return value of interrupt() inside the writer
    await graph.ainvoke(Command(resume={"approved": request.approved, "note": request.note}),
                        config=run_config(request.thread_id))
    return await _respond(graph, request.thread_id)


@router.get("/threads/{thread_id}", response_model=AgentResponse)
async def thread(thread_id: str):
    return await _respond(await get_graph(), thread_id)


@router.post("/ask/stream")
async def ask_stream(request: AskRequest):
    graph = await get_graph()
    thread_id = request.thread_id or str(uuid.uuid4())

    run_id_var.set(thread_id[:8])
    logger.info("ask/stream: %s", request.question)

    async def events():
        # stream_mode="updates": one item per finished node, so the client sees each agent as it completes
        yield f"event: start\ndata: {json.dumps({'thread_id': thread_id})}\n\n"
        async for update in graph.astream(initial_state(request.question, request.mode, request.require_approval),
                                          config=run_config(thread_id), stream_mode="updates"):
            for node, delta in update.items():
                steps = (delta or {}).get("trace", []) if isinstance(delta, dict) else []
                yield f"event: {node}\ndata: {json.dumps(steps, default=str)}\n\n"
        final = await _respond(graph, thread_id)
        yield f"event: done\ndata: {final.model_dump_json()}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
