"""
One logging setup for the API, the scripts and the evals.

Every line carries the run it belongs to: `run_id` is a context variable set at the start of each
question (the API uses the LangGraph thread_id). asyncio copies context variables into every task it
creates, so the parallel research agents inherit their run's id automatically.

    12:04:31 INFO    [3f9c2a1e] src.multiagent.agents: research[AAPL] 14.2s: 2 tool calls, found {...}

Usage:  from config.logging_config import setup_logging, run_id_var
        setup_logging()                      # once, at program start
        run_id_var.set(thread_id)            # at the start of each run
"""
import contextvars
import logging
import logging.config
from typing import Optional

from config.settings import settings

# Which run a log line belongs to ("-" outside a run)
run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")

# Libraries that log every HTTP request / MCP message at INFO; WARNING keeps their real problems visible
QUIET_LOGGERS = ["httpx", "httpcore", "urllib3", "mcp", "langchain_mcp_adapters", "sentence_transformers",
                 "pinecone", "huggingface_hub", "filelock", "asyncio"]


class RunIdFilter(logging.Filter):
    """Copies the current run id onto every record, so the formatter can print it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = run_id_var.get()
        return True


def setup_logging(level: Optional[str] = None) -> None:
    """Configures the root logger once: one format, one handler (stderr), noisy libraries quieted."""
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,          # keep loggers created before this call (uvicorn, ...)
        "filters": {"run_id": {"()": RunIdFilter}},
        "formatters": {"standard": {
            "format": "%(asctime)s %(levelname)-7s [%(run_id)s] %(name)s: %(message)s",
            "datefmt": "%H:%M:%S",
        }},
        "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "standard",
                                 "filters": ["run_id"], "stream": "ext://sys.stderr"}},
        "root": {"level": (level or settings.LOG_LEVEL).upper(), "handlers": ["console"]},
        "loggers": {name: {"level": "WARNING"} for name in QUIET_LOGGERS},
    })
