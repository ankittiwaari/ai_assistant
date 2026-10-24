import logging
import logging.config
from contextvars import ContextVar
from uuid import uuid4

from app.infra.config import LOG_LEVEL

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    trace_id = uuid4().hex
    trace_id_var.set(trace_id)
    return trace_id


class TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        return True


def setup_logging(level: str = LOG_LEVEL) -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"trace_id": {"()": TraceIdFilter}},
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)-8s [%(trace_id)s] %(name)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["trace_id"],
                }
            },
            "root": {"level": level, "handlers": ["console"]},
            "loggers": {
                "httpx": {"level": "WARNING"},
                "openai": {"level": "WARNING"},
            },
        }
    )
