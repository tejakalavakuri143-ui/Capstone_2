import contextvars
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()
load_dotenv(BASE_DIR / "langfuse_api", override=False)

logger = logging.getLogger(__name__)

_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "langfuse_trace_id",
    default=None,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


def _shorten(value: Any, limit: int = 12000) -> Any:
    text = repr(value)
    if len(text) <= limit:
        return value
    return text[:limit] + "...[truncated]"


class LangfuseTracer:
    def __init__(self) -> None:
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
        self.enabled = bool(self.public_key and self.secret_key)
        self.last_status: str | None = None

    @property
    def ingestion_url(self) -> str:
        return f"{self.base_url}/api/public/ingestion"

    def emit(self, event_type: str, body: dict[str, Any]) -> bool:
        if not self.enabled:
            self.last_status = "disabled"
            return False

        payload = {
            "batch": [
                {
                    "id": _uuid(),
                    "timestamp": _now(),
                    "type": event_type,
                    "body": body,
                }
            ],
            "metadata": {
                "sdkIntegration": "capstone-invoice-auditor",
            },
        }

        try:
            response = requests.post(
                self.ingestion_url,
                json=payload,
                auth=(self.public_key, self.secret_key),
                timeout=5,
            )
            if response.status_code >= 300:
                self.last_status = f"failed:{response.status_code}"
                logger.warning(
                    "Langfuse ingestion failed: %s %s",
                    response.status_code,
                    response.text[:500],
                )
                return False
            self.last_status = f"sent:{event_type}"
            return True
        except Exception as exc:
            self.last_status = f"error:{exc}"
            logger.warning("Langfuse ingestion skipped: %s", exc)
            return False


tracer = LangfuseTracer()


def current_trace_id() -> str | None:
    return _current_trace_id.get()


@contextmanager
def trace_context(
    name: str,
    input_data: Any = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    trace_id = _uuid()
    token = _current_trace_id.set(trace_id)
    start = time.perf_counter()
    output_data: Any = None
    status_message: str | None = None
    level = "DEFAULT"

    try:
        yield trace_id
    except Exception as exc:
        level = "ERROR"
        status_message = str(exc)
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        body = {
            "id": trace_id,
            "name": name,
            "timestamp": _now(),
            "input": _shorten(input_data),
            "output": _shorten(output_data),
            "metadata": {
                **(metadata or {}),
                "duration_ms": duration_ms,
                "status_message": status_message,
            },
            "tags": tags or ["invoice-auditor"],
            "level": level,
        }
        tracer.emit("trace-create", body)
        _current_trace_id.reset(token)


@contextmanager
def span(
    name: str,
    input_data: Any = None,
    metadata: dict[str, Any] | None = None,
):
    trace_id = current_trace_id()
    if not trace_id:
        yield None
        return

    observation_id = _uuid()
    start_time = _now()
    start = time.perf_counter()
    tracer.emit(
        "span-create",
        {
            "id": observation_id,
            "traceId": trace_id,
            "name": name,
            "startTime": start_time,
            "input": _shorten(input_data),
            "metadata": metadata or {},
        },
    )

    output_data: Any = None
    level = "DEFAULT"
    status_message: str | None = None
    try:
        yield observation_id
    except Exception as exc:
        level = "ERROR"
        status_message = str(exc)
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        tracer.emit(
            "span-update",
            {
                "id": observation_id,
                "traceId": trace_id,
                "endTime": _now(),
                "output": _shorten(output_data),
                "metadata": {
                    **(metadata or {}),
                    "duration_ms": duration_ms,
                },
                "level": level,
                "statusMessage": status_message,
            },
        )


def guardrail_event(
    name: str,
    passed: bool,
    input_data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    existing_trace_id = current_trace_id()
    trace_id = existing_trace_id or _uuid()
    if not existing_trace_id:
        tracer.emit(
            "trace-create",
            {
                "id": trace_id,
                "name": f"guardrail.{name}",
                "timestamp": _now(),
                "input": _shorten(input_data),
                "metadata": {
                    **(metadata or {}),
                    "standalone_guardrail_event": True,
                },
                "tags": ["invoice-auditor", "guardrails"],
            },
        )
    tracer.emit(
        "event-create",
        {
            "id": _uuid(),
            "traceId": trace_id,
            "name": f"guardrail.{name}",
            "startTime": _now(),
            "input": _shorten(input_data),
            "output": {
                "passed": passed,
            },
            "metadata": metadata or {},
            "level": "DEFAULT" if passed else "WARNING",
            "statusMessage": "passed" if passed else "blocked_or_flagged",
        },
    )


def workflow_event(
    name: str,
    input_data: Any = None,
    output_data: Any = None,
    metadata: dict[str, Any] | None = None,
    level: str = "DEFAULT",
) -> None:
    trace_id = current_trace_id() or _uuid()
    if not current_trace_id():
        tracer.emit(
            "trace-create",
            {
                "id": trace_id,
                "name": name,
                "timestamp": _now(),
                "input": _shorten(input_data),
                "output": _shorten(output_data),
                "metadata": metadata or {},
                "tags": ["invoice-auditor", "workflow"],
                "level": level,
            },
        )
        return

    tracer.emit(
        "event-create",
        {
            "id": _uuid(),
            "traceId": trace_id,
            "name": name,
            "startTime": _now(),
            "input": _shorten(input_data),
            "output": _shorten(output_data),
            "metadata": metadata or {},
            "level": level,
        },
    )
