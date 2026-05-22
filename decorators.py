from functools import wraps
from typing import Callable, Any

from integration.langfuse_tracing import guardrail_event, span, trace_context


def observe(*dargs, **dkwargs):
    def _decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_name = dkwargs.get("name") or func.__name__
            with trace_context(
                trace_name,
                input_data={"args": args, "kwargs": kwargs},
                metadata={"decorator": "observe"},
                tags=["invoice-auditor", "observe"],
            ):
                return func(*args, **kwargs)
        return wrapper
    # Allow usage as @observe() or @observe
    if len(dargs) == 1 and callable(dargs[0]):
        return _decorator(dargs[0])
    return _decorator


class CallbackHandler:
    def __init__(self, *args, **kwargs):
        self.trace_name = kwargs.get("trace_name", "langchain_run")

    def on_event(self, *args, **kwargs):
        guardrail_event(
            "langchain_event",
            passed=True,
            metadata={"args": repr(args)[:500], "kwargs": repr(kwargs)[:500]},
        )

    def on_chain_start(self, serialized, inputs, **kwargs):
        with span(
            "langchain.chain",
            input_data=inputs,
            metadata={"serialized": serialized},
        ):
            return None

    def on_llm_start(self, serialized, prompts, **kwargs):
        with span(
            "langchain.llm",
            input_data={"prompts": prompts},
            metadata={"serialized": serialized},
        ):
            return None

    def on_llm_error(self, error, **kwargs):
        guardrail_event(
            "langchain_llm_error",
            passed=False,
            metadata={"error": str(error)},
        )
