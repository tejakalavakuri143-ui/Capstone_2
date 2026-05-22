import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from integration.langfuse_tracing import guardrail_event, span, trace_context, tracer


def main() -> None:
    if not tracer.enabled:
        raise SystemExit(
            "Langfuse is not configured. Set LANGFUSE_PUBLIC_KEY, "
            "LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL or keep them in langfuse_api."
        )

    with trace_context(
        "manual_langfuse_trace_check",
        input_data={"source": "scripts/check_langfuse_trace.py"},
        metadata={"purpose": "connectivity_check"},
        tags=["invoice-auditor", "manual-check"],
    ) as trace_id:
        with span("manual_check_step", input_data={"step": "ping"}):
            guardrail_event(
                "manual_test_guardrail",
                passed=True,
                input_data={"sample": "safe invoice question"},
                metadata={"trace_id": trace_id},
            )

    print(f"Sent Langfuse check trace: {trace_id}")
    print(f"Open Langfuse traces: {tracer.base_url}/project")
    print("In the UI, search for trace name: manual_langfuse_trace_check")


if __name__ == "__main__":
    main()
