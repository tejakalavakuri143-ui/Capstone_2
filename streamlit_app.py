import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

import requests
import streamlit as st
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from streamlit_autorefresh import st_autorefresh


# ============================================================
# PATH SETUP
# ============================================================

UI_DIR = Path(__file__).parent
BASE_DIR = UI_DIR.parent
INCOMING_DIR = BASE_DIR / "data" / "incoming"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

API_URL = os.getenv(
    "INVOICE_API_URL",
    "http://localhost:8000",
).rstrip("/")

BACKEND = os.getenv(
    "INVOICE_UI_BACKEND",
    "api",
).lower()

MCP_TRANSPORT = os.getenv(
    "INVOICE_MCP_TRANSPORT",
    "local",
).lower()

_MCP_SERVER = None


# ============================================================
# HELPERS
# ============================================================

def api_get(path: str, fallback: dict) -> dict:
    try:
        response = requests.get(
            f"{API_URL}{path}",
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return fallback


def mcp_payload(result) -> dict:
    if isinstance(result, dict):
        return result

    if hasattr(result, "structured_content") and result.structured_content:
        return result.structured_content

    if hasattr(result, "structuredContent"):
        return result.structuredContent

    if hasattr(result, "content") and result.content:
        text = getattr(result.content[0], "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}

    return {}


def call_local_mcp_tool(tool_name: str, arguments: dict | None = None) -> dict:
    global _MCP_SERVER

    if _MCP_SERVER is None:
        _MCP_SERVER = importlib.import_module("mcp_servers.invoice_mcp_server")

    tool = getattr(_MCP_SERVER, tool_name)
    return mcp_payload(tool(**(arguments or {})))


async def call_mcp_tool_async(tool_name: str, arguments: dict | None = None) -> dict:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            str(BASE_DIR / "mcp_servers" / "invoice_mcp_server.py"),
        ],
        cwd=BASE_DIR,
        env={
            **os.environ,
            "INVOICE_MCP_AUTOMONITOR": "0",
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments or {})
            return mcp_payload(result)


def call_mcp_tool(tool_name: str, arguments: dict | None = None) -> dict:
    if MCP_TRANSPORT != "stdio":
        return call_local_mcp_tool(tool_name, arguments)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            call_mcp_tool_async(tool_name, arguments)
        )
    finally:
        loop.close()


def get_reports() -> list[dict]:
    if BACKEND == "mcp":
        data = call_mcp_tool("list_reports")
        return data.get("reports", [])

    data = api_get(
        "/reports",
        {"reports": []},
    )
    return data.get("reports", [])


def get_processed_files() -> list[dict]:
    if BACKEND == "mcp":
        data = call_mcp_tool("list_processed_files")
        return data.get("processed_files", [])

    data = api_get(
        "/processed_files",
        {"processed_files": []},
    )
    return data.get("processed_files", [])


@st.cache_data(ttl=10)
def get_dashboard_data() -> tuple[list[dict], list[dict]]:
    if BACKEND == "mcp":
        data = call_mcp_tool("dashboard_data")
        return (
            data.get("reports", []),
            data.get("processed_files", []),
        )

    return get_reports(), get_processed_files()


def invoice_id_for(report: dict) -> str:
    return (
        report.get("report_metadata", {}).get("invoice_id")
        or report.get("validation_result", {}).get("invoice_id")
        or "Unknown"
    )


def status_for(report: dict) -> str:
    return report.get("validation_result", {}).get("status", "UNKNOWN")


def recommendation_for(report: dict) -> str:
    return report.get("recommendation", "N/A")


def upload_invoice(uploaded_file) -> None:
    if BACKEND == "mcp":
        INCOMING_DIR.mkdir(parents=True, exist_ok=True)
        target = INCOMING_DIR / Path(uploaded_file.name).name
        target.write_bytes(uploaded_file.getvalue())

        result = call_mcp_tool(
            "process_invoice_file",
            {
                "file_path": str(target),
                "skip_processed": False,
            },
        )
        if result.get("status") == "error":
            st.error(result.get("error"))
        else:
            st.success("Invoice uploaded and processed through MCP.")
            get_dashboard_data.clear()
            st.rerun()
        return

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type,
        )
    }
    response = requests.post(
        f"{API_URL}/upload_and_process",
        files=files,
        timeout=30,
    )
    if response.status_code == 200:
        st.success("Invoice uploaded. Auto-processing has started.")
    else:
        st.error(response.text)


def process_pending_folder() -> None:
    if BACKEND == "mcp":
        result = call_mcp_tool("process_pending_invoices_tool")
        if result.get("status") == "error":
            st.error(result.get("error"))
        else:
            st.success("Pending invoice processing triggered through MCP.")
            get_dashboard_data.clear()
            st.rerun()
        return

    response = requests.post(
        f"{API_URL}/process_pending",
        timeout=120,
    )
    if response.status_code == 200:
        st.success("Pending invoice processing triggered.")
        get_dashboard_data.clear()
        st.rerun()
    else:
        st.error(response.text)


def ask_invoice_question(question: str) -> dict:
    if BACKEND == "mcp":
        return call_mcp_tool(
            "rag_query",
            {
                "question": question,
            },
        )

    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"question": question},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    return {
        "answer": "Backend connection failed.",
        "retrieved_sources": [],
    }


def submit_human_review(invoice_id: str, decision: str, comments: str) -> dict:
    if BACKEND == "mcp":
        return call_mcp_tool(
            "save_human_review_tool",
            {
                "invoice_id": invoice_id,
                "decision": decision,
                "comments": comments,
            },
        )

    response = requests.post(
        f"{API_URL}/human_reviews",
        json={
            "invoice_id": invoice_id,
            "decision": decision,
            "comments": comments,
        },
        timeout=10,
    )
    if response.status_code == 200:
        return response.json()

    return {
        "status": "error",
        "error": response.text,
    }


def ensure_mcp_monitor_started() -> None:
    if BACKEND != "mcp":
        return

    if st.session_state.get("mcp_monitor_started"):
        return

    result = call_mcp_tool(
        "start_invoice_monitor",
        {
            "process_existing": False,
        },
    )

    if result.get("status") in {"running", "already_running"}:
        st.session_state.mcp_monitor_started = True


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Invoice Auditor",
    page_icon="🧾",
    layout="wide",
)

st_autorefresh(
    interval=30000,
    key="invoice_dashboard_refresh",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

ensure_mcp_monitor_started()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("AI Invoice Auditor")
    st.caption("Upload multilingual invoices")
    st.caption(f"Backend: {BACKEND.upper()}")
    if BACKEND != "mcp":
        st.caption(f"API: {API_URL}")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Invoice",
        type=["pdf", "docx", "png", "jpg", "jpeg", "webp"],
    )

    if uploaded_file and st.button("Upload Invoice", width="stretch"):
        try:
            upload_invoice(uploaded_file)
        except Exception as e:
            st.error(str(e))

    if st.button("Process pending folder", width="stretch"):
        try:
            process_pending_folder()
        except Exception as e:
            st.error(str(e))

    if BACKEND == "mcp" and st.button("Refresh MCP data", width="stretch"):
        get_dashboard_data.clear()
        st.rerun()


# ============================================================
# TABS
# ============================================================

tab_reports, tab_rag, tab_feedback = st.tabs([
    "Reports Dashboard",
    "Invoice Q&A",
    "Human Review",
])


# ============================================================
# REPORTS DASHBOARD
# ============================================================

with tab_reports:
    st.title("AI Invoice Auditor")
    st.caption("AI-powered invoice processing and validation dashboard")

    reports, processed_files = get_dashboard_data()

    manual_reviews = sum(
        1 for report in reports
        if recommendation_for(report) == "Manual Review"
    )
    approved = sum(
        1 for report in reports
        if recommendation_for(report) in {"Approve", "Approve With Warning"}
    )
    rejected = sum(
        1 for report in reports
        if recommendation_for(report) == "Reject"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Invoices", len(reports))
    col2.metric("Approved", approved)
    col3.metric("Rejected", rejected)
    col4.metric("Manual Review", manual_reviews)

    st.divider()

    if processed_files:
        with st.expander("Processing status", expanded=True):
            st.dataframe(
                processed_files,
                width="stretch",
                hide_index=True,
            )

    if not reports:
        st.info("No reports available yet.")
    else:
        for report in reports:
            inv_id = invoice_id_for(report)
            status = status_for(report)
            recommendation = recommendation_for(report)
            generated_at = report.get("report_metadata", {}).get("generated_at", "")

            with st.expander(f"Invoice: {inv_id} | {recommendation}"):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Status:** {status}")
                c2.write(f"**Recommendation:** {recommendation}")
                c3.write(f"**Generated:** {generated_at}")
                st.json(report)


# ============================================================
# RAG CHAT
# ============================================================

with tab_rag:
    st.subheader("Invoice Q&A")
    st.write("Ask questions about processed invoices and reports.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask invoice-related questions...")

    if user_q:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_q,
        })

        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing invoices..."):
                result_state = ask_invoice_question(user_q)

            answer = result_state.get("answer", "No answer generated.")
            sources = result_state.get("retrieved_sources", [])
            trace_id = result_state.get("langfuse_trace_id")
            langfuse_status = result_state.get("langfuse_status")

            st.markdown(answer)

            if trace_id:
                st.caption(
                    f"Langfuse trace id: {trace_id}"
                    + (f" | status: {langfuse_status}" if langfuse_status else "")
                )

            if sources:
                st.markdown("### Sources")
                for source in sources:
                    st.write(f"- {source}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
        })


# ============================================================
# HUMAN REVIEW
# ============================================================

with tab_feedback:
    st.subheader("Human Review")
    st.write("Review invoices requiring manual approval.")

    reports, _ = get_dashboard_data()
    pending_reviews = [
        report for report in reports
        if recommendation_for(report) == "Manual Review"
    ]

    if not pending_reviews:
        st.success("No invoices pending manual review.")
    else:
        for review in pending_reviews:
            invoice_id = invoice_id_for(review)

            st.markdown(f"### Invoice: {invoice_id}")
            st.json(review)

            decision = st.radio(
                "Decision",
                ["Approve", "Reject"],
                horizontal=True,
                key=f"decision_{invoice_id}",
            )
            comments = st.text_area(
                "Comments",
                key=f"comments_{invoice_id}",
            )

            if st.button("Submit Review", key=f"submit_{invoice_id}"):
                result = submit_human_review(
                    invoice_id,
                    decision,
                    comments,
                )
                if result.get("status") == "error":
                    st.error(result.get("error"))
                else:
                    st.success("Review saved and report status updated.")
                    get_dashboard_data.clear()
                    st.rerun()
