
import asyncio
import importlib
import os
import logging
import json
import sys
import warnings

from pathlib import Path

from mcp import (

    ClientSession,

    StdioServerParameters

)

from mcp.client.stdio import (

    stdio_client

)

# ============================================================
# ENV SETTINGS
# ============================================================

os.environ[
    "TRANSFORMERS_VERBOSITY"
] = "error"

os.environ[
    "HF_HUB_DISABLE_PROGRESS_BARS"
] = "1"

os.environ[
    "TRANSFORMERS_NO_ADVISORY_WARNINGS"
] = "1"

if os.getenv(
    "INVOICE_ALLOW_MODEL_DOWNLOADS",
    "0"
).lower() not in {"1", "true", "yes"}:

    os.environ[
        "TRANSFORMERS_OFFLINE"
    ] = "1"

    os.environ[
        "HF_HUB_OFFLINE"
    ] = "1"

warnings.filterwarnings(
    "ignore",
    message="Recommended: pip install sacremoses.*",
)

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated.*",
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# SETTINGS
# ============================================================

SHOW_SOURCES = True

DEBUG = False

BASE_DIR = Path(__file__).resolve().parent

MCP_TRANSPORT = os.getenv(
    "INVOICE_MCP_TRANSPORT",
    "local"
).lower()

MCP_TOOL_NAMES = [
    "start_invoice_monitor",
    "stop_invoice_monitor",
    "process_pending_invoices_tool",
    "process_invoice_file",
    "list_reports",
    "get_report",
    "list_processed_files",
    "dashboard_data",
    "rag_query",
    "save_human_review_tool",
]

INCOMING_DIR = BASE_DIR / "data" / "incoming"

# ============================================================
# MCP RESULT HELPER
# ============================================================

def mcp_payload(
    result
):

    if isinstance(
        result,
        dict
    ):

        return result

    if hasattr(
        result,
        "structured_content"
    ) and result.structured_content:

        return result.structured_content

    if hasattr(
        result,
        "structuredContent"
    ):

        return result.structuredContent

    if hasattr(
        result,
        "content"
    ) and result.content:

        first = result.content[0]

        text = getattr(
            first,
            "text",
            None
        )

        if text:

            try:

                return json.loads(text)

            except json.JSONDecodeError:

                return {
                    "text":
                    text
                }

    return {}

# ============================================================
# LOCAL MCP SESSION
# ============================================================

class LocalMcpSession:

    def __init__(self):

        self.server = importlib.import_module(
            "mcp_servers.invoice_mcp_server"
        )

    async def initialize(self):

        return None

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict | None = None
    ):

        tool = getattr(
            self.server,
            tool_name
        )

        return tool(
            **(arguments or {})
        )

# ============================================================
# PRINT ANSWER
# ============================================================

def print_answer(
    response
):

    print("\n" + "=" * 60)

    print("ANSWER")

    print("=" * 60)

    answer = response.get(

        "answer",

        "No response generated."

    )

    print(answer)

    if SHOW_SOURCES:

        sources = response.get(

            "retrieved_sources",

            []

        )

        if len(sources) > 0:

            print("\nSources:")

            for s in set(sources):

                print(f"- {s}")

# ============================================================
# CHAT LOOP
# ============================================================

async def interactive_chat(
    session
):

    print("\n" + "=" * 60)

    print(" AI INVOICE AUDITOR ")

    print("=" * 60)

    print(
        "\nAsk invoice-related questions."
    )

    print(
        "Commands: /tools, /incoming, /monitor, /monitor-stop, /process, /process-file PATH, /reports, /report ID, /processed"
    )

    print(
        "Type 'exit' to quit."
    )

    while True:

        try:

            question = input(
                "\nAsk Question: "
            )

            question = question.strip()

            if not question:
                continue

            if question.lower() in [

                "exit",

                "quit"

            ]:

                print("\nGoodbye.")

                break

            if question == "/tools":

                print("\nAvailable MCP tools:")

                for tool_name in MCP_TOOL_NAMES:

                    print(
                        f"- {tool_name}"
                    )

                continue

            if question == "/incoming":

                files = sorted(
                    path.name
                    for path in INCOMING_DIR.iterdir()
                    if path.is_file()
                ) if INCOMING_DIR.exists() else []

                print(
                    f"\nWatching: {INCOMING_DIR}"
                )

                if files:

                    print(
                        "Files in incoming:"
                    )

                    for file_name in files:

                        print(
                            f"- {file_name}"
                        )

                else:

                    print(
                        "No files currently in incoming."
                    )

                continue

            if question == "/monitor":

                result = await session.call_tool(
                    "start_invoice_monitor",
                    {
                        "process_existing":
                        False
                    }
                )

                print(
                    json.dumps(
                        mcp_payload(result),
                        indent=2
                    )
                )

                continue

            if question == "/monitor-stop":

                result = await session.call_tool(
                    "stop_invoice_monitor",
                    {}
                )

                print(
                    json.dumps(
                        mcp_payload(result),
                        indent=2
                    )
                )

                continue

            if question == "/process":

                result = await session.call_tool(
                    "process_pending_invoices_tool",
                    {}
                )

                print(
                    json.dumps(
                        mcp_payload(result),
                        indent=2
                    )
                )

                continue

            if question.startswith("/process-file "):

                file_path = question.replace(
                    "/process-file ",
                    "",
                    1
                ).strip()

                result = await session.call_tool(
                    "process_invoice_file",
                    {
                        "file_path":
                        file_path
                    }
                )

                print(
                    json.dumps(
                        mcp_payload(result),
                        indent=2
                    )
                )

                continue

            if question == "/reports":

                result = await session.call_tool(
                    "list_reports",
                    {}
                )

                payload = mcp_payload(result)

                reports = payload.get(
                    "reports",
                    []
                )

                for report in reports:

                    metadata = report.get(
                        "report_metadata",
                        {}
                    )

                    validation = report.get(
                        "validation_result",
                        {}
                    )

                    print(
                        "- "
                        f"{metadata.get('invoice_id')}: "
                        f"{validation.get('status')} / "
                        f"{report.get('recommendation')}"
                    )

                if not reports:

                    print(
                        "No reports found."
                    )

                continue

            if question.startswith("/report "):

                invoice_id = question.replace(
                    "/report ",
                    "",
                    1
                ).strip()

                result = await session.call_tool(
                    "get_report",
                    {
                        "invoice_id":
                        invoice_id
                    }
                )

                print(
                    json.dumps(
                        mcp_payload(result),
                        indent=2
                    )
                )

                continue

            if question == "/processed":

                result = await session.call_tool(
                    "list_processed_files",
                    {}
                )

                print(
                    json.dumps(
                        mcp_payload(result),
                        indent=2
                    )
                )

                continue

            rag_result = await session.call_tool(

                "rag_query",

                {
                    "question":
                    question
                }

            )

            response = mcp_payload(
                rag_result
            )

            if response:

                print_answer(
                    response
                )

            else:

                print(
                    "\nNo valid response."
                )

        except KeyboardInterrupt:

            print("\nInterrupted.")

            break

        except Exception as e:

            if DEBUG:

                print(
                    f"\n[ERROR] {e}"
                )

            else:

                print(
                    "\nSomething went wrong."
                )

# ============================================================
# MAIN
# ============================================================

async def main():

    if MCP_TRANSPORT != "stdio":

        print(
            "Loading local MCP server tools..."
        )

        session = LocalMcpSession()

        monitor_result = await session.call_tool(
            "start_invoice_monitor",
            {
                "process_existing":
                False
            }
        )

        print(
            "Invoice monitor: "
            f"{mcp_payload(monitor_result).get('status')}"
        )

        print(
            "Loaded MCP tools: "
            f"{len(MCP_TOOL_NAMES)} "
            "(type /tools to list them)"
        )

        try:

            await interactive_chat(
                session
            )

        finally:

            await session.call_tool(
                "stop_invoice_monitor",
                {}
            )

        return

    server_params = (

        StdioServerParameters(

            command=sys.executable,

            args=[

                str(
                    BASE_DIR
                    / "mcp_servers"
                    / "invoice_mcp_server.py"
                )

            ],

            cwd=BASE_DIR,

            env={
                **os.environ,
                "INVOICE_MCP_AUTOMONITOR":
                "0",
            },

        )

    )

    async with stdio_client(

        server_params

    ) as (read, write):

        async with ClientSession(

            read,
            write

        ) as session:

            await session.initialize()

            await interactive_chat(
                session
            )

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())
