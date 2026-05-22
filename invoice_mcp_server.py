import os
import sys
import time
import logging
import warnings

from pathlib import Path
from datetime import datetime

# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

if os.getenv("INVOICE_ALLOW_MODEL_DOWNLOADS", "0").lower() not in {"1", "true", "yes"}:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

warnings.filterwarnings(
    "ignore",
    message="Recommended: pip install sacremoses.*",
)

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated.*",
)

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================
# ROOT PATH
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT_DIR)

# ============================================================
# MCP
# ============================================================

from fastmcp import FastMCP

# ============================================================
# WATCHDOG
# ============================================================

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ============================================================
# RAG
# ============================================================

from agents.Rag_agents.rag_pipeline import (
    run_rag_pipeline
)

# ============================================================
# INDEXING
# ============================================================

from agents.Rag_agents.indexing_agent import (
    build_faiss_index
)

# ============================================================
# DATABASE
# ============================================================

from database.invoice_db import (

    load_human_reviews,
    load_processed_files,
    load_reports,

    save_human_review,
    update_report_status,

)

# ============================================================
# PIPELINE
# ============================================================

from services.invoice_pipeline import (

    process_invoice_files,
    process_pending_invoices,
    INCOMING_DIR

)

# ============================================================
# TOOLS
# ============================================================

from tools.extraction_tool import (
    extraction_tool
)

from tools.translation_tool import (
    translation_tool
)

from tools.data_validation_tool import (
    data_validation_tool
)

from tools.bussiness_validation_tool import (
    business_validation_tool
)

from tools.reporting_tool import (
    reporting_tool
)

# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP(
    "Invoice-Auditor"
)

# ============================================================
# SUPPORTED FILES
# ============================================================

SUPPORTED_EXTENSIONS = {

    ".pdf",
    ".docx",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"

}

# ============================================================
# AUTO PROCESS WRAPPER
# ============================================================

def auto_process_invoice(files):

    try:

        logger.info(
            "Starting automatic invoice processing..."
        )

        result = process_invoice_files(
            files,
            skip_processed=True
        )

        logger.info(
            "Invoice processing completed."
        )

        # ----------------------------------------------------
        # REFRESH FAISS
        # ----------------------------------------------------

        try:

            build_faiss_index(
                force_reload=True
            )

            logger.info(
                "FAISS index refreshed successfully."
            )

        except Exception as e:

            logger.exception(
                "FAISS refresh failed: %s",
                e
            )

        return result

    except Exception as e:

        logger.exception(
            "Auto invoice processing failed: %s",
            e
        )

        return {

            "status":
            "error",

            "error":
            str(e)

        }

# ============================================================
# WATCHDOG HANDLER
# ============================================================

class InvoiceHandler(
    FileSystemEventHandler
):

    def __init__(
        self,
        on_files_ready
    ):

        self.on_files_ready = (
            on_files_ready
        )

        self._processing = set()

    def _handle_path(
        self,
        path
    ):

        logger.info(
            "Detected file: %s",
            path
        )

        # ----------------------------------------------------
        # FILE TYPE FILTER
        # ----------------------------------------------------

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:

            logger.info(
                "Skipped unsupported file: %s",
                path.name
            )

            return

        if path in self._processing:
            return

        self._processing.add(path)

        try:

            logger.info(
                "New invoice detected: %s",
                path.name
            )

            # ------------------------------------------------
            # WAIT FOR FILE COPY COMPLETE
            # ------------------------------------------------

            time.sleep(3)

            # ------------------------------------------------
            # VERIFY FILE EXISTS
            # ------------------------------------------------

            if not path.exists():

                logger.error(
                    "File disappeared before processing: %s",
                    path
                )

                return

            logger.info(
                "Triggering automatic processing..."
            )

            self.on_files_ready(
                [path]
            )

        except Exception as e:

            logger.exception(
                "Invoice processing failed: %s",
                e
            )

        finally:

            self._processing.discard(path)

    def on_created(
        self,
        event
    ):

        if event.is_directory:
            return

        self._handle_path(
            Path(event.src_path)
        )

    def on_moved(
        self,
        event
    ):

        if event.is_directory:
            return

        self._handle_path(
            Path(event.dest_path)
        )

# ============================================================
# MONITOR
# ============================================================

class InvoiceMonitor:

    def __init__(
        self,
        watch_dir,
        on_files_ready
    ):

        self.watch_dir = Path(
            watch_dir
        )

        self.on_files_ready = (
            on_files_ready
        )

        self.observer = None

        self._running = False

    def start(
        self,
        process_existing=False
    ):

        if self._running:

            return {

                "status":
                "already_running",

                "watch_dir":
                str(self.watch_dir)

            }

        self.watch_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        logger.info(
            "Watching folder: %s",
            self.watch_dir
        )

        handler = InvoiceHandler(
            self.on_files_ready
        )

        self.observer = Observer()

        self.observer.schedule(

            handler,

            str(self.watch_dir),

            recursive=False

        )

        self.observer.start()

        self._running = True

        logger.info(
            "Invoice monitor started successfully."
        )

        # ----------------------------------------------------
        # OPTIONAL EXISTING FILE PROCESSING
        # ----------------------------------------------------

        if process_existing:

            existing = [

                path

                for path in sorted(
                    self.watch_dir.iterdir()
                )

                if (
                    path.is_file()
                    and
                    path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                )

            ]

            if existing:

                logger.info(
                    "Processing existing invoices..."
                )

                self.on_files_ready(
                    existing
                )

        return {

            "status":
            "running",

            "watch_dir":
            str(self.watch_dir),

            "process_existing":
            process_existing

        }

    def stop(self):

        if not self._running or self.observer is None:

            return {

                "status":
                "not_running"

            }

        self.observer.stop()

        self.observer.join()

        self.observer = None

        self._running = False

        logger.info(
            "Invoice monitor stopped."
        )

        return {

            "status":
            "stopped"

        }

# ============================================================
# START MONITOR INSTANCE
# ============================================================

logger.info(
    "INCOMING_DIR = %s",
    INCOMING_DIR
)

monitor = InvoiceMonitor(

    watch_dir=INCOMING_DIR,

    on_files_ready=auto_process_invoice

)

# ============================================================
# EXTRACTION TOOL
# ============================================================

@mcp.tool()
def extract_invoices(
    files: list
):

    logger.info(
        "Extracting invoices..."
    )

    return extraction_tool(
        files
    )

# ============================================================
# TRANSLATION TOOL
# ============================================================

@mcp.tool()
def translate_invoices(
    extracted_data: list
):

    logger.info(
        "Translating invoices..."
    )

    translated_results = []

    for item in extracted_data:

        structured = translation_tool(
            item["file_text"]
        )

        translated_results.append({

            "file_name":
            item["file_name"],

            "structured_data":
            structured

        })

    return {

        "translated_data":
        translated_results

    }

# ============================================================
# DATA VALIDATION
# ============================================================

@mcp.tool()
def data_validation(
    translated_data: list
):

    logger.info(
        "Running data validation..."
    )

    return data_validation_tool(
        translated_data
    )

# ============================================================
# BUSINESS VALIDATION
# ============================================================

@mcp.tool()
def business_validation(
    validation_data: dict
):

    logger.info(
        "Running business validation..."
    )

    return business_validation_tool(
        validation_data
    )

# ============================================================
# REPORT GENERATION
# ============================================================

@mcp.tool()
def generate_report(
    validation_result: dict
):

    logger.info(
        "Generating report..."
    )

    return reporting_tool(
        validation_result
    )

# ============================================================
# PROCESS INVOICES
# ============================================================

@mcp.tool()
def process_invoices(
    watch_dir: str = "data/incoming"
):

    logger.info(
        "Processing pending invoices..."
    )

    result = process_pending_invoices(
        watch_dir
    )

    try:

        build_faiss_index(
            force_reload=True
        )

        logger.info(
            "FAISS refresh completed."
        )

    except Exception as e:

        logger.exception(
            "FAISS refresh failed: %s",
            e
        )

    return result

@mcp.tool()
def process_pending_invoices_tool(
    watch_dir: str | None = None
):

    return process_invoices(
        watch_dir or str(INCOMING_DIR)
    )

@mcp.tool()
def process_invoice_file(
    file_path: str,
    skip_processed: bool = True
):

    path = Path(
        file_path
    ).expanduser()

    if not path.is_absolute():

        path = Path(
            ROOT_DIR
        ) / path

    if not path.exists() or not path.is_file():

        return {

            "status":
            "error",

            "error":
            f"Invoice file not found: {file_path}"

        }

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:

        return {

            "status":
            "error",

            "error":
            f"Unsupported invoice file type: {path.suffix}"

        }

    return process_invoice_files(
        [path],
        skip_processed=skip_processed
    )

@mcp.tool()
def start_invoice_monitor(
    process_existing: bool = True
):

    return monitor.start(
        process_existing=process_existing
    )

@mcp.tool()
def stop_invoice_monitor():

    return monitor.stop()

# ============================================================
# REPORTS
# ============================================================

@mcp.tool()
def list_reports():

    return {

        "reports":
        load_reports()

    }

@mcp.tool()
def get_report(
    invoice_id: str
):

    for report in load_reports():

        metadata = report.get(
            "report_metadata",
            {}
        )

        validation = report.get(
            "validation_result",
            {}
        )

        if invoice_id in {
            metadata.get("invoice_id"),
            validation.get("invoice_id"),
            report.get("invoice_id"),
        }:

            return {

                "status":
                "found",

                "report":
                report

            }

    return {

        "status":
        "not_found",

        "error":
        f"No report found for invoice_id: {invoice_id}"

    }

@mcp.tool()
def list_processed_files():

    return {

        "processed_files":
        load_processed_files()

    }

@mcp.tool()
def dashboard_data():

    return {

        "reports":
        load_reports(),

        "processed_files":
        load_processed_files()

    }

# ============================================================
# RAG QUERY
# ============================================================

@mcp.tool()
def rag_query(
    question: str
):

    logger.info(
        "RAG Question: %s",
        question
    )

    return run_rag_pipeline(
        question
    )

# ============================================================
# HUMAN REVIEW
# ============================================================

@mcp.tool()
def save_human_review_tool(

    invoice_id: str,

    decision: str,

    comments: str = ""

):

    if not invoice_id or not decision:

        return {

            "status":
            "error",

            "error":
            "invoice_id and decision are required"

        }

    created_at = (
        datetime.utcnow().isoformat()
        + "Z"
    )

    save_human_review(

        invoice_id=invoice_id,

        decision=decision,

        comments=comments,

        created_at=created_at,

    )

    updated = update_report_status(

        invoice_id=invoice_id,

        decision=decision,

        comments=comments,

        decided_at=created_at,

    )

    return {

        "status":
        "saved",

        "report_updated":
        updated

    }

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60, file=sys.stderr)

    print(" MCP SERVER - INVOICE AUDITOR ", file=sys.stderr)

    print("=" * 60, file=sys.stderr)

    monitor_enabled = (
        os.getenv(
            "INVOICE_MCP_AUTOMONITOR",
            "0"
        ).lower()
        in {"1", "true", "yes"}
    )

    try:

        if monitor_enabled:

            monitor.start(
                process_existing=False
            )

            logger.info(
                "Invoice monitor running in background."
            )

        mcp.run(
            transport="stdio",
            show_banner=False
        )

    finally:

        if monitor_enabled:

            monitor.stop()
