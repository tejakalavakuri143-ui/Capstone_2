import logging
from datetime import datetime
from pathlib import Path

from fastapi import Body, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from agents.business_validation import BusinessValidationAgent
from agents.data_validation import DataValidationAgent
from agents.monitor_agent import InvoiceMonitor
from agents.Rag_agents.rag_pipeline import run_rag_pipeline
from agents.Rag_agents.indexing_agent import build_faiss_index

from database.invoice_db import (
    load_human_reviews,
    load_processed_files,
    load_reports,
    save_human_review,
    update_report_status,
)

from services.invoice_pipeline import (
    INCOMING_DIR,
    process_invoice_files,
    process_pending_invoices,
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format=
    "%(asctime)s %(levelname)s %(name)s %(message)s"

)

logger = logging.getLogger(__name__)

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Invoice Auditor API"
)

# ============================================================
# AGENTS
# ============================================================

data_agent = DataValidationAgent()

business_agent = BusinessValidationAgent()

monitor: InvoiceMonitor | None = None

# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global monitor

    INCOMING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(
        "Starting invoice monitor..."
    )

    if monitor is None:

        monitor = InvoiceMonitor(

            INCOMING_DIR,

            on_files_ready=
            process_invoice_files

        )

        monitor.start(
            process_existing=False
        )

        logger.info(
            "Invoice monitor started successfully."
        )

# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
def shutdown_event():

    global monitor

    if monitor is not None:

        logger.info(
            "Stopping invoice monitor..."
        )

        monitor.stop()

        logger.info(
            "Invoice monitor stopped."
        )

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {

        "message":
        "AI Invoice Auditor API is running",

        "incoming_dir":
        str(INCOMING_DIR),

        "auto_monitor":
        "enabled"

    }

# ============================================================
# DATA VALIDATION
# ============================================================

@app.post("/data_validation")
def data_validation(
    invoice_json: dict = Body(...)
):

    return data_agent.validate(
        invoice_json
    )

# ============================================================
# BUSINESS VALIDATION
# ============================================================

@app.post("/business_validation")
def business_validation(
    validation_result: dict = Body(...)
):

    return business_agent.validate(
        validation_result
    )

# ============================================================
# FULL VALIDATION
# ============================================================

@app.post("/validate_invoice")
def validate_invoice(
    invoice_json: dict = Body(...)
):

    data_result = data_agent.validate(
        invoice_json
    )

    if data_result.get(
        "status"
    ) != "PASSED":

        return {

            "data_validation":
            data_result,

            "final_status":
            "FAILED",

        }

    business_result = business_agent.validate(
        data_result
    )

    return {

        "data_validation":
        data_result,

        "business_validation":
        business_result,

        "final_status":
        business_result.get(
            "status",
            "UNKNOWN"
        )

    }

# ============================================================
# PROCESS PENDING
# ============================================================

@app.post("/process_pending")
def process_pending():

    return process_pending_invoices(
        INCOMING_DIR
    )

# ============================================================
# UPLOAD INVOICE
# ============================================================

@app.post("/upload_and_process")
async def upload_and_process(
    file: UploadFile = File(...)
):

    suffix = Path(
        file.filename or ""
    ).suffix.lower()

    if suffix not in {

        ".pdf",
        ".docx",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"

    }:

        return JSONResponse(

            status_code=400,

            content={

                "error":
                f"Unsupported invoice file type: {suffix}"

            },

        )

    target = (
        INCOMING_DIR /
        Path(file.filename).name
    )

    content = await file.read()

    target.write_bytes(content)

    logger.info(
        "Invoice uploaded: %s",
        target.name
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # DO NOT PROCESS HERE
    # Monitor agent will auto-trigger processing
    # --------------------------------------------------------

    return {

        "status":
        "uploaded",

        "message":
        "Invoice uploaded successfully. Auto-processing started.",

        "file":
        target.name

    }

# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
def chat(
    payload: dict = Body(...)
):

    question = payload.get(
        "question",
        ""
    )

    logger.info(
        "RAG Question: %s",
        question
    )

    # --------------------------------------------------------
    # Ensure FAISS exists
    # --------------------------------------------------------

    build_faiss_index(
        force_reload=False
    )

    return run_rag_pipeline(
        question
    )

# ============================================================
# REPORTS
# ============================================================

@app.get("/reports")
def reports():

    return {

        "reports":
        load_reports()

    }

# ============================================================
# PROCESSED FILES
# ============================================================

@app.get("/processed_files")
def processed_files():

    return {

        "processed_files":
        load_processed_files()

    }

# ============================================================
# HUMAN REVIEWS
# ============================================================

@app.get("/human_reviews")
def human_reviews():

    return {

        "reviews":
        load_human_reviews()

    }

# ============================================================
# SAVE HUMAN REVIEW
# ============================================================

@app.post("/human_reviews")
def create_human_review(
    payload: dict = Body(...)
):

    invoice_id = payload.get(
        "invoice_id"
    )

    decision = payload.get(
        "decision"
    )

    comments = payload.get(
        "comments",
        ""
    )

    if not invoice_id or not decision:

        return JSONResponse(

            status_code=400,

            content={

                "error":
                "invoice_id and decision are required"

            },

        )

    created_at = datetime.utcnow().isoformat() + "Z"

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
