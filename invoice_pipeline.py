import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, TypedDict

from langgraph.graph import END, StateGraph

from agents.extractor_agent import extractor_agent
from agents.translation_agent import translate_invoice
from agents.Rag_agents.indexing_agent import build_faiss_index
from database.invoice_db import (
    is_file_processed,
    mark_file_processed,
)
from integration.langfuse_tracing import span, trace_context, workflow_event
from tools.bussiness_validation_tool import business_validation_tool
from tools.data_validation_tool import data_validation_tool
from tools.reporting_tool import reporting_tool

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
INCOMING_DIR = BASE_DIR / "data" / "incoming"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".jpg", ".jpeg", ".png", ".webp"}


class InvoicePipelineState(TypedDict, total=False):
    files: list[str]
    skipped_files: list[str]
    extracted_data: list[dict]
    translated_data: list[dict]
    data_validation_result: dict
    final_validation_result: dict
    report_results: list[dict]
    index_result: dict | None


def ensure_runtime_dirs() -> None:
    for path in [
        INCOMING_DIR,
        BASE_DIR / "outputs" / "reports",
        BASE_DIR / "outputs" / "accepted",
        BASE_DIR / "outputs" / "rejected",
        BASE_DIR / "data" / "vector_db",
        BASE_DIR / "database",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def is_supported_invoice(path: str | Path) -> bool:
    return Path(path).suffix.lower() in ALLOWED_EXTENSIONS


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_pending_invoices(watch_dir: str | Path = INCOMING_DIR) -> list[str]:
    ensure_runtime_dirs()
    root = Path(watch_dir)
    root.mkdir(parents=True, exist_ok=True)
    pending: list[str] = []

    for path in sorted(root.iterdir()):
        if not path.is_file() or not is_supported_invoice(path):
            continue
        try:
            file_hash = file_sha256(path)
        except OSError:
            continue
        if not is_file_processed(file_hash):
            pending.append(str(path))

    return pending


def _failed_payload(file_name: str, error: str, bad_data: dict | None = None) -> dict:
    bad_data = bad_data or {}
    header = bad_data.get("header", {}) if isinstance(bad_data, dict) else {}
    return {
        "status": "FAILED",
        "stage": "PIPELINE",
        "invoice_id": header.get("invoice_no") or Path(file_name).stem,
        "vendor_id": header.get("vendor_id", "UNKNOWN"),
        "currency": header.get("currency", "UNKNOWN"),
        "translation_confidence": bad_data.get("translation_confidence", 0),
        "line_items": bad_data.get("line_items", []),
        "schema_errors": error,
        "validated_invoice": bad_data if isinstance(bad_data, dict) else {},
    }


def _record_processed(path: str, report: dict | None, error: str | None = None) -> None:
    file_hash = file_sha256(path)
    invoice_id = None
    status = None
    if report:
        invoice_id = report.get("invoice_id")
        status = report.get("status")
    mark_file_processed(
        file_hash=file_hash,
        file_path=str(Path(path).resolve()),
        file_name=Path(path).name,
        processed_at=datetime.utcnow().isoformat() + "Z",
        invoice_id=invoice_id,
        status=status,
        error=error,
    )

    workflow_event(
        "invoice.processed_status",
        input_data={"file_name": Path(path).name},
        output_data={
            "invoice_id": invoice_id,
            "status": status or "ERROR",
            "error": error,
        },
        metadata={
            "file_path": str(Path(path).resolve()),
            "stage": "record_processed",
        },
        level="ERROR" if error else "DEFAULT",
    )


def _process_translated(translated_results: list[dict]) -> tuple[dict, dict, list[dict]]:
    with span(
        "data_validation",
        input_data={"invoice_count": len(translated_results)},
    ):
        data_validation_result = data_validation_tool(translated_results)

    report_results: list[dict] = []
    business_validated_results: list[dict] = []

    for result in data_validation_result.get("validated_results", []):
        validation_payload = result.get("validation_result", {})
        fallback_invoice_id = Path(result.get("file_name") or "UNKNOWN").stem or "UNKNOWN"
        if validation_payload.get("invoice_id") in (None, "", "UNKNOWN"):
            validation_payload["invoice_id"] = fallback_invoice_id
        validated_invoice = validation_payload.get("validated_invoice")

        if validation_payload.get("status") != "PASSED":
            report_results.append(reporting_tool(validation_payload))
            continue

        with span(
            "business_validation",
            input_data={"invoice_id": validation_payload.get("invoice_id")},
        ):
            business_result = business_validation_tool(validation_payload)

        if business_result.get("invoice_id") in (None, "", "UNKNOWN"):
            business_result["invoice_id"] = fallback_invoice_id
        business_validated_results.append({
            "file_name": result.get("file_name"),
            "validation_result": business_result,
        })
        with span(
            "reporting",
            input_data={"invoice_id": business_result.get("invoice_id")},
        ):
            report_results.append(reporting_tool(business_result))

    for result in data_validation_result.get("failed_results", []):
        failed_payload = _failed_payload(
            result.get("file_name", "UNKNOWN"),
            result.get("error", "Unknown validation error"),
            result.get("bad_data", {}),
        )
        with span(
            "reporting_failed_invoice",
            input_data={"invoice_id": failed_payload.get("invoice_id")},
        ):
            report_results.append(reporting_tool(failed_payload))

    final_validation_result = {
        "validated_results": business_validated_results,
        "failed_results": data_validation_result.get("failed_results", []),
    }
    return data_validation_result, final_validation_result, report_results


def extraction_node(state: InvoicePipelineState) -> InvoicePipelineState:
    selected_files = state.get("files", [])
    with span(
        "extractor",
        input_data={"files": [Path(path).name for path in selected_files]},
    ):
        extracted = extractor_agent(selected_files)

    return {
        "extracted_data": extracted.get("extracted_data", []),
    }


def translation_node(state: InvoicePipelineState) -> InvoicePipelineState:
    translated_results: list[dict] = []

    for item in state.get("extracted_data", []):
        file_name = item.get("file_name", "UNKNOWN")
        try:
            with span(
                "translation",
                input_data={"file_name": file_name},
            ):
                structured = translate_invoice(item.get("file_text", ""))
            translated_results.append({
                "file_name": file_name,
                "structured_data": structured,
            })
        except Exception as exc:
            logger.exception("Translation failed for %s", file_name)
            translated_results.append({
                "file_name": file_name,
                "structured_data": _failed_payload(file_name, str(exc)),
            })

    return {
        "translated_data": translated_results,
    }


def validation_reporting_node(state: InvoicePipelineState) -> InvoicePipelineState:
    data_validation_result, final_validation_result, report_results = _process_translated(
        state.get("translated_data", [])
    )

    for report in report_results:
        workflow_event(
            "invoice.final_decision",
            input_data={"invoice_id": report.get("invoice_id")},
            output_data={
                "status": report.get("status"),
                "recommendation": report.get("recommendation"),
                "report_json": report.get("json_report_path"),
                "report_html": report.get("html_report_path"),
            },
            metadata={
                "vendor_id": report.get("vendor_id"),
                "currency": report.get("currency"),
            },
            level="WARNING"
            if report.get("status") in {"FAILED", "REJECTED", "MANUAL_REVIEW"}
            else "DEFAULT",
        )

    return {
        "data_validation_result": data_validation_result,
        "final_validation_result": final_validation_result,
        "report_results": report_results,
    }


def record_and_index_node(state: InvoicePipelineState) -> InvoicePipelineState:
    extracted_data = state.get("extracted_data", [])
    translated_results = state.get("translated_data", [])
    report_results = state.get("report_results", [])

    extraction_by_name = {item.get("file_name"): item for item in extracted_data}
    reports_by_invoice: dict[str, dict] = {}
    for report in report_results:
        if report.get("invoice_id"):
            reports_by_invoice[report["invoice_id"]] = report

    for path in state.get("files", []):
        file_name = Path(path).name
        matching_report = None
        translated = next(
            (item for item in translated_results if item.get("file_name") == file_name),
            None,
        )
        if translated:
            invoice_id = (
                translated.get("structured_data", {})
                .get("header", {})
                .get("invoice_no")
            )
            matching_report = reports_by_invoice.get(invoice_id)
            if matching_report is None:
                matching_report = reports_by_invoice.get(Path(file_name).stem)
        if file_name not in extraction_by_name:
            _record_processed(path, None, "No text could be extracted from this invoice.")
        else:
            _record_processed(path, matching_report)

    with span("rag_index_refresh", input_data={"force_reload": True}):
        index_result = build_faiss_index(force_reload=True)

    return {
        "index_result": index_result,
    }


invoice_workflow = StateGraph(InvoicePipelineState)
invoice_workflow.add_node("extractor", extraction_node)
invoice_workflow.add_node("translator", translation_node)
invoice_workflow.add_node("validator_reporter", validation_reporting_node)
invoice_workflow.add_node("recorder_indexer", record_and_index_node)
invoice_workflow.set_entry_point("extractor")
invoice_workflow.add_edge("extractor", "translator")
invoice_workflow.add_edge("translator", "validator_reporter")
invoice_workflow.add_edge("validator_reporter", "recorder_indexer")
invoice_workflow.add_edge("recorder_indexer", END)
invoice_graph = invoice_workflow.compile()


def process_invoice_files(files: Iterable[str | Path], skip_processed: bool = True) -> dict:
    ensure_runtime_dirs()
    files = [str(Path(f)) for f in files if is_supported_invoice(f) and Path(f).is_file()]
    selected_files: list[str] = []
    skipped_files: list[str] = []

    for path in files:
        try:
            file_hash = file_sha256(path)
        except OSError as exc:
            logger.warning("Cannot read invoice %s: %s", path, exc)
            continue
        if skip_processed and is_file_processed(file_hash):
            skipped_files.append(path)
        else:
            selected_files.append(path)

    if not selected_files:
        return {
            "files": [],
            "skipped_files": skipped_files,
            "extracted_data": [],
            "translated_data": [],
            "data_validation_result": {"validated_results": [], "failed_results": []},
            "final_validation_result": {"validated_results": [], "failed_results": []},
            "report_results": [],
            "index_result": None,
        }

    logger.info("Processing %s invoice file(s)", len(selected_files))

    with trace_context(
        "invoice_processing_workflow",
        input_data={
            "files": [Path(path).name for path in selected_files],
            "skip_processed": skip_processed,
        },
        metadata={"selected_file_count": len(selected_files)},
        tags=["invoice-auditor", "workflow"],
    ):
        result_state = invoice_graph.invoke({
            "files": selected_files,
            "skipped_files": skipped_files,
        })

    return {
        "files": selected_files,
        "skipped_files": skipped_files,
        "extracted_data": result_state.get("extracted_data", []),
        "translated_data": result_state.get("translated_data", []),
        "data_validation_result": result_state.get(
            "data_validation_result",
            {"validated_results": [], "failed_results": []},
        ),
        "final_validation_result": result_state.get(
            "final_validation_result",
            {"validated_results": [], "failed_results": []},
        ),
        "report_results": result_state.get("report_results", []),
        "index_result": result_state.get("index_result"),
    }


def process_pending_invoices(watch_dir: str | Path = INCOMING_DIR) -> dict:
    return process_invoice_files(discover_pending_invoices(watch_dir))
