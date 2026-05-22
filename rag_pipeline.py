import json
import logging
import re
from database.invoice_db import load_reports

from integration.config import (

    RAG_TOP_K

)

from integration.guardrails import (

    ResponsibleAIGuardrails

)

from integration.langfuse_tracing import span, trace_context, tracer, workflow_event

from agents.Rag_agents.indexing_agent import (

    load_and_chunk_reports,

    build_faiss_index

)

from agents.Rag_agents.retrieval_agent import (

    embed_question,

    search_faiss

)

from agents.Rag_agents.augmentation_agent import (

    rerank_chunks,

    enrich_context

)

from agents.Rag_agents.generation_agent import (

    build_prompt,

    generate_answer

)

from agents.Rag_agents.reflection_agent import (

    compute_scores

)

# ---------------------------------------------------
# Logging
# ---------------------------------------------------

logging.basicConfig(
    level=logging.ERROR
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Debug Mode
# ---------------------------------------------------

DEBUG = False

# ---------------------------------------------------
# Small Talk Handler
# ---------------------------------------------------

def handle_small_talk(
    question: str
):

    greetings = [

        "hi",

        "hello",

        "hlo",

        "hey"

    ]

    question = question.lower().strip()

    if question in greetings:

        return (
            "Hello! "
            "Ask me anything about "
            "processed invoices, "
            "validation results, "
            "discrepancies, "
            "recommendations, "
            "or invoice reports."
        )

    return None

# ---------------------------------------------------
# Invoice Intent Detection
# ---------------------------------------------------

def is_invoice_query(
    question: str
) -> bool:

    question = question.lower()

    invoice_keywords = [

        "invoice",

        "processed",

        "flagged",

        "missing",

        "recommendation",

        "validation",

        "vendor",

        "currency",

        "review",

        "approved",

        "rejected",

        "status",

        "confidence",

        "discrepancy",

        "report"

    ]

    return any(

        k in question

        for k in invoice_keywords

    )


def answer_from_reports(question: str):
    reports = load_reports()
    if not reports:
        return None

    q = question.lower()

    def as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(value)]

    def reason_for(row, include_human_comment=True):
        reasons = []
        reasons.extend(row["missing"])
        reasons.extend(row["discrepancies"])
        if not reasons and row["summary"]:
            reasons.append(row["summary"])
        if include_human_comment and row["human_comment"]:
            reasons.append(f"Human comment: {row['human_comment']}")
        return "; ".join(reasons) if reasons else "No specific issue was recorded."

    def is_rejected(row):
        return (
            row["status"].upper() in {"REJECTED", "FAILED", "HUMAN_REJECTED"}
            or row["recommendation"].lower() == "reject"
        )

    def is_approved(row):
        return (
            row["status"].upper() in {"APPROVED", "PASSED", "HUMAN_APPROVED"}
            or row["recommendation"].lower() in {
                "approve",
                "approve with warning",
            }
        )

    def is_manual_review(row):
        return (
            row["status"].upper() in {"REVIEW_REQUIRED", "MANUAL_REVIEW"}
            or row["recommendation"].lower() == "manual review"
        )

    rows = []
    for report in reports:
        metadata = report.get("report_metadata", {})
        validation = report.get("validation_result", {})
        summary = report.get("invoice_summary", {})
        human_review = report.get("human_review", {})
        rows.append({
            "invoice_id": metadata.get("invoice_id", "UNKNOWN"),
            "generated_at": metadata.get("generated_at", ""),
            "status": validation.get("status", "UNKNOWN"),
            "recommendation": report.get("recommendation", "UNKNOWN"),
            "vendor_id": summary.get("vendor_id", "UNKNOWN"),
            "currency": summary.get("currency", "UNKNOWN"),
            "confidence": summary.get("translation_confidence", "UNKNOWN"),
            "summary": validation.get("summary", ""),
            "missing": as_list(validation.get("missing_fields", [])),
            "discrepancies": as_list(validation.get("discrepancy_summary", [])),
            "line_items": report.get("line_items", []),
            "human_decision": human_review.get("decision", ""),
            "human_comment": human_review.get("comments", ""),
            "human_decided_at": human_review.get("decided_at", ""),
        })

    def md_table(headers, data_rows):
        if not data_rows:
            return ""
        header = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        body = [
            "| " + " | ".join(str(cell) if cell not in (None, "") else "-" for cell in row) + " |"
            for row in data_rows
        ]
        return "\n".join([header, separator, *body])

    def status_table(title, selected_rows):
        if not selected_rows:
            return f"{title}\n\nNo invoices found."
        return (
            f"{title}\n\n"
            + md_table(
                ["Invoice", "Status", "Recommendation", "Vendor", "Reason"],
                [
                    [
                        r["invoice_id"],
                        r["status"],
                        r["recommendation"],
                        r["vendor_id"],
                        reason_for(r),
                    ]
                    for r in selected_rows
                ],
            )
        )

    def normalize_id(value):
        return re.sub(r"[^a-z0-9]", "", value.lower())

    def find_invoice_rows():
        normalized_q = normalize_id(q)
        number_tokens = re.findall(r"\d+", q)
        matches = []
        for row in rows:
            invoice_id = row["invoice_id"]
            normalized_id = normalize_id(invoice_id)
            if normalized_id and normalized_id in normalized_q:
                matches.append(row)
                continue
            if normalized_id and normalized_q in normalized_id and len(normalized_q) >= 5:
                matches.append(row)
                continue
            if any(token and token in normalized_id for token in number_tokens):
                matches.append(row)
        return matches

    specific_invoice_rows = find_invoice_rows()

    if specific_invoice_rows and (
        "detail" in q
        or "details" in q
        or "status" in q
        or "reason" in q
        or "comment" in q
        or "invoice" in q
    ):
        details = []
        for row in specific_invoice_rows:
            line_items = row["line_items"] or []
            line_table = md_table(
                ["Item Code", "Description", "Qty", "Unit Price", "Total"],
                [
                    [
                        item.get("item_code", "-"),
                        item.get("description", "-"),
                        item.get("qty", "-"),
                        item.get("unit_price", "-"),
                        item.get("total", "-"),
                    ]
                    for item in line_items
                ],
            )
            details.append(
                f"### {row['invoice_id']}\n\n"
                + md_table(
                    ["Field", "Value"],
                    [
                        ["Status", row["status"]],
                        ["Recommendation", row["recommendation"]],
                        ["Vendor", row["vendor_id"]],
                        ["Currency", row["currency"]],
                        ["Validation reason", reason_for(row, include_human_comment=False)],
                        ["Human decision", row["human_decision"] or "-"],
                        ["Human comment", row["human_comment"] or "-"],
                    ],
                )
                + "\n\nLine items\n\n"
                + (line_table if line_table else "No line items recorded.")
            )
        return "\n\n".join(details)

    if "human" in q and ("comment" in q or "reason" in q or "reject" in q):
        human_rejected = [
            r for r in rows
            if r["status"].upper() == "HUMAN_REJECTED"
            or r["human_decision"].lower() == "reject"
        ]
        if not human_rejected:
            return "No human-rejected invoices were found."
        return (
            "Human rejected invoices\n\n"
            + md_table(
                ["Invoice", "Validation summary", "Human comment"],
                [
                    [
                        r["invoice_id"],
                        reason_for(r, include_human_comment=False),
                        r["human_comment"] or "-",
                    ]
                    for r in human_rejected
                ],
            )
        )

    approved_rows = [r for r in rows if is_approved(r)]
    rejected_rows = [r for r in rows if is_rejected(r)]
    manual_rows = [r for r in rows if is_manual_review(r)]

    is_count_question = "count" in q or "how many" in q or "number of" in q
    asks_approved = "approved" in q or "approve" in q or "approval" in q
    asks_rejected = "rejected" in q or "reject" in q
    asks_manual = "manual" in q or "review" in q or "under review" in q
    asks_list = "show" in q or "list" in q or "give me" in q

    if is_count_question:
        if asks_manual and not asks_approved and not asks_rejected:
            return f"{len(manual_rows)} invoice(s) require manual review."
        if asks_approved and not asks_rejected and not asks_manual:
            return f"{len(approved_rows)} invoice(s) are approved."
        if asks_rejected and not asks_approved and not asks_manual:
            return f"{len(rejected_rows)} invoice(s) are rejected."
        return (
            "Invoice status counts\n\n"
            + md_table(
                ["Status", "Count"],
                [
                    ["Approved", len(approved_rows)],
                    ["Rejected", len(rejected_rows)],
                    ["Manual Review", len(manual_rows)],
                    ["Total Processed", len(rows)],
                ],
            )
        )

    if asks_list and asks_approved and asks_rejected:
        return "\n\n".join([
            status_table("Approved invoices", approved_rows),
            status_table("Rejected invoices", rejected_rows),
        ])

    if asks_list and asks_approved:
        return status_table("Approved invoices", approved_rows)

    if asks_list and asks_rejected:
        return status_table("Rejected invoices", rejected_rows)

    if asks_list and asks_manual:
        return status_table("Invoices requiring manual review", manual_rows)

    if asks_manual and not asks_rejected and not asks_approved:
        return status_table("Invoices requiring manual review", manual_rows)

    if (
        "show" in q
        or "list" in q
        or "processed" in q
        or "procees" in q
        or "process" in q
        or "summary" in q
    ) and not asks_rejected and not asks_approved and not asks_manual and "why" not in q:
        return (
            "Processed invoice summary\n\n"
            + md_table(
                ["Invoice", "Status", "Recommendation", "Vendor", "Currency"],
                [
                    [
                        r["invoice_id"],
                        r["status"],
                        r["recommendation"],
                        r["vendor_id"],
                        r["currency"],
                    ]
                    for r in rows
                ],
            )
        )

    if "recent" in q or "latest" in q or "newest" in q:
        recent = sorted(rows, key=lambda r: r["generated_at"], reverse=True)[:5]
        return "Recent invoices: " + "; ".join(
            f"{r['invoice_id']} ({r['status']}, {r['recommendation']})"
            for r in recent
        ) + "."

    if "why" in q and ("rejected" in q or "reject" in q or "this invoice" in q):
        rejected = [r for r in rows if is_rejected(r)]
        if not rejected:
            return "No rejected invoices were found, so there is no rejection reason to explain."
        return "Rejection reasons: " + "; ".join(
            f"{r['invoice_id']}: {reason_for(r)}"
            for r in rejected
        ) + "."

    if asks_rejected:
        return status_table("Rejected invoices", rejected_rows)

    if asks_approved:
        return status_table("Approved invoices", approved_rows)

    if asks_manual:
        return status_table("Invoices requiring manual review", manual_rows)

    if "vendor" in q:
        return "Vendor information: " + "; ".join(
            f"{r['invoice_id']}: {r['vendor_id']}" for r in rows
        ) + "."

    if "recommendation" in q:
        return "Recommendations: " + "; ".join(
            f"{r['invoice_id']}: {r['recommendation']}" for r in rows
        ) + "."

    if "missing" in q or "field" in q:
        findings = [
            f"{r['invoice_id']}: {', '.join(r['missing']) if r['missing'] else 'None'}"
            for r in rows
        ]
        return "Missing field findings: " + "; ".join(findings) + "."

    if "error" in q or "issue" in q or "discrep" in q or "reason" in q:
        findings = [
            f"{r['invoice_id']}: {reason_for(r)}"
            for r in rows
        ]
        return "Validation findings: " + "; ".join(findings) + "."

    return None

# ---------------------------------------------------
# Initialize / Refresh Pipeline
# ---------------------------------------------------

def initialize_rag_pipeline() -> bool:
    """
    Refresh reports and rebuild FAISS index.
    """

    try:

        # -----------------------------------------
        # Load Reports + Create Chunks
        # -----------------------------------------

        chunk_result = (

            load_and_chunk_reports()

        )

        chunk_data = json.loads(
            chunk_result
        )

        if chunk_data.get(
            "status"
        ) != "success":

            logger.error(
                "Chunk loading failed."
            )

            return False

        # -----------------------------------------
        # Build FAISS Index
        # -----------------------------------------

        index_result = (

            build_faiss_index()

        )

        index_data = json.loads(
            index_result
        )

        if index_data.get(
            "status"
        ) != "success":

            logger.error(
                "FAISS build failed."
            )

            return False

        return True

    except Exception as e:

        logger.error(
            f"Pipeline init error: {e}"
        )

        return False

# ---------------------------------------------------
# Main RAG Pipeline
# ---------------------------------------------------

def _run_rag_pipeline(
    question: str
) -> dict:

    # -----------------------------------------
    # Small Talk Handling
    # -----------------------------------------

    small_talk = handle_small_talk(
        question
    )

    if small_talk:
        workflow_event(
            "a2a.rag.router",
            input_data={"question": question},
            output_data={"route": "small_talk"},
            metadata={"agent": "router_agent"},
        )

        return {

            "question":
            question,

            "answer":
            small_talk,

            "retrieved_sources":
            []

        }

    # -----------------------------------------
    # Invoice Intent Detection
    # -----------------------------------------

    if not is_invoice_query(
        question
    ):
        workflow_event(
            "a2a.rag.router",
            input_data={"question": question},
            output_data={"route": "non_invoice"},
            metadata={"agent": "router_agent"},
            level="WARNING",
        )

        return {

            "question":
            question,

            "answer":
            (
                "I can only answer "
                "invoice-related questions."
            ),

            "retrieved_sources":
            []

        }

    # -----------------------------------------
    # Prompt Injection Protection
    # -----------------------------------------

    is_malicious = (

        ResponsibleAIGuardrails
        .detect_prompt_injection(
            question
        )

    )

    if is_malicious:
        workflow_event(
            "a2a.rag.guardrail_agent",
            input_data={"question": question},
            output_data={"blocked": True, "reason": "prompt_injection"},
            metadata={"agent": "guardrail_agent"},
            level="WARNING",
        )

        return {

            "question":
            question,

            "answer":
            (
                "Unsafe query detected. "
                "Request blocked."
            ),

            "retrieved_sources":
            []

        }

    direct_answer = answer_from_reports(question)
    if direct_answer:
        workflow_event(
            "a2a.rag.report_lookup_agent",
            input_data={"question": question},
            output_data={"answer": direct_answer},
            metadata={"agent": "report_lookup_agent", "route": "direct_answer"},
        )
        return {
            "question": question,
            "answer": direct_answer,
            "retrieved_sources": [],
        }

    # -----------------------------------------
    # Initialize / Refresh Pipeline
    # -----------------------------------------

    success = initialize_rag_pipeline()

    if not success:
        workflow_event(
            "a2a.rag.index_agent",
            input_data={"question": question},
            output_data={"ready": False},
            metadata={"agent": "index_agent"},
            level="WARNING",
        )

        return {

            "question":
            question,

            "answer":
            (
                "No invoice reports are "
                "available yet. "
                "Please process invoices first."
            ),

            "retrieved_sources":
            []

        }

    try:

        # -----------------------------------------
        # STEP 1 — Embed Question
        # -----------------------------------------

        with span(
            "a2a.rag.retrieval_agent.embed_question",
            input_data={"question": question},
            metadata={"agent": "retrieval_agent"},
        ):
            embedding_json = embed_question(
                question
            )

        workflow_event(
            "a2a.rag.retrieval_agent.embedding_created",
            input_data={"question": question},
            output_data={"embedding_status": json.loads(embedding_json).get("status")},
            metadata={"agent": "retrieval_agent"},
        )

        # -----------------------------------------
        # STEP 2 — Search FAISS
        # -----------------------------------------

        with span(
            "a2a.rag.retrieval_agent.search_faiss",
            input_data={"embedding_status": json.loads(embedding_json).get("status")},
            metadata={"agent": "retrieval_agent"},
        ):
            chunks_json = search_faiss(
                embedding_json
            )

        retrieved_data = json.loads(
            chunks_json
        )

        workflow_event(
            "a2a.rag.retrieval_agent.chunks_retrieved",
            input_data={"question": question},
            output_data={
                "status": retrieved_data.get("status"),
                "chunk_count": len(retrieved_data.get("chunks", [])),
            },
            metadata={"agent": "retrieval_agent"},
        )

        # -----------------------------------------
        # Retrieval Error Protection
        # -----------------------------------------

        if (

            retrieved_data.get("status")
            != "success"

        ):
            workflow_event(
                "a2a.rag.retrieval_agent.no_results",
                input_data={"question": question},
                output_data={"status": retrieved_data.get("status")},
                metadata={"agent": "retrieval_agent"},
                level="WARNING",
            )

            return {

                "question":
                question,

                "answer":
                (
                    "No invoice information "
                    "was found."
                ),

                "retrieved_sources":
                []

            }

        chunks = retrieved_data.get(
            "chunks",
            []
        )

        # -----------------------------------------
        # Empty Retrieval Protection
        # -----------------------------------------

        if len(chunks) == 0:
            workflow_event(
                "a2a.rag.retrieval_agent.empty_results",
                input_data={"question": question},
                output_data={"chunk_count": 0},
                metadata={"agent": "retrieval_agent"},
                level="WARNING",
            )

            return {

                "question":
                question,

                "answer":
                (
                    "No relevant invoice "
                    "information found."
                ),

                "retrieved_sources":
                []

            }

        # -----------------------------------------
        # STEP 3 — Rerank Chunks
        # -----------------------------------------

        with span(
            "a2a.rag.augmentation_agent.rerank_chunks",
            input_data={"question": question, "chunk_count": len(chunks)},
            metadata={"agent": "augmentation_agent"},
        ):
            reranked_json = rerank_chunks(

                question,

                json.dumps(
                    retrieved_data
                )

            )

        reranked_data = json.loads(
            reranked_json
        )

        workflow_event(
            "a2a.rag.augmentation_agent.chunks_reranked",
            input_data={"question": question},
            output_data={"chunk_count": len(reranked_data.get("chunks", []))},
            metadata={"agent": "augmentation_agent"},
        )

        # -----------------------------------------
        # STEP 4 — Enrich Context
        # -----------------------------------------

        with span(
            "a2a.rag.augmentation_agent.enrich_context",
            input_data={"chunk_count": len(reranked_data.get("chunks", []))},
            metadata={"agent": "augmentation_agent"},
        ):
            context_json = enrich_context(
                reranked_json
            )

        context_data = json.loads(
            context_json
        )

        enriched_context = context_data.get(
            "enriched_context",
            ""
        )

        workflow_event(
            "a2a.rag.augmentation_agent.context_enriched",
            input_data={"question": question},
            output_data={"context_chars": len(enriched_context)},
            metadata={"agent": "augmentation_agent"},
        )

        # -----------------------------------------
        # STEP 5 — Build Prompt
        # -----------------------------------------

        with span(
            "a2a.rag.generation_agent.build_prompt",
            input_data={"question": question},
            metadata={"agent": "generation_agent"},
        ):
            prompt_json = build_prompt(

                question,

                reranked_json

            )

        workflow_event(
            "a2a.rag.generation_agent.prompt_built",
            input_data={"question": question},
            output_data={"prompt_chars": len(prompt_json)},
            metadata={"agent": "generation_agent"},
        )

        # -----------------------------------------
        # STEP 6 — Generate Answer
        # -----------------------------------------

        with span(
            "a2a.rag.generation_agent.generate_answer",
            input_data={"prompt_chars": len(prompt_json)},
            metadata={"agent": "generation_agent"},
        ):
            answer_json = generate_answer(
                prompt_json
            )

        answer_data = json.loads(
            answer_json
        )

        answer = answer_data.get(

            "answer",

            "No answer generated."

        )

        workflow_event(
            "a2a.rag.generation_agent.answer_generated",
            input_data={"question": question},
            output_data={"answer": answer},
            metadata={"agent": "generation_agent"},
        )

        # -----------------------------------------
        # STEP 7 — Groundedness Check
        # -----------------------------------------

        with span(
            "a2a.rag.reflection_agent.groundedness_check",
            input_data={"answer": answer, "context_chars": len(enriched_context)},
            metadata={"agent": "reflection_agent"},
        ):
            is_grounded = (

                ResponsibleAIGuardrails
                .check_rag_groundedness(

                    answer,

                    enriched_context

                )

            )

        workflow_event(
            "a2a.rag.reflection_agent.groundedness_result",
            input_data={"question": question},
            output_data={"grounded": is_grounded},
            metadata={"agent": "reflection_agent"},
            level="DEFAULT" if is_grounded else "WARNING",
        )

        if not is_grounded:

            answer = (
                "The retrieved invoice "
                "information was insufficient "
                "to generate a grounded answer."
            )

        # -----------------------------------------
        # STEP 8 — Reflection (DEBUG ONLY)
        # -----------------------------------------

        quality_scores = {}

        if DEBUG:

            quality_json = compute_scores(

                question,

                answer,

                reranked_json

            )

            quality_scores = json.loads(
                quality_json
            )

        # -----------------------------------------
        # Sources
        # -----------------------------------------

        retrieved_sources = list(set([

            c["source"]

            for c in reranked_data.get(
                "chunks",
                []
            )

        ]))

        # -----------------------------------------
        # Final Response
        # -----------------------------------------

        response = {

            "question":
            question,

            "answer":
            answer,

            "retrieved_sources":
            retrieved_sources

        }

        if DEBUG:

            response[
                "quality_scores"
            ] = quality_scores

        return response

    except Exception as e:

        logger.error(
            f"RAG pipeline error: {e}"
        )

        return {

            "question":
            question,

            "answer":
            (
                "An internal error occurred "
                "while processing "
                "your invoice query."
            ),

            "retrieved_sources":
            []

        }

# ---------------------------------------------------
# Traced RAG Pipeline Entry Point
# ---------------------------------------------------

def run_rag_pipeline(
    question: str
) -> dict:

    with trace_context(
        "rag_invoice_question",
        input_data={"question": question},
        metadata={"workflow": "rag"},
        tags=["invoice-auditor", "rag", "guardrails"],
    ) as trace_id:
        workflow_event(
            "rag.question_received",
            input_data={"question": question},
            metadata={"workflow": "rag"},
        )

        response = _run_rag_pipeline(
            question
        )

        workflow_event(
            "rag.answer_returned",
            input_data={"question": question},
            output_data={
                "answer": response.get("answer"),
                "retrieved_sources": response.get("retrieved_sources", []),
            },
            metadata={
                "source_count": len(response.get("retrieved_sources", [])),
                "workflow": "rag",
            },
        )

        response["langfuse_trace_id"] = trace_id
        response["langfuse_status"] = tracer.last_status

        return response

# ---------------------------------------------------
# Interactive Testing
# ---------------------------------------------------

def interactive_chat():

    print("\n" + "=" * 60)
    print(" AI INVOICE AUDITOR ")
    print("=" * 60)

    while True:

        question = input(
            "\nAsk Question: "
        )

        if question.lower() in [

            "exit",

            "quit"

        ]:

            print("\nGoodbye.")

            break

        result = run_rag_pipeline(
            question
        )

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(result["answer"])

# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":

    interactive_chat()
