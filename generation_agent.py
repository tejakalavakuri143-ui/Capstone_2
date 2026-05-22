import json
import logging
import re

from integration.config import BEDROCK_MODEL, BEDROCK_REGION, USE_BEDROCK

logger = logging.getLogger(__name__)
_bedrock_llm = None


def build_prompt(question: str, chunks_json: str) -> str:
    data = json.loads(chunks_json)
    chunks = data.get("chunks", [])
    context = "\n\n".join(c.get("chunk", "") for c in chunks[:3])
    sources = [c.get("source") for c in chunks[:3] if c.get("source")]
    return json.dumps({
        "status": "success",
        "question": question,
        "context": context,
        "sources": sources,
    })


def _get_bedrock_llm():
    if not USE_BEDROCK:
        raise RuntimeError("Bedrock RAG generation disabled. Set INVOICE_USE_BEDROCK=1 to enable it.")
    global _bedrock_llm
    if _bedrock_llm is None:
        from langchain_aws import ChatBedrockConverse

        _bedrock_llm = ChatBedrockConverse(
            model=BEDROCK_MODEL,
            region_name=BEDROCK_REGION,
            temperature=0,
            max_tokens=400,
        )
    return _bedrock_llm


def _extractive_answer(question: str, context: str) -> str:
    if not context.strip():
        return "No relevant invoice information found."

    q = question.lower()
    invoice_ids = re.findall(r"Invoice ID:\s*([^\n]+)", context)
    statuses = re.findall(r"Validation Status:\s*([^\n]+)", context)
    recommendations = re.findall(r"Recommendation:\s*([^\n]+)", context)
    vendors = re.findall(r"Vendor ID:\s*([^\n]+)", context)
    discrepancies = re.findall(r"Discrepancies:\s*([^\n]+)", context)
    missing = re.findall(r"Missing Fields:\s*([^\n]+)", context)

    if any(term in q for term in ["count", "how many", "number of"]):
        unique_ids = sorted(set(invoice_ids))
        return f"{len(unique_ids)} processed invoice(s) are available: {', '.join(unique_ids)}."

    if "recent" in q or "latest" in q or "newest" in q:
        return f"Recent invoice sources found: {', '.join(invoice_ids[:3])}."

    if "vendor" in q:
        pairs = [f"{inv}: {vendor}" for inv, vendor in zip(invoice_ids, vendors)]
        return "Vendor information: " + "; ".join(pairs) + "."

    if "recommendation" in q or "approve" in q or "reject" in q or "review" in q:
        pairs = [f"{inv}: {rec}" for inv, rec in zip(invoice_ids, recommendations)]
        return "Recommendations: " + "; ".join(pairs) + "."

    if "missing" in q or "field" in q:
        pairs = [f"{inv}: {value}" for inv, value in zip(invoice_ids, missing)]
        return "Missing field findings: " + "; ".join(pairs) + "."

    if "discrep" in q or "error" in q or "issue" in q:
        pairs = [f"{inv}: {value}" for inv, value in zip(invoice_ids, discrepancies)]
        return "Validation findings: " + "; ".join(pairs) + "."

    pairs = [
        f"{inv}: status {status}, recommendation {rec}"
        for inv, status, rec in zip(invoice_ids, statuses, recommendations)
    ]
    if pairs:
        return "Invoice summary: " + "; ".join(pairs) + "."

    return "I could not find that information in the invoice reports."


def generate_answer(prompt_json: str) -> str:
    data = json.loads(prompt_json)
    question = data.get("question", "")
    context = data.get("context", "")

    if not context.strip():
        return json.dumps({
            "status": "success",
            "answer": "No relevant invoice information found.",
        })

    prompt = f"""
You are an AI Invoice Audit Assistant.

Answer only with facts found in the invoice context. If the answer is not in
the context, say "I could not find that information in the invoice reports."

QUESTION:
{question}

INVOICE CONTEXT:
{context}
"""

    try:
        response = _get_bedrock_llm().invoke(prompt)
        answer = response.content.strip()
    except Exception as exc:
        logger.warning("Bedrock RAG generation failed; using extractive answer: %s", exc)
        answer = _extractive_answer(question, context)

    return json.dumps({
        "status": "success",
        "answer": answer or _extractive_answer(question, context),
    })


generation_agent = None
