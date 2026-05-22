import os
import logging
import json
import faiss
import numpy as np

from transformers.utils import (
    logging as hf_logging
)

hf_logging.set_verbosity_error()

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

# ---------------------------------------------------
# Disable noisy logs
# ---------------------------------------------------

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"

os.environ[
    "HF_HUB_DISABLE_PROGRESS_BARS"
] = "1"

logging.getLogger(
    "httpx"
).setLevel(logging.ERROR)

logging.getLogger(
    "sentence_transformers"
).setLevel(logging.ERROR)

logging.getLogger(
    "huggingface_hub"
).setLevel(logging.ERROR)

# ---------------------------------------------------
# DB
# ---------------------------------------------------

from database.invoice_db import (
    load_reports
)

# ---------------------------------------------------
# Logging
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Embedding Model
# ---------------------------------------------------

embedding_model = None


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        if SentenceTransformer is None:
            return None
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model


def encode_text(text: str) -> np.ndarray:
    model = get_embedding_model()
    if model is not None:
        return np.array(model.encode(text, show_progress_bar=False), dtype="float32")

    vector = np.zeros(384, dtype="float32")
    for token in text.lower().split():
        bucket = abs(hash(token)) % 384
        vector[bucket] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector

# ---------------------------------------------------
# Global FAISS Store
# ---------------------------------------------------

_faiss_index = None

_chunks = []

_sources = []

# ---------------------------------------------------
# Build Rich Semantic Report Text
# ---------------------------------------------------

def build_report_text(report):

    metadata = report.get(
        "report_metadata",
        {}
    )

    validation = report.get(
        "validation_result",
        {}
    )

    invoice_summary = report.get(
        "invoice_summary",
        {}
    )

    # -----------------------------------------
    # Metadata
    # -----------------------------------------

    invoice_id = metadata.get(
        "invoice_id",
        "UNKNOWN"
    )

    generated_at = metadata.get(
        "generated_at",
        "UNKNOWN"
    )

    # -----------------------------------------
    # Validation
    # -----------------------------------------

    status = validation.get(
        "status",
        "UNKNOWN"
    )

    recommendation = validation.get(
        "recommendation",
        report.get(
            "recommendation",
            "UNKNOWN"
        )
    )

    summary = validation.get(
        "summary",
        "No summary available"
    )

    discrepancies = validation.get(
        "discrepancy_summary",
        []
    )

    missing_fields = validation.get(
        "missing_fields",
        []
    )

    # -----------------------------------------
    # Invoice Summary
    # -----------------------------------------

    vendor_id = invoice_summary.get(
        "vendor_id",
        "UNKNOWN"
    )

    currency = invoice_summary.get(
        "currency",
        "UNKNOWN"
    )

    translation_confidence = invoice_summary.get(
        "translation_confidence",
        "UNKNOWN"
    )

    # -----------------------------------------
    # Convert Lists
    # -----------------------------------------

    discrepancy_text = ", ".join(

        discrepancies

    ) if discrepancies else "None"

    missing_text = ", ".join(

        missing_fields

    ) if missing_fields else "None"

    # -----------------------------------------
    # Rich Semantic Context
    # -----------------------------------------

    text = f"""
Invoice ID: {invoice_id}

Vendor ID: {vendor_id}

Currency: {currency}

Generated At: {generated_at}

Translation Confidence: {translation_confidence}

Validation Status: {status}

Recommendation: {recommendation}

Summary:
{summary}

Missing Fields:
{missing_text}

Discrepancies:
{discrepancy_text}

Invoice Keywords:
invoice audit validation discrepancy approval rejection warning review missing fields invoice processing
"""

    return text.strip()

# ---------------------------------------------------
# Better Semantic Chunking
# ---------------------------------------------------

def chunk_text(
    text: str,
    invoice_id: str
):

    sections = [

        s.strip()

        for s in text.split("\n\n")

        if s.strip()

    ]

    chunks = []

    current_chunk = ""

    for section in sections:

        if (

            len(current_chunk)

            + len(section)

            < 500

        ):

            current_chunk += (
                "\n\n" + section
            )

        else:

            if current_chunk.strip():

                chunks.append({

                    "chunk":
                    current_chunk.strip(),

                    "source":
                    invoice_id

                })

            current_chunk = section

    if current_chunk.strip():

        chunks.append({

            "chunk":
            current_chunk.strip(),

            "source":
            invoice_id

        })

    return chunks

# ---------------------------------------------------
# Load Reports + Chunk
# ---------------------------------------------------

def load_and_chunk_reports() -> str:

    reports = load_reports()

    if not reports:

        return json.dumps({

            "status": "error",

            "message":
            "No reports found in database"

        })

    all_chunks = []

    all_sources = []

    processed_ids = set()

    for report in reports:

        try:

            invoice_id = report.get(

                "report_metadata",
                {}

            ).get(

                "invoice_id",

                "UNKNOWN"

            )

            # ---------------------------------
            # Skip duplicates
            # ---------------------------------

            if invoice_id in processed_ids:
                continue

            processed_ids.add(
                invoice_id
            )

            # ---------------------------------
            # Build semantic text
            # ---------------------------------

            text = build_report_text(
                report
            )

            # ---------------------------------
            # Semantic chunking
            # ---------------------------------

            chunks = chunk_text(
                text,
                invoice_id
            )

            for item in chunks:

                all_chunks.append(
                    item["chunk"]
                )

                all_sources.append(
                    item["source"]
                )

        except Exception as e:

            logger.error(

                f"Chunking error "
                f"for {invoice_id}: {e}"

            )

    global _chunks, _sources

    _chunks = all_chunks

    _sources = all_sources

    logger.info(

        f"Created "
        f"{len(_chunks)} chunks"

    )

    return json.dumps({

        "status": "success",

        "total_chunks":
        len(_chunks)

    })

# ---------------------------------------------------
# Build FAISS Index
# ---------------------------------------------------

def build_faiss_index(force_reload: bool = False) -> str:

    global _faiss_index

    if force_reload or len(_chunks) == 0:

        load_and_chunk_reports()

    if len(_chunks) == 0:

        return json.dumps({

            "status": "error",

            "message":
            "No chunks available"

        })

    embeddings = []

    for chunk in _chunks:

        try:

            emb = encode_text(chunk)

            embeddings.append(

                emb

            )

        except Exception as e:

            logger.error(
                f"Embedding error: {e}"
            )

    if len(embeddings) == 0:

        return json.dumps({

            "status": "error",

            "message":
            "No embeddings created"

        })

    matrix = np.vstack(
        embeddings
    ).astype("float32")

    # -----------------------------------------
    # Fresh FAISS Index
    # -----------------------------------------

    _faiss_index = faiss.IndexFlatL2(

        matrix.shape[1]

    )

    _faiss_index.add(matrix)

    logger.info(

        f"Indexed "
        f"{_faiss_index.ntotal} vectors"

    )

    return json.dumps({

        "status": "success",

        "vectors":
        _faiss_index.ntotal

    })

indexing_agent = None

# ---------------------------------------------------
# Shared FAISS Store
# ---------------------------------------------------

def get_faiss_store():

    return (

        _faiss_index,

        _chunks,

        _sources

    )
