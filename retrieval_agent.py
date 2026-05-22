import os
import logging
import json
import faiss
import numpy as np
import re

from transformers.utils import (
    logging as hf_logging
)

hf_logging.set_verbosity_error()

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

# ---------------------------------------------------
# Integration Imports
# ---------------------------------------------------

from integration.config import (

    RAG_TOP_K,

    EMBEDDING_MODEL_NAME

)

from integration.guardrails import (

    ResponsibleAIGuardrails

)

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
# Import FAISS Store
# ---------------------------------------------------

from agents.Rag_agents.indexing_agent import (
    get_faiss_store
)

# ---------------------------------------------------
# Logging
# ---------------------------------------------------

logging.basicConfig(
    level=logging.ERROR
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Retrieval Config
# ---------------------------------------------------

DISTANCE_THRESHOLD = 3.0

# ---------------------------------------------------
# Local Embedding Model
# ---------------------------------------------------

embedding_model = None


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        if SentenceTransformer is None:
            return None
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return embedding_model


def encode_text(text: str) -> list[float]:
    model = get_embedding_model()
    if model is not None:
        return model.encode(text, show_progress_bar=False).tolist()

    vector = np.zeros(384, dtype="float32")
    for token in text.lower().split():
        bucket = abs(hash(token)) % 384
        vector[bucket] += 1.0
    norm = np.linalg.norm(vector)
    if norm:
        vector = vector / norm
    return vector.tolist()

# ---------------------------------------------------
# Query Normalization
# ---------------------------------------------------

def normalize_question(
    question: str
) -> str:
    """
    Normalize semantic variations.
    """

    question = question.lower()

    replacements = {

        "processes":
        "processed",

        "processing":
        "processed",

        "done":
        "processed",

        "latest":
        "recent",

        "newest":
        "recent",

        "flag":
        "flagged",

        "reject":
        "rejected",

        "errors":
        "validation issues",

        "problem":
        "issue"

    }

    for old, new in replacements.items():

        question = question.replace(
            old,
            new
        )

    question = re.sub(

        r"\s+",

        " ",

        question

    )

    return question.strip()

# ---------------------------------------------------
# Embed User Question
# ---------------------------------------------------

def embed_question(
    question: str
) -> str:

    try:

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

            return json.dumps({

                "status": "error",

                "message":
                "Unsafe query detected."

            })

        # -----------------------------------------
        # Normalize Question
        # -----------------------------------------

        normalized_question = (
            normalize_question(
                question
            )
        )

        # -----------------------------------------
        # Generate Embedding
        # -----------------------------------------

        embedding = encode_text(normalized_question)

        return json.dumps({

            "status": "success",

            "question":
            normalized_question,

            "embedding":
            embedding

        })

    except Exception as e:

        logger.error(
            f"Embedding error: {e}"
        )

        return json.dumps({

            "status": "error",

            "message": str(e)

        })

# ---------------------------------------------------
# Search FAISS
# ---------------------------------------------------

def search_faiss(
    question_embedding_json: str
) -> str:

    # -----------------------------------------
    # Load FAISS Store
    # -----------------------------------------

    index, chunks, sources = (
        get_faiss_store()
    )

    # -----------------------------------------
    # Check Index
    # -----------------------------------------

    if index is None:

        return json.dumps({

            "status": "error",

            "message":
            "FAISS index not initialized"

        })

    try:

        # -----------------------------------------
        # Parse Embedding
        # -----------------------------------------

        data = json.loads(
            question_embedding_json
        )

        if data.get(
            "status"
        ) == "error":

            return json.dumps(data)

        question = data.get(
            "question",
            ""
        )

        embedding = np.array(

            [data["embedding"]],

            dtype="float32"

        )

        # -----------------------------------------
        # Search FAISS
        # -----------------------------------------

        distances, indices = index.search(

            embedding,

            RAG_TOP_K

        )

        results = []

        seen_chunks = set()

        seen_sources = set()

        # -----------------------------------------
        # Build Results
        # -----------------------------------------

        for rank, idx in enumerate(

            indices[0]

        ):

            # ---------------------------------
            # Invalid Index Protection
            # ---------------------------------

            if idx >= len(chunks):
                continue

            chunk = chunks[idx]

            source = sources[idx]

            distance = float(

                distances[0][rank]

            )

            # ---------------------------------
            # Similarity Filtering
            # ---------------------------------

            if distance > DISTANCE_THRESHOLD:
                continue

            # ---------------------------------
            # Duplicate Chunk Protection
            # ---------------------------------

            if chunk in seen_chunks:
                continue

            # ---------------------------------
            # Duplicate Source Protection
            # ---------------------------------

            if source in seen_sources:
                continue

            seen_chunks.add(
                chunk
            )

            seen_sources.add(
                source
            )

            # ---------------------------------
            # Invoice Semantic Boost
            # ---------------------------------

            semantic_score = 0

            lowered_chunk = chunk.lower()

            important_terms = [

                "invoice id",

                "recommendation",

                "validation status",

                "vendor id",

                "generated at"

            ]

            for term in important_terms:

                if term in lowered_chunk:
                    semantic_score += 1

            results.append({

                "rank":
                rank + 1,

                "chunk":
                chunk,

                "source":
                source,

                "distance":
                round(
                    distance,
                    4
                ),

                "semantic_score":
                semantic_score

            })

        # -----------------------------------------
        # Recent Invoice Prioritization
        # -----------------------------------------

        if "recent" in question:

            results = sorted(

                results,

                key=lambda x:

                x["chunk"].count(
                    "Generated At"
                ),

                reverse=True

            )

        # -----------------------------------------
        # Semantic Score Sorting
        # -----------------------------------------

        results = sorted(

            results,

            key=lambda x: (

                -x["semantic_score"],

                x["distance"]

            )

        )

        # -----------------------------------------
        # Empty Result Protection
        # -----------------------------------------

        if len(results) == 0:

            return json.dumps({

                "status": "success",

                "chunks": [],

                "message":
                "No highly relevant chunks found."

            })

        # -----------------------------------------
        # Return Results
        # -----------------------------------------

        return json.dumps({

            "status": "success",

            "chunks":
            results

        })

    except Exception as e:

        logger.error(
            f"Retrieval error: {e}"
        )

        return json.dumps({

            "status": "error",

            "message": str(e)

        })

retrieval_agent = None
