import json
import re

# ---------------------------------------------------
# Query Normalization
# ---------------------------------------------------

def normalize_question(
    question: str
) -> str:

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
        "rejected"

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
# Rerank Retrieved Chunks
# ---------------------------------------------------

def rerank_chunks(
    question: str,
    chunks_json: str
) -> str:
    """
    Rerank retrieved chunks
    using semantic heuristics.
    """

    try:

        data = json.loads(
            chunks_json
        )

        chunks = data.get(
            "chunks",
            []
        )

        normalized_question = (
            normalize_question(
                question
            )
        )

        q_words = set(

            normalized_question.split()

        )

        reranked = []

        for c in chunks:

            chunk_text = c.get(
                "chunk",
                ""
            ).lower()

            score = 0

            # ---------------------------------
            # Base Keyword Overlap
            # ---------------------------------

            overlap = len(

                q_words & set(
                    chunk_text.split()
                )

            )

            score += overlap * 2

            # ---------------------------------
            # Semantic Boosts
            # ---------------------------------

            important_patterns = {

                "processed":
                [
                    "validation status",
                    "recommendation",
                    "invoice id"
                ],

                "flagged":
                [
                    "flagged",
                    "manual review",
                    "warning",
                    "discrepancies"
                ],

                "missing":
                [
                    "missing fields",
                    "schema error"
                ],

                "recommendation":
                [
                    "recommendation"
                ],

                "recent":
                [
                    "generated at"
                ],

                "confidence":
                [
                    "translation confidence"
                ]
            }

            for keyword, patterns in (

                important_patterns.items()

            ):

                if keyword in normalized_question:

                    for p in patterns:

                        if p in chunk_text:
                            score += 5

            # ---------------------------------
            # Metadata Importance
            # ---------------------------------

            metadata_terms = [

                "invoice id",

                "vendor id",

                "validation status",

                "recommendation"

            ]

            for term in metadata_terms:

                if term in chunk_text:
                    score += 1

            reranked.append({

                **c,

                "relevance_score":
                round(score, 3)

            })

        # -------------------------------------
        # Sort by relevance
        # -------------------------------------

        reranked = sorted(

            reranked,

            key=lambda x:

            x["relevance_score"],

            reverse=True

        )

        # -------------------------------------
        # Reassign rank
        # -------------------------------------

        for i, item in enumerate(
            reranked
        ):

            item["rank"] = i + 1

        return json.dumps({

            "status": "success",

            "chunks":
            reranked

        })

    except Exception as e:

        return json.dumps({

            "status": "error",

            "message": str(e)

        })

# ---------------------------------------------------
# Enrich Final Context
# ---------------------------------------------------

def enrich_context(
    reranked_json: str
) -> str:
    """
    Build structured grounded context.
    """

    try:

        data = json.loads(
            reranked_json
        )

        chunks = data.get(
            "chunks",
            []
        )

        # -------------------------------------
        # Limit final context
        # -------------------------------------

        top_chunks = chunks[:3]

        context_parts = []

        for c in top_chunks:

            context_parts.append(

                f"""
========================================
Invoice Source: {c['source']}
Relevance Score: {c['relevance_score']}
========================================

{c['chunk']}
"""

            )

        final_context = "\n\n".join(

            context_parts

        )

        return json.dumps({

            "status": "success",

            "enriched_context":
            final_context

        })

    except Exception as e:

        return json.dumps({

            "status": "error",

            "message": str(e)

        })

augmentation_agent = None
