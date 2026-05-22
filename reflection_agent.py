import json



def compute_scores(question: str, answer: str, chunks_json: str) -> str:

    chunks = json.loads(chunks_json)["chunks"]

    context = " ".join(
        c["chunk"] for c in chunks
    ).lower()

    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())

    relevance = (
        len(q_words & a_words) / len(q_words)
        if len(q_words) > 0 else 0
    )

    groundedness = (
        sum(1 for w in a_words if w in context) / len(a_words)
        if len(a_words) > 0 else 0
    )

    context_relevance = (
        sum(1 for w in q_words if w in context) / len(q_words)
        if len(q_words) > 0 else 0
    )

    overall_score = round(
        (relevance + groundedness + context_relevance) / 3,
        2
    )

    quality = (
        "Good"
        if overall_score >= 0.7
        else "Acceptable"
        if overall_score >= 0.4
        else "Poor"
    )

    return json.dumps({
        "relevance": relevance,
        "groundedness": groundedness,
        "context_relevance": context_relevance,
        "overall_score": overall_score,
        "quality": quality
    })


reflection_agent = None
