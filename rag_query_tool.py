from agents.Rag_agents.rag_pipeline import (
    run_rag_pipeline
)


def rag_query_tool(
    question: str
):
    """
    Run RAG query pipeline.
    """

    return run_rag_pipeline(
        question
    )