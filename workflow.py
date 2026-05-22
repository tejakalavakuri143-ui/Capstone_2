from langgraph.graph import StateGraph, END

from state.invoice_state import InvoiceState

from agents.monitor_agent import wait_for_batch
from agents.extractor_agent import extractor_agent
from agents.translation_agent import translate_invoice


# ---------------------------------------------------
# Monitor Node
# ---------------------------------------------------

def monitor_node(state):

    files = wait_for_batch("data/incoming")

    return {
        "files": files
    }


# ---------------------------------------------------
# Extractor Node
# ---------------------------------------------------

def extractor_node(state):

    files = state["files"]

    extracted = extractor_agent(files)

    return {
        "extracted_data": extracted["extracted_data"]
    }


# ---------------------------------------------------
# Translation Node
# ---------------------------------------------------

def translation_node(state):

    extracted_docs = state["extracted_data"]

    translated_results = []

    for doc in extracted_docs:

        print(f"\n[Translation] Processing {doc['file_name']}")

        structured = translate_invoice(doc["file_text"])

        translated_results.append({
            "file_name": doc["file_name"],
            "structured_data": structured
        })

    return {
        "translated_data": translated_results
    }


# ---------------------------------------------------
# Build Graph
# ---------------------------------------------------

builder = StateGraph(InvoiceState)

builder.add_node("monitor", monitor_node)

builder.add_node("extractor", extractor_node)

builder.add_node("translator", translation_node)


builder.set_entry_point("monitor")

builder.add_edge("monitor", "extractor")

builder.add_edge("extractor", "translator")

builder.add_edge("translator", END)


invoice_graph = builder.compile()