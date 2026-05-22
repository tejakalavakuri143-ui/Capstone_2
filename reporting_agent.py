# reporting_agent.py

import json
import yaml
import os

from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional

from database.invoice_db import save_report

from langgraph.graph import (
    StateGraph,
    END
)

from langchain_core.tools import tool


# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TEMPLATE_PATH = os.path.join(
    BASE_DIR,
    "config",
    "invoice_report_template.html"
)

RULES_DEFAULT = os.path.join(
    BASE_DIR,
    "config",
    "rules.yaml"
)


# =====================================================
# STATE
# =====================================================

class ReportingState(TypedDict):

    raw_validation_result: dict

    validation_result: dict

    rules: dict

    report: dict

    json_report_path: str

    html_report_path: str

    recommendation: str

    status: str

    rules_path: str


# =====================================================
# NORMALIZATION TOOL
# =====================================================

@tool
def normalize_validation_tool(
    raw: dict
) -> dict:
    """
    Normalize validation payload.
    """

    status_map = {

        "PASSED":
        "APPROVED",

        "APPROVED_WITH_WARNING":
        "WARNING",

        "MANUAL_REVIEW":
        "REVIEW_REQUIRED",

        "AUTO_APPROVED":
        "APPROVED",

        "FAILED":
        "REJECTED",

        "REJECTED":
        "REJECTED"
    }

    raw_status = raw.get(
        "status",
        "UNKNOWN"
    )

    validated_invoice = raw.get(
        "validated_invoice"
    )

    if not isinstance(
        validated_invoice,
        dict
    ):

        validated_invoice = {}

    normalized_status = status_map.get(

        raw_status,

        "REVIEW_REQUIRED"

    )

    discrepancies = []
    missing_fields = []

    # -------------------------------------------------
    # Validation Errors
    # -------------------------------------------------

    for key in [

        "data_validation_errors",

        "business_validation_errors"

    ]:

        val = raw.get(key, [])

        if isinstance(val, list):

            for item in val:

                text = str(item)

                if text.startswith("Missing required"):

                    missing_fields.append(text)

                else:

                    discrepancies.append(text)

        elif isinstance(val, str) and val:

            if val.startswith("Missing required"):

                missing_fields.append(val)

            else:

                discrepancies.append(val)

    # -------------------------------------------------
    # Structured discrepancies
    # -------------------------------------------------

    structured_discrepancies = raw.get(
        "discrepancies",
        []
    )

    for d in structured_discrepancies:

        if isinstance(d, dict):

            discrepancies.append(

                d.get(
                    "message",
                    str(d)
                )

            )

        else:

            discrepancies.append(
                str(d)
            )

    # -------------------------------------------------
    # Missing fields
    # -------------------------------------------------

    schema_err = raw.get(
        "schema_errors",
        ""
    )

    if schema_err:

        missing_fields.append(

            f"Schema error: {schema_err}"

        )

    # -------------------------------------------------
    # Invoice ID
    # -------------------------------------------------

    invoice_id = (

        raw.get("invoice_id")

        or validated_invoice.get(
            "header",
            {}
        ).get(
            "invoice_no"
        )

        or "UNKNOWN"

    )

    def safe_str(
        val,
        default="UNKNOWN"
    ):

        return (

            str(val)

            if val not in (
                None,
                "",
                "None"
            )

            else default
        )

    return {

        "invoice_id":
        safe_str(invoice_id),

        "vendor_id":
        safe_str(
            raw.get("vendor_id")
        ),

        "currency":
        safe_str(
            raw.get("currency")
        ),

        "translation_confidence":
        raw.get(
            "translation_confidence"
        ),

        "validation_status":
        normalized_status,

        "missing_fields":
        missing_fields,

        "discrepancies":
        discrepancies,

        "warnings":
        raw.get(
            "warnings",
            []
        ),

        "line_items":
        raw.get("line_items")
        or validated_invoice.get(
            "line_items",
            []
        ),

        "original_status":
        raw_status
    }


# =====================================================
# LOAD RULES TOOL
# =====================================================

@tool
def load_rules_tool(
    rules_path: str
) -> dict:
    """
    Load YAML rules.
    """

    with open(
        rules_path,
        "r",
        encoding="utf-8"
    ) as f:

        return yaml.safe_load(f)


# =====================================================
# RECOMMENDATION TOOL
# =====================================================

@tool
def determine_recommendation_tool(

    status: str,

    discrepancies: list,

    missing_fields: list,

    translation_confidence: Optional[float],

    currency: str,

    original_status: str,

    rules: dict

) -> str:
    """
    Determine recommendation.
    """

    if original_status == "PASSED":

        return "Approve"

    if original_status == "APPROVED_WITH_WARNING":

        return "Approve With Warning"

    if original_status == "MANUAL_REVIEW":

        return "Manual Review"

    if original_status == "REJECTED":

        return "Reject"

    if original_status == "FAILED":

        return "Reject"

    return "Manual Review"


# =====================================================
# SUMMARY TOOL
# =====================================================

@tool
def build_summary_tool(

    status: str,

    discrepancies: list,

    missing_fields: list,

    recommendation: str

) -> str:
    """
    Build semantic summary.
    """

    parts = []

    # -------------------------------------------------
    # STATUS SUMMARY
    # -------------------------------------------------

    if status == "APPROVED":

        parts.append(
            "Invoice passed validation."
        )

    elif status == "WARNING":

        parts.append(

            "Invoice approved with warnings."

        )

    elif status == "REVIEW_REQUIRED":

        parts.append(

            "Invoice requires manual review."

        )

    elif status == "REJECTED":

        parts.append(
            "Invoice was rejected."
        )

    # -------------------------------------------------
    # MISSING FIELDS
    # -------------------------------------------------

    if missing_fields:

        parts.append(

            "Missing fields: "

            + ", ".join(missing_fields)

        )

    # -------------------------------------------------
    # DISCREPANCIES
    # -------------------------------------------------

    if discrepancies:

        parts.append(

            "Discrepancies found: "

            + ", ".join(discrepancies)

        )

    # -------------------------------------------------
    # FINAL RECOMMENDATION
    # -------------------------------------------------

    parts.append(

        f"Recommendation: "
        f"{recommendation}"

    )

    return " ".join(parts)


# =====================================================
# SAVE JSON TOOL
# =====================================================

@tool
def save_json_tool(
    report: dict,
    output_dir: str
) -> str:
    """
    Save JSON report.
    """

    out = Path(output_dir)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    path = out / (

        f"{report['report_metadata']['invoice_id']}_report.json"

    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    return str(path)


# =====================================================
# SAVE HTML TOOL
# =====================================================

@tool
def save_html_tool(

    report: dict,

    rules: dict,

    output_dir: str,

    template_path: str = TEMPLATE_PATH

) -> str:
    """
    Save HTML report.
    """

    out = Path(output_dir)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    path = out / (

        f"{report['report_metadata']['invoice_id']}_report.html"

    )

    html = f"""
    <html>
    <body>

    <h1>Invoice Audit Report</h1>

    <p>
    Invoice ID:
    {report['report_metadata']['invoice_id']}
    </p>

    <p>
    Status:
    {report['validation_result']['status']}
    </p>

    <p>
    Summary:
    {report['validation_result']['summary']}
    </p>

    <p>
    Recommendation:
    {report['recommendation']}
    </p>

    </body>
    </html>
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)

    return str(path)


# =====================================================
# NORMALIZE NODE
# =====================================================

def normalize_input_node(
    state: ReportingState
) -> dict:

    normalized = normalize_validation_tool.invoke({

        "raw":
        state["raw_validation_result"]

    })

    return {

        "validation_result":
        normalized

    }


# =====================================================
# RULES NODE
# =====================================================

def load_rules_node(
    state: ReportingState
) -> dict:

    rules = load_rules_tool.invoke({

        "rules_path":
        state["rules_path"]

    })

    return {

        "rules":
        rules

    }


# =====================================================
# BUILD REPORT NODE
# =====================================================

def build_report_node(
    state: ReportingState
) -> dict:

    v = state["validation_result"]

    rules = state["rules"]

    recommendation = determine_recommendation_tool.invoke({

        "status":
        v["validation_status"],

        "discrepancies":
        v["discrepancies"],

        "missing_fields":
        v["missing_fields"],

        "translation_confidence":
        v["translation_confidence"],

        "currency":
        v["currency"],

        "original_status":
        v["original_status"],

        "rules":
        rules

    })

    summary = build_summary_tool.invoke({

        "status":
        v["validation_status"],

        "discrepancies":
        v["discrepancies"],

        "missing_fields":
        v["missing_fields"],

        "recommendation":
        recommendation

    })

    report = {

        "report_metadata": {

            "invoice_id":
            v["invoice_id"],

            "generated_at":
            datetime.utcnow().isoformat() + "Z"
        },

        "invoice_summary": {

            "vendor_id":
            v["vendor_id"],

            "currency":
            v["currency"],

            "translation_confidence":
            v["translation_confidence"]
        },

        "validation_result": {

            "status":
            v["validation_status"],

            "summary":
            summary,

            "missing_fields":
            v["missing_fields"],

            "warnings":
            v["warnings"],

            "discrepancy_summary":
            v["discrepancies"]
        },

        "line_items":
        v["line_items"],

        "recommendation":
        recommendation
    }

    return {

        "report":
        report,

        "recommendation":
        recommendation,

        "status":
        v["validation_status"]

    }


# =====================================================
# SAVE JSON NODE
# =====================================================

def save_json_node(
    state: ReportingState
) -> dict:

    path = save_json_tool.invoke({

        "report":
        state["report"],

        "output_dir":
        "outputs/reports"

    })

    save_report(
        state["report"]
    )

    return {

        "json_report_path":
        path

    }


# =====================================================
# SAVE HTML NODE
# =====================================================

def save_html_node(
    state: ReportingState
) -> dict:

    path = save_html_tool.invoke({

        "report":
        state["report"],

        "rules":
        state["rules"],

        "output_dir":
        "outputs/reports"

    })

    return {

        "html_report_path":
        path

    }


# =====================================================
# GRAPH
# =====================================================

workflow = StateGraph(
    ReportingState
)

workflow.add_node(
    "normalize_input",
    normalize_input_node
)

workflow.add_node(
    "load_rules",
    load_rules_node
)

workflow.add_node(
    "build_report",
    build_report_node
)

workflow.add_node(
    "save_json",
    save_json_node
)

workflow.add_node(
    "save_html",
    save_html_node
)

workflow.set_entry_point(
    "normalize_input"
)

workflow.add_edge(
    "normalize_input",
    "load_rules"
)

workflow.add_edge(
    "load_rules",
    "build_report"
)

workflow.add_edge(
    "build_report",
    "save_json"
)

workflow.add_edge(
    "save_json",
    "save_html"
)

workflow.add_edge(
    "save_html",
    END
)

reporting_graph = workflow.compile()


# =====================================================
# MAIN FUNCTION
# =====================================================

def run_reporting_agent(

    validation_result: dict,

    rules_path: str = RULES_DEFAULT

):

    result = reporting_graph.invoke({

        "raw_validation_result":
        validation_result,

        "rules_path":
        rules_path

    })

    return {

        "invoice_id":
        result["report"]["report_metadata"]["invoice_id"],

        "status":
        result["status"],

        "recommendation":
        result["recommendation"],

        "json_report_path":
        result["json_report_path"],

        "html_report_path":
        result["html_report_path"]

    }
