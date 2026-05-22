import os
import json
import yaml
import logging
import difflib


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RULES_PATH = os.path.join(
    BASE_DIR,
    "config",
    "rules.yaml"
)

PO_MASTER_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ERP_mockdata",
    "PO Records.json"
)

SKU_MASTER_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ERP_mockdata",
    "sku_master.json"
)


# ---------------------------------------------------
# BUSINESS VALIDATION AGENT
# ---------------------------------------------------

class BusinessValidationAgent:

    def __init__(self):

        # -----------------------------------------
        # Load Rules
        # -----------------------------------------

        with open(RULES_PATH, "r") as f:

            self.rules = yaml.safe_load(f)

        # -----------------------------------------
        # Load PO Master
        # -----------------------------------------

        with open(PO_MASTER_PATH, "r") as f:

            self.po_master = json.load(f)

        with open(SKU_MASTER_PATH, "r") as f:

            self.sku_master = json.load(f)

        self.sku_codes = {

            item.get("item_code")

            for item in self.sku_master

        }

    # -------------------------------------------------
    # NORMALIZE TEXT
    # -------------------------------------------------

    def normalize_text(
        self,
        text
    ):

        if not text:

            return ""

        text = text.lower().strip()

        replacements = {

            "boxes": "box",
            "gloves": "glove",
            "helmets": "helmet",
            "labels": "label"

        }

        words = text.split()

        normalized_words = [

            replacements.get(
                word,
                word
            )

            for word in words
        ]

        return " ".join(
            normalized_words
        )

    # -------------------------------------------------
    # FIND PO RECORD
    # -------------------------------------------------

    def find_po_record(
        self,
        po_number
    ):

        for po in self.po_master:

            if po.get(
                "po_number"
            ) == po_number:

                return po

        return None

    # -------------------------------------------------
    # FIND ERP RECORD BY VENDOR
    # -------------------------------------------------

    def find_vendor_record(
        self,
        vendor_id
    ):

        for po in self.po_master:

            if po.get("vendor_id") == vendor_id:

                return po

        return None

    # -------------------------------------------------
    # FIND MATCHING PO ITEM
    # -------------------------------------------------

    def find_matching_po_item(

        self,

        invoice_item,

        po_items

    ):

        # -----------------------------------------
        # STEP 1 — Match using item_code
        # -----------------------------------------

        invoice_code = invoice_item.get(
            "item_code"
        )

        if invoice_code:

            for po_item in po_items:

                if (

                    po_item.get(
                        "item_code"
                    )

                    == invoice_code

                ):

                    return po_item

        # -----------------------------------------
        # STEP 2 — Exact Description Match
        # -----------------------------------------

        invoice_desc = self.normalize_text(

            invoice_item.get(
                "description",
                ""
            )

        )

        for po_item in po_items:

            po_desc = self.normalize_text(

                po_item.get(
                    "description",
                    ""
                )

            )

            if invoice_desc == po_desc:

                return po_item

        # -----------------------------------------
        # STEP 3 — Fuzzy Match
        # -----------------------------------------

        best_match = None
        best_score = 0

        for po_item in po_items:

            po_desc = self.normalize_text(

                po_item.get(
                    "description",
                    ""
                )

            )

            similarity = difflib.SequenceMatcher(

                None,

                invoice_desc,

                po_desc

            ).ratio()

            if similarity > best_score:

                best_score = similarity
                best_match = po_item

        # -----------------------------------------
        # Similarity Threshold
        # -----------------------------------------

        if best_score >= 0.75:

            logging.info(

                f"Fuzzy Match Found | "

                f"Invoice='{invoice_desc}' | "

                f"PO='{best_match.get('description')}' | "

                f"Score={best_score:.2f}"

            )

            return best_match

        return None

    # -------------------------------------------------
    # MAIN VALIDATION
    # -------------------------------------------------

    def validate(
        self,
        validation_result
    ):

        # -----------------------------------------
        # Extract Invoice Data
        # -----------------------------------------

        invoice_json = validation_result.get(

            "validated_invoice",

            validation_result

        )

        header = invoice_json.get(
            "header",
            {}
        )

        invoice_id = header.get(
            "invoice_no",
            "UNKNOWN"
        )

        vendor_id = header.get(
            "vendor_id",
            "UNKNOWN"
        )

        # -----------------------------------------
        # FIXED
        # No fake UNKNOWN PO number
        # -----------------------------------------

        po_number = header.get(
            "po_number"
        )

        currency = header.get(
            "currency",
            "UNKNOWN"
        )

        translation_confidence = invoice_json.get(
            "translation_confidence",
            validation_result.get("translation_confidence"),
        )

        items = invoice_json.get(
            "line_items",
            []
        )

        business_errors = []
        warnings = []
        discrepancies = []

        # -----------------------------------------
        # Find ERP Record
        # -----------------------------------------

        po_record = None

        if po_number:

            po_record = self.find_po_record(
                po_number
            )

        if not po_record:

            po_record = self.find_vendor_record(
                vendor_id
            )

        # =================================================
        # FALLBACK LOGIC IF PO MISSING
        # =================================================

        # -----------------------------------------
        # FIXED
        # Handles both:
        # missing PO
        # invalid PO
        # -----------------------------------------

        if not po_record:

            logging.warning(

                f"PO not found for "
                f"{invoice_id}"

            )

            # -----------------------------------------
            # Check SKU availability
            # -----------------------------------------

            sku_exists = any([

                item.get("item_code")

                for item in items

            ])

            # -----------------------------------------
            # CASE 1
            # Vendor + SKU exists
            # -----------------------------------------

            if (

                vendor_id != "UNKNOWN"

                and

                sku_exists

            ):

                return {

                    "status":
                    "APPROVED_WITH_WARNING",

                    "stage":
                    "BUSINESS_VALIDATION",

                    "recommendation":
                    "Approve With Warning",

                    "invoice_id":
                    invoice_id,

                    "vendor_id":
                    vendor_id,

                    "po_number":
                    po_number,

                    "currency":
                    currency,

                    "translation_confidence":
                    translation_confidence,

                    "warnings": [

                        "PO number missing.",

                        "Fallback validation used "
                        "vendor_id + item_code."

                    ],

                    "business_validation_errors":
                    [],

                    "discrepancies":
                    [],

                    "validated_invoice":
                    invoice_json
                }

            # -----------------------------------------
            # CASE 2
            # Vendor exists but no SKU
            # -----------------------------------------

            elif vendor_id != "UNKNOWN":

                return {

                    "status":
                    "MANUAL_REVIEW",

                    "stage":
                    "BUSINESS_VALIDATION",

                    "recommendation":
                    "Manual Review",

                    "invoice_id":
                    invoice_id,

                    "vendor_id":
                    vendor_id,

                    "po_number":
                    po_number,

                    "currency":
                    currency,

                    "translation_confidence":
                    translation_confidence,

                    "warnings": [

                        "PO number missing.",

                        "Item codes unavailable "
                        "for automatic validation."

                    ],

                    "business_validation_errors":
                    [],

                    "discrepancies":
                    [],

                    "validated_invoice":
                    invoice_json
                }

            # -----------------------------------------
            # CASE 3
            # Nothing usable
            # -----------------------------------------

            return {

                "status":
                "REJECTED",

                "stage":
                "BUSINESS_VALIDATION",

                "recommendation":
                "Reject",

                "invoice_id":
                invoice_id,

                "vendor_id":
                vendor_id,

                "po_number":
                po_number,

                "currency":
                currency,

                "translation_confidence":
                translation_confidence,

                "warnings":
                [],

                "business_validation_errors": [

                    "PO number missing.",

                    "Vendor unavailable."

                ],

                "discrepancies":
                [],

                "validated_invoice":
                invoice_json
            }

        # =================================================
        # STRICT PO VALIDATION
        # =================================================

        po_items = po_record.get(
            "line_items",
            []
        )

        po_currency = po_record.get(
            "currency",
            "UNKNOWN"
        )

        # -----------------------------------------
        # Currency Validation
        # -----------------------------------------

        if (

            currency != "UNKNOWN"

            and

            po_currency != "UNKNOWN"

        ):

            if currency != po_currency:

                business_errors.append(

                    f"Currency mismatch: "

                    f"Invoice={currency}, "

                    f"PO={po_currency}"

                )

        # -----------------------------------------
        # Tolerances
        # -----------------------------------------

        tolerance = self.rules.get(
            "tolerances",
            {}
        ).get(
            "price_difference_percent",
            10
        )

        # -----------------------------------------
        # Validate Items
        # -----------------------------------------

        for invoice_item in items:

            matched_po_item = self.find_matching_po_item(

                invoice_item,

                po_items

            )

            # -------------------------------------
            # No Match Found
            # -------------------------------------

            if not matched_po_item:

                discrepancies.append({

                    "severity":
                    "HIGH",

                    "message":

                    f"No matching PO item found "

                    f"for "

                    f"'{invoice_item.get('description')}'"

                })

                continue

            # -------------------------------------
            # Quantity Validation
            # -------------------------------------

            invoice_qty = invoice_item.get(
                "qty",
                0
            )

            po_qty = matched_po_item.get(
                "qty",
                0
            )

            if invoice_qty != po_qty:

                discrepancies.append({

                    "severity":
                    "MEDIUM",

                    "message":

                    f"Quantity mismatch for "

                    f"{invoice_item.get('description')} "

                    f"(Invoice={invoice_qty}, "

                    f"PO={po_qty})"

                })

            # -------------------------------------
            # Price Validation
            # -------------------------------------

            invoice_price = invoice_item.get(
                "unit_price",
                0
            )

            po_price = matched_po_item.get(
                "unit_price",
                0
            )

            if po_price == 0:

                continue

            diff_percent = abs(

                (
                    invoice_price -
                    po_price
                ) / po_price

            ) * 100

            if diff_percent > tolerance:

                discrepancies.append({

                    "severity":
                    "MEDIUM",

                    "message":

                    f"Price mismatch for "

                    f"{invoice_item.get('description')} "

                    f"(Invoice={invoice_price}, "

                    f"PO={po_price})"

                })

            # -------------------------------------
            # Line Currency Validation
            # -------------------------------------

            po_item_currency = matched_po_item.get(
                "currency",
                "UNKNOWN"
            )

            if (

                currency != "UNKNOWN"

                and

                po_item_currency != "UNKNOWN"

                and

                currency != po_item_currency

            ):

                business_errors.append(

                    f"Currency mismatch for "
                    f"{invoice_item.get('description')} "
                    f"(Invoice={currency}, "
                    f"ERP={po_item_currency})"

                )

        # =================================================
        # FINAL STATUS
        # =================================================

        status = "PASSED"
        recommendation = "Approve"

        if business_errors:

            status = "REJECTED"
            recommendation = "Reject"

        elif discrepancies:

            status = "MANUAL_REVIEW"
            recommendation = "Manual Review"

        elif warnings:

            status = "APPROVED_WITH_WARNING"
            recommendation = "Approve With Warning"

        # =================================================
        # FINAL RESPONSE
        # =================================================

        return {

            "status":
            status,

            "stage":
            "BUSINESS_VALIDATION",

            "recommendation":
            recommendation,

            "invoice_id":
            invoice_id,

            "vendor_id":
            vendor_id,

            "po_number":
            po_number,

            "currency":
            currency,

            "translation_confidence":
            translation_confidence,

            "warnings":
            warnings,

            "business_validation_errors":
            business_errors,

            "discrepancies":
            discrepancies,

            "validated_invoice":
            invoice_json
        }
