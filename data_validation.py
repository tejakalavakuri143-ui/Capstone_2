import logging
from pathlib import Path

import yaml

from schemas.invoice_schema import Invoice


class DataValidationAgent:

    def __init__(self):

        rules_path = Path(__file__).resolve().parent.parent / "config" / "rules.yaml"

        with open(rules_path) as f:

            self.rules = yaml.safe_load(f)

    # -------------------------------------------------
    # MAIN VALIDATION
    # -------------------------------------------------

    def validate(
        self,
        invoice_json
    ):

        data_errors = []

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

        currency = header.get(
            "currency",
            "UNKNOWN"
        )

        translation_confidence = invoice_json.get(

            "translation_confidence",

            0

        )

        # ---------------------------------------------
        # SCHEMA VALIDATION
        # ---------------------------------------------

        try:

            invoice = Invoice(
                **invoice_json
            )

        except Exception as e:

            logging.error(

                f"Schema validation failed "
                f"for {invoice_id}: {e}"

            )

            return {

                "status":
                "FAILED",

                "stage":
                "SCHEMA_VALIDATION",

                "invoice_id":
                invoice_id,

                "vendor_id":
                vendor_id,

                "currency":
                currency,

                "translation_confidence":
                translation_confidence,

                "errors":
                str(e)

            }

        # ---------------------------------------------
        # CURRENCY VALIDATION
        # ---------------------------------------------

        required_header_fields = self.rules.get(
            "required_fields",
            {}
        ).get(
            "header",
            []
        )

        for field in required_header_fields:

            value = getattr(
                invoice.header,
                field,
                None
            )

            if value in (None, "", "UNKNOWN"):

                data_errors.append(
                    f"Missing required header field: {field}"
                )

        required_line_fields = self.rules.get(
            "required_fields",
            {}
        ).get(
            "line_item",
            []
        )

        for index, item in enumerate(invoice.line_items, start=1):

            for field in required_line_fields:

                value = getattr(
                    item,
                    field,
                    None
                )

                if value in (None, "", "UNKNOWN"):

                    data_errors.append(
                        f"Missing required line item field "
                        f"{field} at row {index}"
                    )

            if item.total != round(item.qty * item.unit_price, 2):

                data_errors.append(
                    f"Line item total mismatch at row {index}"
                )

        allowed_currencies = self.rules.get(

            "accepted_currencies",

            []

        )

        if invoice.header.currency not in allowed_currencies:

            data_errors.append(
                "Invalid currency"
            )

        # ---------------------------------------------
        # AMOUNT VALIDATION
        # ---------------------------------------------

        if invoice.header.total_amount <= 0:

            data_errors.append(
                "Invalid invoice amount"
            )

        # ---------------------------------------------
        # LINE ITEM VALIDATION
        # ---------------------------------------------

        if not invoice.line_items:

            data_errors.append(
                "No line items found"
            )

        # ---------------------------------------------
        # TRANSLATION CONFIDENCE CHECK
        # OPTIONAL
        # ---------------------------------------------

        min_confidence = self.rules.get(

            "validation_policies",

            {}

        ).get(

            "minimum_translation_confidence",

            self.rules.get("minimum_translation_confidence", 0.0)

        )

        if translation_confidence < min_confidence:

            data_errors.append(

                f"Low translation confidence: "
                f"{translation_confidence}"

            )

        # ---------------------------------------------
        # VALIDATION FAILED
        # ---------------------------------------------

        if data_errors:

            logging.warning(

                f"Data validation failed "
                f"for {invoice_id}"

            )

            return {

                "status":
                "FAILED",

                "stage":
                "DATA_VALIDATION",

                "invoice_id":
                invoice_id,

                "vendor_id":
                vendor_id,

                "currency":
                currency,

                "translation_confidence":
                translation_confidence,

                "data_validation_errors":
                data_errors,

                "line_items":
                invoice.model_dump().get("line_items", []),

                "validated_invoice":
                invoice.model_dump()

            }

        # ---------------------------------------------
        # VALIDATION PASSED
        # ---------------------------------------------

        logging.info(

            f"Data validation passed "
            f"for {invoice_id}"

        )

        return {

            "status":
            "PASSED",

            "stage":
            "DATA_VALIDATION",

            "validated_invoice":
            invoice.model_dump(),

            "invoice_id":
            invoice_id,

            "vendor_id":
            vendor_id,

            "currency":
            currency,

            "translation_confidence":
            translation_confidence

        }
