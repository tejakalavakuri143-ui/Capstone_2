import json
import re
import logging
import os
from datetime import datetime

from transformers import (
    MarianMTModel,
    MarianTokenizer
)

from langdetect import detect

from langchain_aws import (
    ChatBedrockConverse
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

from rapidfuzz import process

# --------------------------------------------------
# Integration Imports
# --------------------------------------------------

from integration.config import (

    BEDROCK_MODEL,

    BEDROCK_REGION,

    USE_BEDROCK,

    MIN_TRANSLATION_CONFIDENCE

)

from integration.guardrails import (

    ResponsibleAIGuardrails

)

# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.ERROR
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# MarianMT Models
# --------------------------------------------------

MODEL_MAP = {

    "de":
    "Helsinki-NLP/opus-mt-de-en",

    "es":
    "Helsinki-NLP/opus-mt-es-en",

    "fr":
    "Helsinki-NLP/opus-mt-fr-en",

    "it":
    "Helsinki-NLP/opus-mt-it-en"

}

DEFAULT_MODEL = (
    "Helsinki-NLP/opus-mt-mul-en"
)

models = {}

tokenizers = {}

# --------------------------------------------------
# Load Translation Model
# --------------------------------------------------

def get_model(lang):

    model_name = MODEL_MAP.get(
        lang,
        DEFAULT_MODEL
    )

    if model_name not in models:

        tokenizers[model_name] = (

            MarianTokenizer
            .from_pretrained(
                model_name
            )

        )

        models[model_name] = (

            MarianMTModel
            .from_pretrained(
                model_name
            )

        )

    return (

        tokenizers[model_name],

        models[model_name]

    )

_chain = None

# --------------------------------------------------
# Prompt Template
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
"""
You are an AI invoice translation and extraction agent.

The input invoice text may contain multilingual fields.

Tasks:
1. Detect the language
2. Translate invoice into English
3. Extract structured invoice data

Return ONLY valid JSON.

FORMAT:

{{
  "header": {{
    "vendor_name": "",
    "invoice_no": "",
    "po_number": "",
    "invoice_date": "",
    "currency": "",
    "total_amount": 0
  }},
  "line_items": [
    {{
      "item_code": "",
      "description": "",
      "qty": 0,
      "unit_price": 0,
      "total": 0
    }}
  ],
  "translation_confidence": 0.0
}}

RULES:

- Convert dates to YYYY-MM-DD
- Convert currency symbols to ISO code
- Missing fields = null
- Do NOT hallucinate rows
- Do NOT create fake item codes
- Translate descriptions to English
- Keep numeric values accurate

OCR FIXES:

- I ↔ 1
- l ↔ 1
- O ↔ 0

Examples:
INV-I005 → INV-1005
INV-1L005 → INV-1005

IMPORTANT:

- subtotal/tax/VAT/GST are NOT line items
- only purchased items belong in line_items

Return ONLY valid JSON.

Invoice Text:
{invoice}
"""
)

def get_chain():
    if not USE_BEDROCK:
        raise RuntimeError("Bedrock extraction disabled. Set INVOICE_USE_BEDROCK=1 to enable it.")
    global _chain
    if _chain is None:
        llm = ChatBedrockConverse(
            model=BEDROCK_MODEL,
            region_name=BEDROCK_REGION,
            temperature=0,
            max_tokens=700,
        )
        _chain = prompt | llm
    return _chain

# --------------------------------------------------
# Vendor Master Data
# --------------------------------------------------

BASE_DIR = os.path.dirname(

    os.path.dirname(
        os.path.abspath(__file__)
    )

)

VENDOR_PATH = os.path.join(

    BASE_DIR,

    "data",

    "ERP_mockdata",

    "vendors.json"

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

with open(VENDOR_PATH) as f:

    vendors = json.load(f)

with open(PO_MASTER_PATH) as f:

    po_master = json.load(f)

with open(SKU_MASTER_PATH) as f:

    sku_master = json.load(f)

vendor_map = {

    v["vendor_name"]: v["vendor_id"]

    for v in vendors

}

vendor_currency_map = {

    v["vendor_id"]: v.get("currency")

    for v in vendors

}

sku_codes = {

    item["item_code"]

    for item in sku_master

}

# --------------------------------------------------
# Detect Language
# --------------------------------------------------

def detect_language(text):

    try:

        return detect(text)

    except:

        return "en"

# --------------------------------------------------
# Translate Text
# --------------------------------------------------

def translate_text(text):

    lang = detect_language(text)

    # Already English
    if lang == "en":

        return text

    try:
        tokenizer, model = get_model(lang)

        tokens = tokenizer(

            text,

            return_tensors="pt",

            truncation=True

        )

        output = model.generate(
            **tokens
        )

        translated = tokenizer.decode(

            output[0],

            skip_special_tokens=True

        )

        return translated
    except Exception as exc:
        logger.warning("Local translation failed; using original text: %s", exc)
        return text

# --------------------------------------------------
# Translation Confidence
# --------------------------------------------------

def compute_translation_confidence(

    src,

    translated

):

    src_words = len(src.split())

    tgt_words = len(translated.split())

    if src_words == 0:

        return 0.0

    confidence = (

        min(src_words, tgt_words)

        / max(src_words, tgt_words)

    )

    return round(confidence, 2)

# --------------------------------------------------
# JSON Sanitizer
# --------------------------------------------------

def sanitize_json(output: str) -> str:

    output = output.replace(
        "```json",
        ""
    )

    output = output.replace(
        "```",
        ""
    )

    output = output.strip()

    fixed_lines = []

    for line in output.splitlines():

        fixed = re.sub(

            r'^(\s*)([A-Za-z_][A-Za-z0-9_]*)(":\s)',

            r'\1"\2\3',

            line

        )

        fixed_lines.append(fixed)

    output = "\n".join(
        fixed_lines
    )

    # Remove trailing commas
    output = re.sub(

        r',(\s*[}\]])',

        r'\1',

        output

    )

    return output

# --------------------------------------------------
# Normalize Invoice Number
# --------------------------------------------------

def normalize_invoice_number(
    invoice_no
):

    if not invoice_no:

        return invoice_no

    invoice_no = invoice_no.upper()

    replacements = {

        "INV-I":
        "INV-1",

        "INV-L":
        "INV-1",

        "INV-IL":
        "INV-1",

        "INV-1L":
        "INV-1"

    }

    for old, new in replacements.items():

        invoice_no = invoice_no.replace(
            old,
            new
        )

    invoice_no = re.sub(

        r'(?<=\d)L(?=\d)',

        '1',

        invoice_no

    )

    invoice_no = re.sub(

        r'(?<=\d)I(?=\d)',

        '1',

        invoice_no

    )

    invoice_no = re.sub(

        r'(?<=\d)O(?=\d)',

        '0',

        invoice_no

    )

    return invoice_no

# --------------------------------------------------
# Valid Keys
# --------------------------------------------------

VALID_LINE_ITEM_KEYS = {

    "item_code",

    "description",

    "qty",

    "unit_price",

    "total"

}

VALID_HEADER_KEYS = {

    "vendor_name",

    "invoice_no",

    "po_number",

    "invoice_date",

    "currency",

    "total_amount"

}

# --------------------------------------------------
# Cleanup Structured Output
# --------------------------------------------------

def clean_parsed_invoice(
    data: dict
) -> dict:

    # -----------------------------------------
    # HEADER
    # -----------------------------------------

    if (

        "header" in data

        and isinstance(
            data["header"],
            dict
        )

    ):

        data["header"] = {

            k: v

            for k, v in data[
                "header"
            ].items()

            if k in VALID_HEADER_KEYS

        }

    # -----------------------------------------
    # LINE ITEMS
    # -----------------------------------------

    cleaned_items = []

    INVALID_ROWS = [

        "subtotal",

        "tax",

        "vat",

        "gst",

        "total",

        "grand total"

    ]

    if (

        "line_items" in data

        and isinstance(
            data["line_items"],
            list
        )

    ):

        for item in data["line_items"]:

            if not isinstance(
                item,
                dict
            ):

                continue

            item = {

                k: v

                for k, v in item.items()

                if k in VALID_LINE_ITEM_KEYS

            }

            description = str(

                item.get(
                    "description",
                    ""
                )

            ).lower()

            # Remove subtotal rows
            if any(

                word in description

                for word in INVALID_ROWS

            ):

                continue

            # Keep only meaningful rows
            if (

                item.get("item_code")

                or item.get("qty")

                or item.get("unit_price")

            ):

                cleaned_items.append(
                    item
                )

    data["line_items"] = cleaned_items

    return data

# --------------------------------------------------
# Extract Structured Fields
# --------------------------------------------------

def extract_invoice_fields(
    invoice_text
):

    try:
        response = get_chain().invoke({

            "invoice":
            invoice_text

        })

        output = response.content.strip()
    except Exception as exc:
        logger.warning("Bedrock extraction failed; using deterministic parser: %s", exc)
        return heuristic_invoice_parse(invoice_text)

    # Remove markdown
    output = output.replace(
        "```json",
        ""
    )

    output = output.replace(
        "```",
        ""
    )

    output = output.strip()

    # Extract JSON block
    start = output.find("{")

    end = output.rfind("}")

    if start != -1 and end != -1:

        output = output[
            start:end+1
        ]

    # Direct Parse
    try:

        parsed = json.loads(
            output
        )

        return clean_parsed_invoice(
            parsed
        )

    except json.JSONDecodeError:

        pass

    # Sanitized Parse
    try:

        sanitized = sanitize_json(
            output
        )

        parsed = json.loads(
            sanitized
        )

        return clean_parsed_invoice(
            parsed
        )

    except json.JSONDecodeError as e:

        logger.warning("Invalid JSON from LLM; using deterministic parser: %s", e)
        return heuristic_invoice_parse(invoice_text)


def _first_match(patterns, text, default=None, flags=re.IGNORECASE):
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1).strip()
    return default


def _parse_amount(value, default=0.0):
    if value in (None, ""):
        return default
    cleaned = re.sub(r"[^0-9.,-]", "", str(value)).replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return default


def _detect_currency(text):
    upper = text.upper()
    if "€" in text or " EUR" in upper:
        return "EUR"
    if "₹" in text or " INR" in upper:
        return "INR"
    if "£" in text or " GBP" in upper:
        return "GBP"
    return "USD" if "$" in text or " USD" in upper else None


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def _normalize_date(value):
    if not value:
        return None

    raw = str(value).strip()

    for fmt in [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%d %B %Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
    ]:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass

    match = re.search(
        r"(\d{1,2})\s+de\s+([A-Za-zÁÉÍÓÚáéíóúñÑ]+)\s+de\s+(\d{4})",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        month = MONTHS.get(match.group(2).lower())
        if month:
            return datetime(
                int(match.group(3)),
                month,
                int(match.group(1))
            ).date().isoformat()

    match = re.search(
        r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        month = MONTHS.get(match.group(2).lower())
        if month:
            return datetime(
                int(match.group(3)),
                month,
                int(match.group(1))
            ).date().isoformat()

    match = re.search(
        r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        month = MONTHS.get(match.group(1).lower())
        if month:
            return datetime(
                int(match.group(3)),
                month,
                int(match.group(2))
            ).date().isoformat()

    return None


def _protect_document_tokens(text):
    replacements = {}

    def replace(match):
        key = f"__DOCID_{len(replacements)}__"
        replacements[key] = match.group(0)
        return key

    protected = re.sub(
        r"\b(?:INV|FAC|RE|PO|SKU)[-\s]?[A-Z0-9]+(?:-[A-Z0-9]+)*\b",
        replace,
        text,
        flags=re.IGNORECASE,
    )
    return protected, replacements


def _restore_document_tokens(text, replacements):
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def _find_po_record(po_number):
    if not po_number:
        return None
    normalized = po_number.replace(" ", "-").upper()
    for po in po_master:
        if str(po.get("po_number", "")).upper() == normalized:
            return po
    return None


def _infer_vendor_id(vendor_name):
    if not vendor_name:
        return None

    match = process.extractOne(
        vendor_name,
        vendor_map.keys()
    )

    if match and match[1] > 80:
        return vendor_map[match[0]]

    return None


def _vendor_po_records(vendor_id):
    if not vendor_id:
        return []

    return [
        po
        for po in po_master
        if po.get("vendor_id") == vendor_id
    ]


def _infer_vendor_name(text):
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    compact_text = " ".join(text.split())
    candidates = [first_line, compact_text[:120]]

    best = None
    best_score = 0
    for candidate in candidates:
        match = process.extractOne(candidate, vendor_map.keys())
        if match and match[1] > best_score:
            best = match[0]
            best_score = match[1]

    return best if best_score > 70 else ""


def _infer_item_code(description, po_record=None, vendor_id=None):
    if not description:
        return None

    records = []

    if po_record:
        records.append(po_record)

    records.extend(_vendor_po_records(vendor_id))

    if not records:
        records = po_master

    po_items = [
        item
        for record in records
        for item in record.get("line_items", [])
    ]

    if not po_items:
        return None

    match = process.extractOne(
        description,
        [item.get("description", "") for item in po_items],
    )
    if not match or match[1] < 75:
        return None

    for item in po_items:
        if item.get("description") == match[0]:
            return item.get("item_code")
    return None


def _parse_labeled_line_item(text, po_record, vendor_id=None):
    description = _first_match([
        r"description\s*[:#-]?\s*(.+?)\s+quantity\s*[:#-]?",
        r"item\s*description\s*[:#-]?\s*(.+?)\s+quantity\s*[:#-]?",
    ], text)
    qty = _parse_amount(_first_match([
        r"quantity\s*[:#-]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"qty\s*[:#-]?\s*([0-9]+(?:\.[0-9]+)?)",
    ], text))
    unit_price = _parse_amount(_first_match([
        r"unit\s*price\s*[:#-]?\s*[$€£₹]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"price\s*[:#-]?\s*[$€£₹]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    ], text))

    if not description or qty <= 0 or unit_price <= 0:
        return None

    return {
        "item_code": _infer_item_code(description, po_record, vendor_id),
        "description": description.strip(),
        "qty": qty,
        "unit_price": unit_price,
        "total": round(qty * unit_price, 2),
    }


def _parse_erp_description_line_items(text, po_record=None, vendor_id=None):
    records = []

    if po_record:
        records.append(po_record)

    records.extend(_vendor_po_records(vendor_id))

    if not records:
        records = po_master

    line_items = []
    seen_codes = set()

    for record in records:
        for erp_item in record.get("line_items", []):
            description = erp_item.get("description", "")
            item_code = erp_item.get("item_code")

            if not description or item_code in seen_codes:
                continue

            words = description.split()
            desc_pattern = r"\s+".join(
                re.escape(part)
                for part in words
            )
            match = re.search(
                desc_pattern,
                text,
                flags=re.IGNORECASE,
            )

            if not match and len(words) > 2:
                partial_pattern = (
                    re.escape(words[0])
                    + r"(?:\s+\w+){0,3}\s+"
                    + re.escape(words[-1])
                )
                match = re.search(
                    partial_pattern,
                    text,
                    flags=re.IGNORECASE,
                )

            if not match:
                continue

            amount_tokens = re.findall(
                r"[$€£₹]?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)",
                text[match.end():match.end() + 120],
            )

            if len(amount_tokens) < 2:
                continue

            qty = _parse_amount(amount_tokens[0])
            unit_price = _parse_amount(amount_tokens[1])
            total = (
                _parse_amount(amount_tokens[2])
                if len(amount_tokens) > 2
                else round(qty * unit_price, 2)
            )

            if qty <= 0 or unit_price <= 0:
                continue

            line_items.append({
                "item_code": item_code if item_code in sku_codes else None,
                "description": description,
                "qty": qty,
                "unit_price": unit_price,
                "total": total,
            })
            seen_codes.add(item_code)

    return line_items


def heuristic_invoice_parse(invoice_text):
    invoice_no = _first_match([
        r"invoice\s*(?:no|number|#)\s*[:#-]?\s*([A-Z0-9-]+)",
        r"(?:billing|bill|rechnung|factura)\s*(?:no|number|nummer|nr|#)?\s*[:#-]?\s*([A-Z]+-[0-9][A-Z0-9-]*)",
        r"\b(INV[-\s][A-Z0-9-]+)\b",
        r"\b(FAC[-\s][A-Z0-9-]+)\b",
        r"\b(RE[-\s][A-Z0-9-]+)\b",
    ], invoice_text, "UNKNOWN")
    po_number = _first_match([
        r"\b(PO[-\s]?\d+)\b",
        r"purchase\s*order\s*[:#-]?\s*([A-Z0-9-]+)",
    ], invoice_text)
    date = _first_match([
        r"invoice\s*date\s*[:#-]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
        r"invoice\s*date\s*[:#-]?\s*([0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4})",
        r"date\s+of\s+invoice\s*[:#-]?\s*([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4})",
        r"date\s*[:#-]?\s*([0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4})",
        r"date\s*[:#-]?\s*([A-Za-z]+\s+[0-9]{1,2},\s*[0-9]{4})",
        r"date\s*[:#-]?\s*([0-9]{1,2}-[A-Za-z]+-[0-9]{4})",
        r"date\s*[:#-]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
        r"date\s*[:#-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
        r"rechnungsdatum\s*[:#-]?\s*([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4})",
        r"fecha\s*[:#-]?\s*([0-9]{1,2}\s+de\s+[A-Za-zÁÉÍÓÚáéíóúñÑ]+\s+de\s+[0-9]{4})",
        r"\bbate\s*[:#-]?\s*([0-9]{1,2}-[A-Za-z]+-[0-9]{4})",
    ], invoice_text)
    date = _normalize_date(date)
    total_amount = _parse_amount(_first_match([
        r"total\s*amount\s*due\s*[:#-]?\s*[$€£₹]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"total\s*amount\s*[:#-]?\s*[$€£₹]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"amount\s*due\s*[:#-]?\s*[$€£₹]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"\bgrand\s*total\s*[:#-]?\s*[$€£₹]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"\btotal\s*[:#-]?\s*[$€£₹]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    ], invoice_text))

    vendor_name = _first_match([
        r"vendor\s*[:#-]?\s*(.+?)(?:\s{2,}| invoice| po |$)",
        r"supplier\s*[:#-]?\s*(.+?)(?:\s{2,}| invoice| po |$)",
    ], invoice_text, "")
    if not vendor_name:
        vendor_name = _infer_vendor_name(invoice_text)

    vendor_id = _infer_vendor_id(vendor_name)
    currency = (
        _detect_currency(invoice_text)
        or vendor_currency_map.get(vendor_id)
        or "USD"
    )
    po_record = _find_po_record(po_number)

    sku_matches = re.findall(
        r"\b(SKU[-\s]?\d+)\b\s+(.+?)\s+([0-9]+(?:\.[0-9]+)?)\s+[$€£₹]?\s*([0-9]+(?:\.[0-9]+)?)(?:\s*[$€£₹])?\s+[$€£₹]?\s*([0-9]+(?:\.[0-9]+)?)(?:\s*[$€£₹])?",
        invoice_text,
        flags=re.IGNORECASE,
    )
    line_items = [
        {
            "item_code": code.replace(" ", "-").upper(),
            "description": desc.strip(),
            "qty": _parse_amount(qty),
            "unit_price": _parse_amount(unit_price),
            "total": _parse_amount(total),
        }
        for code, desc, qty, unit_price, total in sku_matches
    ]

    if not line_items:
        line_items = _parse_erp_description_line_items(
            invoice_text,
            po_record,
            vendor_id
        )

    if not line_items:
        labeled_item = _parse_labeled_line_item(
            invoice_text,
            po_record,
            vendor_id
        )
        if labeled_item:
            line_items = [labeled_item]

    return clean_parsed_invoice({
        "header": {
            "vendor_name": vendor_name,
            "invoice_no": invoice_no,
            "po_number": po_number,
            "invoice_date": date,
            "currency": currency,
            "total_amount": total_amount,
        },
        "line_items": line_items,
        "translation_confidence": 0.75,
    })

# --------------------------------------------------
# Vendor Mapping
# --------------------------------------------------

def map_vendor_id(header):

    vendor_name = header.get(
        "vendor_name",
        ""
    )

    if not vendor_name and header.get("vendor_id"):
        return header

    match = process.extractOne(

        vendor_name,

        vendor_map.keys()

    )

    if match and match[1] > 80:

        vendor_id = vendor_map[
            match[0]
        ]

    else:

        vendor_id = "UNKNOWN"

    header["vendor_id"] = vendor_id

    header.pop(
        "vendor_name",
        None
    )

    return header

# --------------------------------------------------
# Main Translation Agent
# --------------------------------------------------

def translate_invoice(
    raw_text
):

    # -----------------------------------------
    # PII Sanitization
    # -----------------------------------------

    protected_text, protected_tokens = _protect_document_tokens(
        raw_text
    )

    sanitized_text = (

        ResponsibleAIGuardrails
        .sanitize_pii(
            protected_text
        )

    )

    sanitized_text = _restore_document_tokens(
        sanitized_text,
        protected_tokens
    )

    # -----------------------------------------
    # Translation
    # -----------------------------------------

    translated_text = translate_text(
        sanitized_text
    )

    # -----------------------------------------
    # Extraction
    # -----------------------------------------

    structured = extract_invoice_fields(
        translated_text
    )

    if "[PHONE]" in translated_text:
        raw_structured = extract_invoice_fields(
            raw_text
        )

        if raw_structured.get("line_items"):
            structured["line_items"] = raw_structured["line_items"]

    # -----------------------------------------
    # Invoice Number Normalization
    # -----------------------------------------

    if "header" in structured:

        invoice_no = structured[
            "header"
        ].get(
            "invoice_no"
        )

        structured["header"][
            "invoice_no"
        ] = normalize_invoice_number(
            invoice_no
        )

    # -----------------------------------------
    # Vendor Mapping
    # -----------------------------------------

    structured.setdefault("header", {})

    structured["header"] = map_vendor_id(

        structured["header"]

    )

    # -----------------------------------------
    # Translation Confidence
    # -----------------------------------------

    confidence = (

        compute_translation_confidence(

            raw_text,

            translated_text

        )

    )

    structured[
        "translation_confidence"
    ] = confidence

    # -----------------------------------------
    # Low Confidence Warning
    # -----------------------------------------

    if (

        confidence
        < MIN_TRANSLATION_CONFIDENCE

    ):

        structured[
            "translation_warning"
        ] = (

            "Low translation confidence."

        )

    return structured
