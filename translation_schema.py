from pydantic import BaseModel
from typing import List, Optional


class TranslatedLineItem(BaseModel):

    # Optional because many invoices
    # do not contain SKU/item codes

    item_code: Optional[str] = None

    description: str

    qty: float

    unit_price: float

    total: float


class TranslatedHeader(BaseModel):

    invoice_no: str

    vendor_id: str

    currency: str

    total_amount: float

    # Optional because some invoices
    # may not contain PO number

    po_number: Optional[str] = None

    invoice_date: Optional[str] = None


class TranslatedInvoice(BaseModel):

    source_language: Optional[str] = "unknown"

    target_language: Optional[str] = "en"

    translation_confidence: float

    header: TranslatedHeader

    line_items: List[TranslatedLineItem]