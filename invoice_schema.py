from typing import List, Optional

from pydantic import BaseModel


class InvoiceHeader(BaseModel):
    invoice_no: str
    vendor_id: str
    currency: str
    total_amount: float
    po_number: Optional[str] = None
    invoice_date: Optional[str] = None


class LineItem(BaseModel):
    item_code: Optional[str] = None
    description: Optional[str] = None
    qty: float
    unit_price: float
    total: float


class Invoice(BaseModel):
    header: InvoiceHeader
    line_items: List[LineItem]
    translation_confidence: float
