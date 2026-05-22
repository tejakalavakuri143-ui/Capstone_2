from pydantic import BaseModel
from typing import Optional, List
 
 
class ExtractedLineItem(BaseModel):
    item_code: Optional[str] = None
    qty: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None
 
 
class ExtractedHeader(BaseModel):
    invoice_no: Optional[str] = None
    vendor_id: Optional[str] = None
    currency: Optional[str] = None
    total_amount: Optional[float] = None
 
 
class ExtractedInvoice(BaseModel):
    raw_text: str
    extraction_confidence: float
    header: ExtractedHeader
    line_items: List[ExtractedLineItem]
 