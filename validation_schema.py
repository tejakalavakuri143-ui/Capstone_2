from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    status: str
    stage: str
    errors: List[str] = Field(default_factory=list)
    invoice_id: Optional[str] = None
    vendor_id: Optional[str] = None
    currency: Optional[str] = None
    translation_confidence: Optional[float] = None
    data_validation_errors: List[str] = Field(default_factory=list)
    business_validation_errors: List[str] = Field(default_factory=list)
    discrepancies: List[Any] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    recommendation: Optional[str] = None
    validated_invoice: Optional[dict] = None
