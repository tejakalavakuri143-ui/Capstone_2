from pydantic import BaseModel
from typing import List


class ValidationReport(BaseModel):
    invoice_id: str
    status: str
    recommendation: str
    json_report_path: str
    html_report_path: str