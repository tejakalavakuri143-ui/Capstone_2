from typing import TypedDict, List


class InvoiceState(TypedDict, total=False):

    files: List[str]

    extracted_data: list

    translated_data: list