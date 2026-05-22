from schemas.translation_schema import TranslatedInvoice
from agents.translation_agent import translate_invoice

def translation_tool(text: str):

    result = translate_invoice(text)

    validated = TranslatedInvoice(**result)

    return validated.model_dump()