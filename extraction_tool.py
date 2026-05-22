from mcp.server.fastmcp import FastMCP
 
from schemas.extraction_schema import ExtractedInvoice
 
from agents.extractor_agent import extractor_agent
 
 

def extraction_tool(payload: dict):
 
    result = extractor_agent(payload)
 
    validated = ExtractedInvoice(**result)
 
    return validated.model_dump()
 