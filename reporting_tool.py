from agents.reporting_agent import run_reporting_agent
from schemas.reporting_schema import ValidationReport


def reporting_tool(validation_result: dict) -> dict:
    """
    Runs Reporting Agent and returns final report paths.
    """

    result = run_reporting_agent(validation_result)

    validated = ValidationReport(**result)

    return validated.model_dump()