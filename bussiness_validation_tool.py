from agents.businness_validation import (
    BusinessValidationAgent
)

agent = BusinessValidationAgent()


def business_validation_tool(
    payload
):
    """
    Run ERP/business validation.
    """

    try:

        result = agent.validate(
            payload
        )

        return result

    except Exception as e:

        return {

            "status":
            "FAILED",

            "stage":
            "BUSINESS_VALIDATION",

            "error":
            str(e)

        }