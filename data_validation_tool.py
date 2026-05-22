from schemas.validation_schema import ValidationResult

from agents.data_validation import (
    DataValidationAgent
)

# ---------------------------------------------------
# Create Agent Instance
# ---------------------------------------------------

agent = DataValidationAgent()

# ---------------------------------------------------
# Tool
# ---------------------------------------------------

def data_validation_tool(payload):

    validated_results = []
    failed_results = []

    for item in payload:

        try:

            structured_data = item.get(
                "structured_data",
                {}
            )

            agent_result = agent.validate(

                structured_data

            )

            # -----------------------------------------
            # Validate Final Output Schema
            # -----------------------------------------

            validated = ValidationResult(

                **agent_result

            )

            validated_results.append({

                "file_name":
                item.get("file_name"),

                "validation_result":
                validated.model_dump()

            })

        except Exception as e:

            failed_results.append({

                "file_name":
                item.get("file_name"),

                "error":
                str(e),

                "bad_data":
                structured_data

            })

    # -------------------------------------------------
    # Final Response
    # -------------------------------------------------

    return {

        "validated_results":
        validated_results,

        "failed_results":
        failed_results
    }
