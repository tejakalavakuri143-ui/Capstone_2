import re

from integration.langfuse_tracing import guardrail_event


class ResponsibleAIGuardrails:

    # -------------------------------------------------
    # Prompt Injection Detection
    # -------------------------------------------------

    @staticmethod
    def detect_prompt_injection(
        text: str
    ) -> bool:

        suspicious_patterns = [

            "ignore previous instructions",

            "reveal system prompt",

            "bypass security",

            "pretend you are",

            "act as",

            "shutdown",

            "delete database",

            "password",

            "api key",

            "secret"

        ]

        lowered = text.lower()

        detected = any(

            pattern in lowered

            for pattern in suspicious_patterns

        )

        guardrail_event(
            "prompt_injection",
            passed=not detected,
            input_data={"text_preview": text[:500]},
            metadata={"detected": detected},
        )

        return detected

    # -------------------------------------------------
    # PII Sanitization
    # -------------------------------------------------

    @staticmethod
    def sanitize_pii(
        text: str
    ) -> str:

        # ---------------------------------------------
        # Email Masking
        # ---------------------------------------------

        original_text = text

        text = re.sub(

            r'[\w\.-]+@[\w\.-]+',

            '[EMAIL]',

            text

        )

        # ---------------------------------------------
        # Phone Masking
        # ---------------------------------------------

        text = re.sub(

            r'\+?\d[\d\s\-]{8,}',

            '[PHONE]',

            text

        )

        # ---------------------------------------------
        # Account Numbers
        # ---------------------------------------------

        text = re.sub(

            r'\b\d{10,18}\b',

            '[ACCOUNT_NUMBER]',

            text

        )

        changed = text != original_text

        guardrail_event(
            "pii_sanitization",
            passed=True,
            input_data={"text_length": len(original_text or "")},
            metadata={"redacted": changed},
        )

        return text

    # -------------------------------------------------
    # Groundedness Validation
    # -------------------------------------------------

    @staticmethod
    def check_rag_groundedness(

        answer: str,

        context: str

    ) -> bool:

        if not context:

            guardrail_event(
                "rag_groundedness",
                passed=False,
                input_data={"answer_preview": answer[:500]},
                metadata={"reason": "missing_context"},
            )

            return False

        answer_words = set(

            answer.lower().split()

        )

        context_words = set(

            context.lower().split()

        )

        overlap = len(

            answer_words & context_words

        )

        score = overlap / max(

            len(answer_words),

            1

        )

        passed = score >= 0.25

        guardrail_event(
            "rag_groundedness",
            passed=passed,
            input_data={"answer_preview": answer[:500]},
            metadata={"score": round(score, 4), "threshold": 0.25},
        )

        return passed
