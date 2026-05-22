import re

from integration.langfuse_tracing import guardrail_event

class ResponsibleAIGuardrails:
    
    @staticmethod
    def sanitize_pii(text: str) -> str:
        """
        Data Privacy Guardrail: Scrubs sensitive personal and financial info 
        from extracted invoice text before it is sent to external LLMs.
        """
        if not text:
            return text
            
        # 1. Redact Email Addresses
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        sanitized = re.sub(email_pattern, '[REDACTED_EMAIL]', text)
        
        # 2. Redact Phone Numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        sanitized = re.sub(phone_pattern, '[REDACTED_PHONE]', sanitized)
        
        # 3. Redact Credit Card / Bank Account patterns (13-16 digit numbers)
        account_pattern = r'\b(?:\d[ -]*?){13,16}\b'
        sanitized = re.sub(account_pattern, '[REDACTED_FINANCIAL_ID]', sanitized)

        guardrail_event(
            "pii_sanitization",
            passed=True,
            input_data={"text_length": len(text)},
            metadata={"redacted": sanitized != text},
        )
        
        return sanitized

    @staticmethod
    def detect_math_hallucination(invoice_json: dict) -> dict:
        """
        Output Guardrail: Verifies that the LLM did not hallucinate math.
        Checks if the sum of line item totals equals the header total_amount.
        """
        try:
            header_total = float(invoice_json.get("header", {}).get("total_amount", 0))
            line_items = invoice_json.get("line_items", [])
            
            calculated_total = sum(float(item.get("total", 0)) for item in line_items)
            
            # Allow a small tolerance for rounding differences (e.g., tax additions)
            difference = abs(header_total - calculated_total)
            
            # We inject a flag into the JSON so the downstream validation agent knows
            if difference > 1.0: # If it's off by more than $1, flag it
                invoice_json["rai_warning"] = f"Hallucination Alert: Header total ({header_total}) does not match line items ({calculated_total})"
                guardrail_event(
                    "math_hallucination",
                    passed=False,
                    input_data={"header_total": header_total, "calculated_total": calculated_total},
                    metadata={"difference": difference},
                )
            else:
                invoice_json["rai_warning"] = None
                guardrail_event(
                    "math_hallucination",
                    passed=True,
                    input_data={"header_total": header_total, "calculated_total": calculated_total},
                    metadata={"difference": difference},
                )
                
            return invoice_json
            
        except Exception as e:
            invoice_json["rai_warning"] = f"Failed hallucination check: {str(e)}"
            guardrail_event(
                "math_hallucination",
                passed=False,
                metadata={"error": str(e)},
            )
            return invoice_json

    @staticmethod
    def check_translation_confidence(confidence_score: float, minimum_threshold: float = 0.85) -> bool:
        # (Your existing code here)
        if confidence_score < minimum_threshold:
            print(f"RAI Alert: Translation confidence ({confidence_score}) is below threshold. Routing to Human-in-the-Loop.")
            guardrail_event(
                "translation_confidence",
                passed=False,
                input_data={"confidence_score": confidence_score},
                metadata={"minimum_threshold": minimum_threshold},
            )
            return False
        guardrail_event(
            "translation_confidence",
            passed=True,
            input_data={"confidence_score": confidence_score},
            metadata={"minimum_threshold": minimum_threshold},
        )
        return True
    
    @staticmethod
    def detect_prompt_injection(user_query: str) -> bool:
        """
        Security Guardrail (RAG): Detects malicious attempts by users to trick 
        the AI into ignoring instructions, dropping databases, or revealing system prompts.
        """
        # A lightweight list of common hacker prompt injection phrases
        injection_keywords = [
            "ignore previous instructions", 
            "system prompt", 
            "bypass", 
            "you are now", 
            "forget everything",
            "drop table"
        ]
        
        query_lower = user_query.lower()
        for word in injection_keywords:
            if word in query_lower:
                print(f"RAI Alert: Potential Prompt Injection detected: '{word}'")
                guardrail_event(
                    "prompt_injection",
                    passed=False,
                    input_data={"query_preview": user_query[:500]},
                    metadata={"matched_keyword": word},
                )
                return True # Injection detected!

        guardrail_event(
            "prompt_injection",
            passed=True,
            input_data={"query_preview": user_query[:500]},
        )
        return False # Query is safe

    @staticmethod
    def check_rag_groundedness(llm_answer: str, retrieved_context: str) -> bool:
        """
        Quality Guardrail (RAG Reflection): Ensures the LLM's answer is actually 
        based on the retrieved invoice document, and it didn't just invent an answer 
        from its pre-training data.
        """
        # A simple keyword overlap check. In production, you'd use an LLM for this!
        # If the answer mentions specific numbers or IDs, they MUST exist in the context.
        
        # Find all numbers/IDs in the answer
        import re
        answer_entities = set(re.findall(r'\b[A-Z0-9-]{4,}\b|\b\d+\.\d+\b', llm_answer))
        
        for entity in answer_entities:
            if entity not in retrieved_context:
                print(f"RAI Hallucination Alert: The AI mentioned '{entity}', but it isn't in the source document!")
                guardrail_event(
                    "rag_groundedness",
                    passed=False,
                    input_data={"answer_preview": llm_answer[:500]},
                    metadata={"missing_entity": entity},
                )
                return False # Not grounded

        guardrail_event(
            "rag_groundedness",
            passed=True,
            input_data={"answer_preview": llm_answer[:500]},
            metadata={"checked_entities": sorted(answer_entities)},
        )
        return True # Safely grounded in reality
    @staticmethod
    def validate_amount_fairness(invoice_price: float, po_price: float, tolerance: float) -> str:
        """
        Business Guardrail: Evaluates if a price difference between an invoice 
        and a Purchase Order is fair and within acceptable tolerance limits.
        """
        difference = abs(invoice_price - po_price)
        if difference <= tolerance:
            return "Pass: Price difference is within fair tolerance."
        
        # If it fails, provide a clear, responsible reason why
        return f"Fail: Price difference of ${difference:.2f} exceeds the allowed fairness tolerance of ${tolerance:.2f}."
