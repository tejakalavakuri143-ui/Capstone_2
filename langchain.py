"""
Shim for langfuse.langchain used by the orchestrator.
Provides a minimal CallbackHandler to avoid import-time failures.
"""
from .decorators import CallbackHandler

__all__ = ["CallbackHandler"]
