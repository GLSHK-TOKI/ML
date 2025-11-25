import contextlib
from nte_aisdk import types

# Check if Flask is available
try:
    from flask import g
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    g = None # type: ignore[assignment] # Fallback for non-Flask environments

class Feature:
    """Base class for features."""

    class SingleFunctionTokenCounter:
        """Helper class to count tokens for a specific function call.
        For tracking tokens in a function with multiple LLM calls, not for the entire request lifecycle.
        """
        def __init__(self):
            self.prompt_tokens = 0
            self.completion_tokens = 0

        def add_prompt_tokens(self, count: int):
            self.prompt_tokens += count

        def add_completion_tokens(self, count: int):
            self.completion_tokens += count

        def to_usage(self):
            return types.LanguageModelResponseUsage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens
            )

    def _add_tokens_to_context(self, token_type: str, token_count: int):
        """Update token count for the specified type, supporting Flask or non-Flask environments.

        Args:
            token_type (str): Type of token ('prompt' or 'completion').
            token_count (int): Number of tokens to add.
        """
        attr_name = f"nte_aisdk_{token_type}_tokens"
        if FLASK_AVAILABLE and g is not None:
            # Flask environment: Use Flask's g
            with contextlib.suppress(RuntimeError): # Suppress RuntimeError if g is accessed outside a Flask app context
                setattr(g, attr_name, getattr(g, attr_name, 0) + token_count)
        else:
            # Non-Flask environment
            pass

    def _add_prompt_tokens_to_context(self, prompt_tokens: int):
        """Add prompt tokens (wrapper for add_tokens)."""
        self._add_tokens_to_context("prompt", prompt_tokens)

    def _add_completion_tokens_to_context(self, completion_tokens: int):
        """Add completion tokens (wrapper for add_tokens)."""
        self._add_tokens_to_context("completion", completion_tokens)

    def _get_user(self):
        """Get the user from Flask's g or return None if not available."""
        if FLASK_AVAILABLE and g is not None:
            with contextlib.suppress(RuntimeError): # Suppress RuntimeError if g is accessed outside a Flask app context
                return getattr(g, "nte_aisdk_user", None)
        return None