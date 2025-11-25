import logging

from google.genai import errors
from openai._exceptions import APIConnectionError, APIResponseValidationError, APIStatusError

from .exception_util import SDKException

# Configure logging
logger = logging.getLogger(__name__)

"""
Exceptions Base Class for Azure Open AI and VertexAI models
"""
class LlmModelError(SDKException):
    """Base exception for AzureOpenAI-related errors."""

"""
Azure Open AI Model Exceptions
"""
class AzureAPIResponseValidationError(LlmModelError):
    """Raised when the response validation failed."""

class AzureAPIStatusError(LlmModelError):
    """Raised when an API response has a status code of 4xx or 5xx."""

class AzureAPIConnectionError(LlmModelError):
    """Raised when the connection error to Azure."""


"""
Google Vertex AI Model Exceptions
"""

class GoogleAPIError(LlmModelError):
    """Base exception for Google GenAI API errors."""

class GoogleClientError(LlmModelError):
    """Raised when all client error (HTTP 4xx) responses."""

class GoogleServerError(LlmModelError):
    """Raised when all server error (HTTP 5xx) responses."""

def handle_llm_model_operation(operation, *args, **kwargs):
    """Utility function to handle Azure operations and exceptions."""
    try:
        return operation(*args, **kwargs)
    except APIResponseValidationError as e:
        msg = f"Response validation error when calling Azure OpenAI API. {e.message}"
        logger.exception(msg)
        raise AzureAPIResponseValidationError(e.status_code, msg) from e
    except APIConnectionError as e:
        msg = f"Connection error when calling Azure OpenAI API. {e.message}"
        logger.exception(msg)
        raise AzureAPIConnectionError(500, msg) from e
    except APIStatusError as e:
        msg = f"API status error when calling Azure OpenAI API. {e.message}"
        logger.exception(msg)
        raise AzureAPIStatusError(e.status_code, msg) from e
    except errors.APIError as e:
        msg = f"API error when calling GenAI API. {e.message}"
        logger.exception(msg)
        raise GoogleAPIError(e.status, msg) from e
    except errors.ClientError as e:
        msg = f"Client error when calling GenAI API. {e.message}"
        logger.exception(msg)
        raise GoogleClientError(400, msg) from e
    except errors.ServerError as e:
        msg = f"Server error when calling GenAI API. {e.message}"
        logger.exception(msg)
        raise SDKException(500, msg) from e
    except Exception as e:
        msg = "Unexpected error occurred when calling LLM API."
        logger.exception(msg)
        raise SDKException(500, msg) from e