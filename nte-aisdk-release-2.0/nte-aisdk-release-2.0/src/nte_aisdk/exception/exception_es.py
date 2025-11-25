import logging

from elasticsearch import ApiError, TransportError

from .exception_util import SDKException

logger = logging.getLogger(__name__)

class ElasticsearchError(SDKException):
    """Base exception for Elasticsearch-related errors."""

class ElasticsearchTransportError(ElasticsearchError):
    """Raised when there is a transport error in Elasticsearch."""

class ElasticsearchApiError(ElasticsearchError):
    """Raised when there is an API error in Elasticsearch."""

def handle_elasticsearch_operation(operation, *args, **kwargs):
    """Utility function to handle Elasticsearch operations and exceptions."""
    try:
        return operation(*args, **kwargs)
    except TransportError as e:
        msg = f"Transport error occurred during an Elasticsearch operation. {e!s}"
        logger.exception(msg)
        raise ElasticsearchTransportError(400, msg) from e
    except ApiError as e:
        msg = f"API error during an Elasticsearch operation. {e!s}"
        logger.exception(msg)
        raise ElasticsearchApiError(e.status_code, msg) from e
    except Exception as e:
        msg = "An unexpected error occurred during an Elasticsearch operation."
        logger.exception(msg)
        raise SDKException(500, msg) from e