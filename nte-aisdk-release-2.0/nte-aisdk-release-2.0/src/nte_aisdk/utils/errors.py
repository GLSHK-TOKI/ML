import inspect
import logging
from functools import wraps

from elasticsearch import ApiError, TransportError

from nte_aisdk.errors import APIError, InvalidArgumentError

logger = logging.getLogger(__name__)

def ensure_arguments(func):
    """Decorator to check for missing required arguments."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_args = signature.bind_partial(
            *args, **kwargs
        )  # bound_args.arguments is a dict of arguments passed to the function

        for name, param in signature.parameters.items():
            if param.default is param.empty and name not in bound_args.arguments:
                msg = f"Missing required argument: '{name}'"
                raise InvalidArgumentError(msg)

        return func(*args, **kwargs)
    return wrapper

def handle_elasticsearch_error(operation, *args, **kwargs):
    """Utility function to handle Elasticsearch operations and exceptions."""
    try:
        return operation(*args, **kwargs)
    except ApiError as e:
        msg = f"API error during an Elasticsearch operation. {e!s}"
        logger.exception(msg)
        raise APIError(msg, e.status_code, e.body) from e
    except TransportError as e:
        msg = f"Transport error occurred during an Elasticsearch operation. {e!s}"
        logger.exception(msg)
        raise APIError(msg) from e
    except Exception as e:
        msg = "An unexpected error occurred during an Elasticsearch operation."
        logger.exception(msg)
        raise APIError(msg) from e