import inspect
import logging
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)


class SDKException(Exception):
    """Base class for all SDK exceptions."""
    status_code: int
    message: str

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message

class MissingArgumentError(SDKException):
    """Raised when a required argument is missing."""

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
                raise MissingArgumentError(400, msg)

        return func(*args, **kwargs)
    return wrapper