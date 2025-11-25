import inspect
import logging
from functools import wraps

from .exception_util import SDKException

logger = logging.getLogger(__name__)

class ExampleMissingInputFieldError(SDKException):
    """Raised when the example is missing a field that is provided in the input field."""

class ExampleInvalidFieldError(SDKException):
    """Raised when the example has extra field that is not provided in the input field."""


def check_example_fields(func):
    """Decorator to check if the input fields provided are present in the example(s)."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_args = signature.bind_partial(
            *args, **kwargs
        )  # bound_args.arguments is a dict of arguments passed to the function

        if "examples" in bound_args.arguments:
            for example in bound_args.arguments["examples"]:
                throw_if_example_missing_field(example, bound_args.arguments["input_fields"])
                throw_if_example_invalid_field(example)

        if "example" in bound_args.arguments:
            throw_if_example_missing_field(bound_args.arguments["example"], bound_args.arguments["input_fields"])
            throw_if_example_invalid_field(bound_args.arguments["example"])

        return func(*args, **kwargs)
    return wrapper

def throw_if_example_missing_field(example, input_fields):
    for field in input_fields:
        if field not in example:
            logger.error("Example %s missing required field: %s", example , field)
            msg = f"Example {example} missing required field: '{field}'"
            raise ExampleMissingInputFieldError(400, msg)

def throw_if_example_invalid_field(example):
    invalid_fields = ["_id", "last_updated_time"]
    delimiter = ", "
    fields = [field for field in example if field in invalid_fields]
    if len(fields) > 0:
        field_str = delimiter.join(fields)
        msg = f"Example should not include the following field(s): {field_str}"
        logger.error(msg)
        raise ExampleInvalidFieldError(400, msg)