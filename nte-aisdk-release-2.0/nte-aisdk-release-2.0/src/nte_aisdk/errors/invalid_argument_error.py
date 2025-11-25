"""Invalid argument error classes."""

from .sdk_error import SDKError


class InvalidArgumentError(SDKError, ValueError):
    """Raised when an invalid argument is provided to a function or method."""
