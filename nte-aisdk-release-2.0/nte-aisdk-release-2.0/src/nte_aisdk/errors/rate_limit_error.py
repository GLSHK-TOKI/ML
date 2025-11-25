"""Rate limit error classes."""

from .sdk_error import SDKError


class RateLimitError(SDKError):
    """Raised when the API rate limit is exceeded."""