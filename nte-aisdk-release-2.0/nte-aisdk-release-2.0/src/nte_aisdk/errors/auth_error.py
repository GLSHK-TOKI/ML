"""Authentication error classes."""

from .sdk_error import SDKError


class AuthError(SDKError):
    """Raised when there is an authentication-related error."""