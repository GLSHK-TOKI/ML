"""API-related error classes."""

from typing import Any

from .sdk_error import SDKError


class APIError(SDKError):
    status_code: int | None
    response: Any | None

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response: Any | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response
