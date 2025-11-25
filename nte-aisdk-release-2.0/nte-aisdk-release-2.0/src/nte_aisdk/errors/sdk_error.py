"""Base SDK error classes."""


class SDKError(Exception):
    """Base class for all SDK errors."""

    message: str

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
