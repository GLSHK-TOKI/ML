"""Parse error classes."""

from .sdk_error import SDKError


class ParseError(SDKError):
    """Raised when there is an error parsing data, such as JSON parsing."""

    text: str # The original text that caused the parse error

    def __init__(self, message: str, text: str):
        super().__init__(message)
        self.text = text