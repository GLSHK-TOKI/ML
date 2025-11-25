import logging

from .exception_util import SDKException

logger = logging.getLogger(__name__)

class RateLimitingError(SDKException):
    """Exception for rate-limiting-related errors."""