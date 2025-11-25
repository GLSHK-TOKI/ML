import logging

from .exception_util import SDKException

logger = logging.getLogger(__name__)

class AuthError(SDKException):
    """Exception for auth-related errors."""