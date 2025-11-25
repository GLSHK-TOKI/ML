import base64
import json
import logging
from dataclasses import dataclass

from google.oauth2 import service_account

from ..exception import SDKException
from ..instance_config import BaseInstanceConfig

logger = logging.getLogger(__name__)

@dataclass
class VertexAIInstanceConfig(BaseInstanceConfig):
    location: str
    project: str
    credentials: str | None

    def init_auth(self):
        if self.credentials is None:
            logger.error("Google credentials value is missing")
            msg = "Google credentials value is missing"
            raise SDKException(400, msg)

        # Resolved by explicitly specifying the required OAuth scopes during authentication:
        # https://github.com/googleapis/python-genai/issues/2
        scopes = [
            "https://www.googleapis.com/auth/generative-language",
            "https://www.googleapis.com/auth/cloud-platform",
        ]
        decoded_string = self.credentials
        base = (base64.b64decode(decoded_string)).decode("utf-8")
        service_account_info = json.loads(base)
        return service_account.Credentials.from_service_account_info(service_account_info, scopes=scopes)
