import base64
import json
import logging
from dataclasses import dataclass

from google.oauth2 import service_account

from ..exception import SDKException
from ..instance_config import BaseInstanceConfig

logger = logging.getLogger(__name__)

@dataclass
class LlamaInstanceConfig(BaseInstanceConfig):
    location: str
    project: str
    credentials: str | None

    def init_auth(self):
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        if self.credentials is None:
            logger.error("Google credentials value is missing")
            msg = "Google credentials value is missing"
            raise SDKException(400, msg)
        decoded_string = self.credentials
        decoded_credentials = (base64.b64decode(decoded_string)).decode("utf-8")
        service_account_info = json.loads(decoded_credentials)
        return service_account.Credentials.from_service_account_info(service_account_info, scopes=scopes)
