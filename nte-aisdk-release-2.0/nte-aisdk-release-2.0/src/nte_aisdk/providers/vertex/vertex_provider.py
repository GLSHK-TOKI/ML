import base64
import json

from google.oauth2 import service_account

from nte_aisdk.exception import ensure_arguments
from nte_aisdk.providers.vertex import VertexLanguageModel
from nte_aisdk.providers.vertex.vertex_multimodal_embedding_model import VertexMultimodalEmbeddingModel
from nte_aisdk.utils import errors


class VertexProvider:
    """VertexProvider offers methods for creating models to interact with model on Vertex AI (Google Cloud Platform)."""
    location: str
    project: str
    credentials_base64: str

    def __init__(
        self,
        location: str,
        project: str,
        credentials_base64: str
    ):
        """Initializes the Vertex provider

        Args:
            location (str): The location for the Vertex AI service, e.g., 'us-central1'.
            project (str): The Google Cloud project ID.
            credentials_base64 (str | None): Optional base64-encoded service account key.
        """
        self.location = location
        self.project = project
        self.credentials_base64 = credentials_base64
        self.credentials = self._create_credentials_from_base64()

    @ensure_arguments
    def create_language_model(
        self,
        model_name: str,
    ):
        """Creates an Vertex language model

        Args:
            model_name (str): The name of the model, e.g. "gpt-4o", "DeepSeek-R1". You can find the model name of your deployment in the Vertex AI Foundry.
        """
        return VertexLanguageModel(model_name, self)

    @ensure_arguments
    def create_embedding_model(
        self,
        model_name: str,
    ):
        return VertexMultimodalEmbeddingModel(model_name, self)

    def _create_credentials_from_base64(self):
        if self.credentials_base64 is None:
            msg = "Missing base64-encoded Vertex AI service account key for authentication."
            raise errors.InvalidArgumentError(msg)

        # Resolved by explicitly specifying the required OAuth scopes during authentication:
        # https://github.com/googleapis/python-genai/issues/2
        scopes = [
            "https://www.googleapis.com/auth/generative-language",
            "https://www.googleapis.com/auth/cloud-platform",
        ]
        decoded_string = self.credentials_base64
        base = (base64.b64decode(decoded_string)).decode("utf-8")
        service_account_info = json.loads(base)
        return service_account.Credentials.from_service_account_info(service_account_info, scopes=scopes)