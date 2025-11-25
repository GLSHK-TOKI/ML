import logging
import random

from nte_aisdk import types
from nte_aisdk.utils import ensure_arguments

logger = logging.getLogger(__name__)

class TextEmbeddingModel:
    """Base class for embedding models."""
    model_id: str
    # List of single or multiple model instances used for load balancing.
    models: list

    def __init__(self, model_id: str, provider):
        self.model_id = model_id
        self.provider = provider
        self.models = []

    @property
    def model(self):
        return random.choice(self.models)

    @ensure_arguments
    def do_embed(
        self,
        text: str
    ) -> types.EmbedResponse:
        """Make a request to generate content from the language model.

        Args:
            text (str): The input text to the embedding model.

        Returns:
            types.GenerateResponse: The response from the language model containing the generated message and metadata.
        """
        msg = "This method should be overridden by subclasses by different providers."
        raise NotImplementedError(msg)
