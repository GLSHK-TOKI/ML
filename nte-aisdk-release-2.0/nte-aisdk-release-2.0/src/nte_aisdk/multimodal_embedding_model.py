from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

from nte_aisdk import types
from nte_aisdk.utils import ensure_arguments

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

class MultimodalEmbeddingModel:
    """Base class for multimodal embedding models that can handle text and images.
    This class is independent from TextEmbeddingModel and specifically designed
    to handle multimodal content including text, images, and mixed content.
    """
    model_id: str
    # List of single or multiple model instances used for load balancing.
    models: list

    def __init__(self, model_id: str, provider):
        self.model_id = model_id
        self.provider = provider
        self.models: list[Any] = []

    @property
    def model(self):
        if not self.models:
            msg = "No provider models have been registered for this multimodal embedding model."
            raise RuntimeError(msg)
        return random.choice(self.models)

    @ensure_arguments
    def do_embed_multimodal(
        self,
        content: types.TextPart | types.FilePart
    ) -> types.EmbedResponse:
        """Make a request to generate multimodal embedding from text and/or image content.
        """
        msg = "This method should be overridden by subclasses by different providers."
        raise NotImplementedError(msg)
