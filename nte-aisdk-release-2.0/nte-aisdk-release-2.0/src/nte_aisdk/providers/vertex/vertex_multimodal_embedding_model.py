from __future__ import annotations

import base64
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel

from nte_aisdk import errors, types
from nte_aisdk.multimodal_embedding_model import MultimodalEmbeddingModel

if TYPE_CHECKING:
    from nte_aisdk.providers.vertex import VertexProvider

logger = logging.getLogger(__name__)


class VertexMultimodalEmbeddingModel(MultimodalEmbeddingModel):
    """Vertex AI Multimodal Embedding Model implementation."""

    DEFAULT_DIMENSION = 1408

    def __init__(self, model_name: str, provider: VertexProvider):
        """Initialize the Vertex multimodal embedding model.

        Args:
            model_name: The name of the embedding model, e.g. "multimodalembedding@001".
            provider: The Vertex provider that manages the connection to Vertex AI services.
        """
        super().__init__(model_name, provider)

        # Initialize Vertex AI with the provider's credentials
        vertexai.init(
            project=self.provider.project,
            location=self.provider.location,
            credentials=self.provider.credentials
        )

        # Create the multimodal embedding model
        embedding_model = MultiModalEmbeddingModel.from_pretrained(model_name)
        self.models = [embedding_model]

    def do_embed_multimodal(
        self,
        content: types.TextPart | types.FilePart,
        **kwargs: Any,
    ) -> types.EmbedResponse:
        """Generate multimodal embeddings for mixed content (text + images).

        Args:
            content: List of content parts (TextPart or FilePart).
            **kwargs: Additional parameters for embedding generation.

        Returns:
            Embedding response containing vector and metadata.
        """
        if not content:
            msg = "No content provided for embedding generation."
            raise errors.InvalidArgumentError(msg)

        dimension = kwargs.pop("dimension", self.DEFAULT_DIMENSION)

        # Separate text and image parts
        text_parts = []
        image_parts = []

        for part in content:
            if isinstance(part, types.TextPart):
                text_parts.append(part.text)
            elif isinstance(part, types.FilePart):
                logger.info("part.file.mime_type: %s, type: %s", part.file.mime_type, type(part.file.mime_type))
                if part.file.mime_type.startswith("image/"):
                    image_parts.append(part.file.bytes)
            else:
                msg = f"Unsupported content part type: {type(part).__name__}"
                raise errors.InvalidArgumentError(msg)

        # Process text parts
        contextual_text = " ".join(text_parts).strip()
        total_tokens = sum(len(text_value.split()) for text_value in text_parts)

        # Process image parts (use the last image if multiple provided)
        image: Image | None = None
        if image_parts:
            # Get base64 string from the last image part
            image_base64 = image_parts[-1]
            if not image_base64:
                msg = "Image data is required for image content parts."
                raise errors.InvalidArgumentError(msg)

            # Decode base64 to bytes
            try:
                decoded_bytes = base64.b64decode(image_base64)
            except Exception as exc:
                msg = "Failed to decode base64 image data."
                raise errors.InvalidArgumentError(msg) from exc

            # Create Vertex AI Image object
            image = Image(image_bytes=decoded_bytes)
            total_tokens += len(image_parts)

        if not contextual_text and image is None:
            msg = "No valid content provided for embedding generation."
            raise errors.InvalidArgumentError(msg)

        # Prepare embedding parameters
        embedding_kwargs: dict[str, Any] = {"dimension": dimension, **kwargs}
        if image and contextual_text:
            embedding_kwargs.update({"image": image, "contextual_text": contextual_text})
        elif image:
            embedding_kwargs["image"] = image
        elif contextual_text:
            embedding_kwargs["contextual_text"] = contextual_text

        # Get embeddings
        embeddings = self.model.get_embeddings(**embedding_kwargs)

        # Extract embedding (prefer image for multimodal content)
        if image and hasattr(embeddings, "image_embedding") and embeddings.image_embedding:
            embedding_values = list(embeddings.image_embedding)
        elif contextual_text and hasattr(embeddings, "text_embedding") and embeddings.text_embedding:
            embedding_values = list(embeddings.text_embedding)
        else:
            msg = "No embedding returned from Vertex AI multimodal model."
            raise errors.APIError(msg)

        return types.EmbedResponse(
            embedding=embedding_values,
            metadata=types.EmbedResponseMetadata(
                model_id=self.model_id,
                usage=types.EmbedResponseUsage(tokens=total_tokens)
            )
        )

    @staticmethod
    def _load_image_from_base64(image_data: str) -> Image:
        """Convert a base64-encoded image string into a Vertex Image object."""
        try:
            img_bytes = base64.b64decode(image_data)
        except (TypeError, ValueError) as exc:
            msg = "Invalid base64 image data provided."
            raise errors.InvalidArgumentError(msg) from exc

        with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as tmp_file:
            tmp_file.write(img_bytes)
            tmp_path = tmp_file.name

        try:
            return Image.load_from_file(tmp_path)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to remove temporary image file at %s", tmp_path, exc_info=True)