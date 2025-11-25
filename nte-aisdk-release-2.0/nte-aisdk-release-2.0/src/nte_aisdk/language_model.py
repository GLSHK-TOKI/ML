import base64
import logging
import random
from collections.abc import Iterator
from typing import Any, ClassVar

from nte_aisdk import types
from nte_aisdk.errors import InvalidArgumentError
from nte_aisdk.utils import ensure_arguments

logger = logging.getLogger(__name__)

class LanguageModel:
    """Base class for language models."""
    model_id: str
    # List of single or multiple model instances used for load balancing.
    models: list
    openai_models: list

    # Constants for file validation
    _MAX_FILE_SIZE_BYTES: ClassVar[int] = 20 * 1024 * 1024  # 20MB
    _SUPPORTED_IMAGE_FORMATS: ClassVar[set[str]] = {"image/png", "image/jpeg"}
    _NON_VISION_MODELS: ClassVar[set[str]] = {"deepseek-r1"}

    def __init__(self, model_id: str, provider):
        self.model_id = model_id
        self.provider = provider
        self.models = []

    @property
    def model(self):
        return random.choice(self.models)

    @property
    def openai_model(self):
        return random.choice(self.openai_models)

    @ensure_arguments
    def do_generate(
        self,
        messages: list[types.MessageOrDict],
        response_type: types.ResponseType = types.ResponseType.TEXT,
        response_schema: dict[Any, Any] | None = None,
        instructions: str | None = None,
    ) -> types.GenerateResponse:
        """Make a request to generate content from the language model.

        Args:
            messages (types.MessageOrDict): The input messages to the language model.
            response_type (types.ResponseType): The type/format of response expected from the language model.
            response_schema (dict[Any, Any] | None): Optional schema to validate the response structure. Need to use along with types.ResponseType.JSON.
            instructions (str | None): Optional system/developer instructions for the language model to follow during generation.

        Returns:
            types.GenerateResponse: The response from the language model containing the generated message and metadata.
        """
        msg = "This method should be overridden by subclasses by different providers."
        raise NotImplementedError(msg)

    def do_stream(
        self,
        messages: list[types.MessageOrDict],
        response_type: types.ResponseType = types.ResponseType.TEXT,
        response_schema: dict[Any, Any] | None = None,
        instructions: str | None = None,
    ) -> Iterator[types.StreamResponse]:
        """Stream the content from the language model.

        Args:
            messages (list[types.MessageOrDict]): The input messages to the language model.
            response_type (types.ResponseType): The type/format of response expected from the language model.
            response_schema (dict[Any, Any] | None): Optional schema to validate the response structure. Need to use along with types.ResponseType.JSON.
            instructions (str | None): Optional system/developer instructions for the language model to follow during generation.

        Yields:
            Iterator[types.StreamResponse]: An iterator that yields parts of the response from the language model as they are generated.
        """
        msg = "This method should be overridden by subclasses by different providers."
        raise NotImplementedError(msg)

    def _validate_file_part_for_model(self, file_part: types.FilePart, model_name: str) -> None:
        """Validate if a FilePart is compatible with the specified model.

        This function handles all validation internally:
        - Only PNG and JPEG formats
        - Maximum 20MB file size
        - File existence checks
        - MIME type detection
        - Model compatibility checks

        Args:
            file_part: The FilePart to validate
            model_name: Name of the model to check compatibility against

        Raises:
            InvalidArgumentError: If FilePart is not compatible with the model
        """
        # Check if model supports images
        if (not self._model_supports_images(model_name) and
                self._is_image_mime_type(file_part.file.mime_type)):
            msg = (
                f"Model '{model_name}' does not support image inputs. "
                "Please use a capable model."
            )
            raise InvalidArgumentError(msg)

        # Additional format validation for vision models
        if (self._model_supports_images(model_name) and
                self._is_image_mime_type(file_part.file.mime_type) and
                not self._is_supported_image_format(file_part.file.mime_type)):
            msg = (
                f"Model '{model_name}' only supports PNG and JPEG images. "
                f"Received: {file_part.file.mime_type}"
            )
            raise InvalidArgumentError(msg)

        # Validate file size for images (internal validation)
        if self._is_image_mime_type(file_part.file.mime_type):
            # Decode base64 to check actual file size
            try:
                decoded_bytes = base64.b64decode(file_part.file.bytes)
            except Exception as e:
                msg = f"Failed to validate file size: {e}"
                raise InvalidArgumentError(msg) from e

            file_size = len(decoded_bytes)
            if file_size > self._MAX_FILE_SIZE_BYTES:
                size_mb = file_size / (1024 * 1024)
                msg = f"File size {size_mb:.2f}MB exceeds maximum allowed size of 20MB"
                raise InvalidArgumentError(msg)

    def _is_supported_image_format(self, mime_type: str) -> bool:
        """Private method to check if image format is supported for vision models.

        Args:
            mime_type: MIME type string

        Returns:
            True if it's a supported image format (PNG or JPEG)
        """
        return mime_type in self._SUPPORTED_IMAGE_FORMATS

    def _model_supports_images(self, model_name: str) -> bool:
        """Private method to check if a model supports image inputs.

        Args:
            model_name: Name of the model

        Returns:
            True if the model supports images
        """
        # Convert to lowercase for case-insensitive comparison
        model_lower = model_name.lower()

        # Return False if any non-vision model matches, True otherwise
        return not any(non_vision_model in model_lower for non_vision_model in self._NON_VISION_MODELS)

    def _is_image_mime_type(self, mime_type: str) -> bool:
        """Private method to check if MIME type represents an image.

        Args:
            mime_type: MIME type string

        Returns:
            True if it's an image MIME type
        """
        return mime_type.startswith("image/")
