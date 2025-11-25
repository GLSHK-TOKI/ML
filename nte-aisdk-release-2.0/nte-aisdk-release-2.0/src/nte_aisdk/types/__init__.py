from enum import Enum
from typing import Any, Literal, NotRequired

from pydantic import Field
from typing_extensions import TypedDict

from ._azure import (
    AzureInstanceConfig,
    AzureInstanceConfigDict,
    AzureInstanceConfigOrDict,
    AzureInstanceConfigWithoutAPIVersion,
    AzureInstanceConfigWithoutAPIVersionDict,
    AzureInstanceConfigWithoutAPIVersionOrDict,
    AzureModelConfigWithAPIVersion,
)
from ._common import BaseModel
from ._pii_detection import (
    PIICategory,
    PIICategoryDict,
    PIICategoryOrDict,
    PIIDetectionConfig,
    PIIDetectionConfigDict,
    PIIDetectionConfigOrDict,
    PIIItem,
)


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class ResponseType(str, Enum):
    TEXT = "text"
    JSON = "json"

class TextPart(BaseModel):
    """A part of a message that contains text content."""

    kind: str = Field(
        default="text",
        description="The type of content part. Always 'text'."
    )
    text: str = Field(
        description="The text content."
    )

class TextPartDict(TypedDict):
    """A part of a message that contains text content."""

    kind: Literal["text"]
    """The type of content part. Always 'text'."""

    text: str
    """The text content."""


class FileWithBytes(BaseModel):
    """File information containing name, mime_type and bytes."""

    name: str = Field(
        description="The file name."
    )
    mime_type: str = Field(
        description="The MIME type of the file."
    )
    bytes: str = Field(
        description="The file content as base64 encoded string."
    )

class FileWithBytesDict(TypedDict):
    name: str
    mime_type: str
    bytes: str

class FilePart(BaseModel):
    """A part of a message that contains file content."""

    kind: str = Field(
        default="file",
        description="The type of content part. Always 'file'."
    )
    file: FileWithBytes = Field(
        description="The file information and the bytes content."
    )

class FilePartDict(TypedDict):
    kind: Literal["file"]
    file: FileWithBytesDict


class DataPart(BaseModel):
    """A part of a message that contains data content."""

    kind: str = Field(
        default="data",
        description="The type of content part. Always 'data'."
    )
    data: dict = Field(
        description="The data content as a structured JSON data in dictionary."
    )


class DataPartDict(TypedDict):
    kind: Literal["data"]
    data: dict

class ReasoningPart(BaseModel):
    """A part of a message that contains reasoning content."""

    kind: str = Field(
        default="reasoning",
        description="The type of content part. Always 'reasoning'."
    )
    reasoning: str = Field(
        description="The reasoning content."
    )

class ReasoningPartDict(TypedDict):
    kind: Literal["reasoning"]
    reasoning: str

Part = TextPart | FilePart | DataPart | ReasoningPart
PartDict = TextPartDict | FilePartDict | DataPartDict | ReasoningPartDict
PartOrDict = Part | PartDict

class Message(BaseModel):
    """Base class for messages for language models."""

    role: Role = Field(
        description="Message sender's role. e.g. 'user', 'assistant'."
    )
    parts: list[Part] = Field(
        description="""A list of content parts. Represents a distinct piece of content within a Message.
            Each part is a union type representing different content types.
        """
    )
    message_id: str | None = Field(
        default=None,
        description="A unique identifier for the message."
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata associated with the message."
    )

    @property
    def text(self) -> str | None:
        """Returns the text of the first TextPart found in parts, else None."""
        for part in self.parts:
            if isinstance(part, TextPart):
                return part.text
        return None

    @property
    def data(self) -> dict | None:
        """Returns the data of the first DataPart found in parts, else None."""
        for part in self.parts:
            if isinstance(part, DataPart):
                return part.data
        return None

    @property
    def reasoning(self) -> str | None:
        """Returns the reasoning of the first ReasoningPart found in parts, else None."""
        for part in self.parts:
            if isinstance(part, ReasoningPart):
                return part.reasoning
        return None

class MessageDict(TypedDict):
    role: Role
    parts: list[PartDict]
    message_id: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]

MessageOrDict = Message | MessageDict

class LanguageModelResponseUsage(BaseModel):
    """Usage statistics for a generation request."""

    prompt_tokens: int = Field(
        description="Number of tokens in the input prompt."
    )
    completion_tokens: int = Field(
        description="Number of tokens in the generated completion."
    )

class GenerateResponseMetadata(BaseModel):
    """Metadata for a generation response."""

    model_id: str = Field(
        description="The id of the model used for generation."
    )
    usage: LanguageModelResponseUsage = Field(
        description="The token usage for the generation. Includes prompt and completion tokens."
    )
    confidence: float | None = Field(
        description="The confidence score of the generation dereived from the logprobs, if available."
    )
    finish_reason: str | None = Field(
        default=None,
        description="The reason why the generation finished"
    )

class GenerateResponse(BaseModel):
    """Response from a language model generation request."""

    message: Message = Field(
        description="The generated message from the language model."
    )
    metadata: GenerateResponseMetadata = Field(
        description="Metadata associated with the generation response."
    )

class StreamResponse(BaseModel):
    """A chunk of streamed response from a language model streaming request.
    Structure is closely aligned with TaskArtifactUpdateEvent in A2A protocol. But not including the concept of task and artifact.
    """

    message: Message = Field(
        description="The message chunk from the language model."
    )
    append: bool = Field(
        default=False,
        description="If true, the message is an append to the previous message chunk."
    )
    last_chunk: bool = Field(
        default=False,
        description="If true, this is the last chunk of the stream."
    )
    metadata: GenerateResponseMetadata | None = Field(
        default=None,
        description="Only available for the last chunk. Metadata associated with the stream response."
    )

class EmbedResponseUsage(BaseModel):
    """Usage statistics for an embedding request."""

    tokens: int = Field(
        description="Number of tokens in the input text for embedding."
    )

class EmbedResponseMetadata(BaseModel):
    """Metadata for an embedding response."""

    model_id: str = Field(
        description="The id of the model used for embedding."
    )
    usage: EmbedResponseUsage = Field(
        description="The token usage for the embedding. Includes prompt and completion tokens."
    )

class EmbedResponse(BaseModel):
    """Response from a embedding model create embedding request."""

    embedding: list[float] = Field(
        description="The generated embedding vector."
    )
    metadata: EmbedResponseMetadata = Field(
        description="Metadata associated with the generation response."
    )

class PIIDetectionResponse(BaseModel):
    """Response from a PII detection request."""

    data: list[PIIItem] = Field(
        description="List of detected PII items"
    )
    usage: LanguageModelResponseUsage = Field(
        description="Token usage statistics for the PII detection"
    )
    confidence: float | None = Field(
        default=None,
        description="The confidence score of the detection if available"
    )

class PIIMaskingResponse(BaseModel):
    """Response from a PII masking request."""

    text: str = Field(
        description="The text with PII items masked"
    )
    usage: LanguageModelResponseUsage = Field(
        description="Token usage statistics for the PII masking"
    )
    confidence: float | None = Field(
        default=None,
        description="The confidence score of the masking if available"
    )

class MarkdownResponse(BaseModel):
    """Response from a markdown PII detection request."""

    data: str = Field(
        description="The markdown table containing detected PII"
    )
    usage: LanguageModelResponseUsage = Field(
        description="Token usage statistics for the markdown generation"
    )
    confidence: float | None = Field(
        default=None,
        description="The confidence score if available"
    )

class BatchMarkdownResult(BaseModel):
    """Result for a single text in a batch markdown generation request."""

    index: int = Field(
        description="The index of the text in the original batch"
    )
    markdown_response: MarkdownResponse = Field(
        description="The markdown PII detection response for this text"
    )

class BatchMarkdownResponse(BaseModel):
    """Response from a batch markdown PII detection request."""

    status: str = Field(
        description="The status of the batch processing ('completed', 'failed', etc.)"
    )
    results: list[BatchMarkdownResult] = Field(
        description="List of markdown detection results for each text in the batch"
    )

class BatchPIIDetectionResult(BaseModel):
    """Result for a single text in a batch PII detection request."""

    index: int = Field(
        description="The index of the text in the original batch"
    )
    detection_response: PIIDetectionResponse = Field(
        description="The PII detection response for this text"
    )

class BatchPIIDetectionResponse(BaseModel):
    """Response from a batch PII detection request."""

    status: str = Field(
        description="The status of the batch processing ('completed', 'failed', etc.)"
    )
    results: list[BatchPIIDetectionResult] = Field(
        description="List of PII detection results for each text in the batch"
    )

class BatchPIIMaskResult(BaseModel):
    """Result for a single text in a batch PII masking request."""

    index: int = Field(
        description="The index of the text in the original batch"
    )
    mask_response: PIIMaskingResponse = Field(
        description="The PII masking response for this text"
    )

class BatchPIIMaskResponse(BaseModel):
    """Response from a batch PII masking request."""

    status: str = Field(
        description="The status of the batch processing ('completed', 'failed', etc.)"
    )
    results: list[BatchPIIMaskResult] = Field(
        description="List of PII masking results for each text in the batch"
    )

class DynamicFewShotResponse(BaseModel):
    """Response from a dynamic few shot generation request."""

    text: str | None = Field(
        default=None,
        description="The generated text response (for both text and JSON modes)"
    )
    examples: list[dict[str, Any]] = Field(
        description="The examples that were used for this generation"
    )
    metadata: GenerateResponseMetadata = Field(
        description="Metadata associated with the generation response."
    )


__all__ = [
    "AzureInstanceConfig",
    "AzureInstanceConfigDict",
    "AzureInstanceConfigOrDict",
    "AzureInstanceConfigWithoutAPIVersion",
    "AzureInstanceConfigWithoutAPIVersionDict",
    "AzureInstanceConfigWithoutAPIVersionOrDict",
    "AzureModelConfigWithAPIVersion",
    "DataPart",
    "DataPartDict",
    "DynamicFewShotResponse",
    "EmbedResponse",
    "FilePart",
    "FilePartDict",
    "FileWithBytes",
    "FileWithBytesDict",
    "GenerateResponse",
    "GenerateResponseMetadata",
    "LanguageModelResponseUsage",
    "Message",
    "MessageDict",
    "MessageOrDict",
    "PIICategory",
    "PIICategoryDict",
    "PIICategoryOrDict",
    "PIIDetectionConfig",
    "PIIDetectionConfigDict",
    "PIIDetectionConfigOrDict",
    "PIIDetectionResponse",
    "PIIItem",
    "PIIMaskingResponse",
    "BatchPIIDetectionResult",
    "BatchPIIDetectionResponse",
    "BatchPIIMaskResult", 
    "BatchPIIMaskResponse",
    "Part",
    "PartDict",
    "PartOrDict",
    "ReasoningPart",
    "ReasoningPartDict",
    "ResponseType",
    "Role",
    "StreamResponse",
    "TextPart",
    "TextPartDict"
]
