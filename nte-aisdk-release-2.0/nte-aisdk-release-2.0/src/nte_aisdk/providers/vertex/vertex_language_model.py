from __future__ import annotations

import base64
import copy
import json
import logging
import math
import re
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from nte_aisdk import errors, types
from nte_aisdk.language_model import LanguageModel
from nte_aisdk.utils.types import normalize_messages

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nte_aisdk.providers.vertex import VertexProvider

logger = logging.getLogger(__name__)

class VertexLanguageModel(LanguageModel):
    provider: VertexProvider
    models: list[genai.Client]

    def __init__(
        self,
        model_name: str,
        provider: VertexProvider,
    ):
        """Initializes the Vertex language model.

        Args:
            model_name (str): The name of the model, e.g. "gemini-2.5-flash". You can find the model name inside Vertex AI > Model Garden.
            provider (VertexProvider): The Vertex provider that manages the connection to Vertex AI services.
        """
        super().__init__(model_name, provider)

        # Vertex AI does not have multiple instance, so we just create one instance and store it in a list.
        self.models = [
            self._create_model_instance()
        ]

    @property
    def model(self) -> genai.Client:
        return super().model

    def do_generate(
            self,
            messages: list[types.MessageOrDict],
            response_type: types.ResponseType = types.ResponseType.TEXT,
            response_schema: dict[Any, Any] | None = None,
            instructions: str | None = None,
            # TODO: Convert into a customizable parameters type for all generate content config
            *,
            safety_settings: list[genai_types.SafetySetting] | None = None,
            thinking_config: genai_types.ThinkingConfig | None = None,
        ) -> types.GenerateResponse:
        """Generates content using the Vertex language model."""
        # Prepare common generation config for both text and json response types
        generation_config = self._prepare_generate_content_config(
            instructions=instructions,
            safety_settings=safety_settings,
            thinking_config=thinking_config
        )
        contents = self._prepare_messages(messages)

        if response_type == types.ResponseType.JSON:
            if response_schema is None:
                msg = "Response schema must be provided when response type is JSON."
                raise errors.InvalidArgumentError(msg)

            generation_config.response_mime_type = "application/json"
            generation_config.response_schema = self._convert_schema_for_google(response_schema)
        elif response_type == types.ResponseType.TEXT:
            # No additional config needed for text response
            pass
        else:
            msg = f"Unsupported response type: {response_type}. Supported types are TEXT and JSON."
            raise errors.InvalidArgumentError(msg)

        response = handle_error(
            self.model.models.generate_content,
            model=self.model_id,
            contents=contents,
            config=generation_config,
        )

        return self._parse_response(response, response_type)

    def do_stream(
            self,
            messages: list[types.MessageOrDict],
            response_type: types.ResponseType = types.ResponseType.TEXT,
            response_schema: dict[Any, Any] | None = None,
            instructions: str | None = None,
        ) -> Iterator[types.StreamResponse]:
        """Stream content using the Vertex language model."""
        # Prepare common generation config for both text and json response types
        generation_config = self._prepare_generate_content_config(
            instructions=instructions,
        )
        contents = self._prepare_messages(messages)

        if response_type == types.ResponseType.JSON:
            if response_schema is None:
                msg = "Response schema must be provided when response type is JSON."
                raise errors.InvalidArgumentError(msg)

            generation_config.response_mime_type = "application/json"
            generation_config.response_schema = self._convert_schema_for_google(response_schema)
        elif response_type != types.ResponseType.TEXT:
            msg = f"Unsupported response type: {response_type}. Supported types are TEXT and JSON."
            raise errors.InvalidArgumentError(msg)

        stream = handle_error(
            self.model.models.generate_content_stream,
            model=self.model_id,
            contents=contents,
            config=generation_config,
        )

        return self._parse_stream(stream)

    def _is_support_thinking_model(self) -> bool:
        """Check if the model supports thinking config. Currently, only 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite' support thinking config."""
        return re.match(r"^gemini-2\.5-(pro|flash(-lite)?)$", self.model_id) is not None

    def _create_model_instance(self) -> genai.Client:
        return genai.Client(
            vertexai=True,
            project=self.provider.project,
            location=self.provider.location,
            credentials=self.provider.credentials
        )

    def _convert_schema_for_google(self, schema: dict[Any, Any]) -> dict[Any, Any]:
        """Prepare schema for Google GenAI SDK.

        1. Converts nullable types:
           - From: {"type": ["integer", "null"]}
           - To:   {"type": "integer", "nullable": true}
        2. Removes unsupported keywords:
           - From: {"additionalProperties": False}
           - To:   (The key is removed entirely)
        """
        schema_copy = copy.deepcopy(schema)
        stack = [schema_copy]
        while stack:
            curr_obj = stack.pop()
            if isinstance(curr_obj, dict):
                # Transformation 1: Handle nullable types
                if "type" in curr_obj and isinstance(curr_obj["type"], list):
                    type_list = curr_obj["type"]

                    # Find the primary type (the one that isn't "null")
                    base_type = next((t for t in type_list if t != "null"), None)
                    if base_type:
                        curr_obj["type"] = base_type
                        # Add 'nullable: true' if 'null' is in type list
                        if "null" in type_list:
                            curr_obj["nullable"] = True

                # Transformation 2: Remove "additionalProperties: False"
                if curr_obj.get("additionalProperties") is False:
                    del curr_obj["additionalProperties"]

                # Add nested objects to the stack for processing
                stack.extend(val for val in curr_obj.values() if isinstance(val, (dict, list)))
            elif isinstance(curr_obj, list):
                # If the current object is a list, add its items to the stack
                stack.extend(item for item in curr_obj if isinstance(item, (dict, list)))
        return schema_copy

    def _prepare_generate_content_config(
            self,
            instructions: str | None,
            safety_settings: list[genai_types.SafetySetting] | None = None,
            thinking_config: genai_types.ThinkingConfig | None = None
        ) -> genai_types.GenerateContentConfig:
        config = genai_types.GenerateContentConfig()
        if instructions:
            config.system_instruction = instructions

        # Set default values
        config.temperature = 0.0
        config.top_p = 0.8
        config.top_k = 1

        # Add default safety settings
        if safety_settings:
            config.safety_settings = safety_settings
        else:
            config.safety_settings = [
                genai_types.SafetySetting(
                    category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                genai_types.SafetySetting(
                    category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                genai_types.SafetySetting(
                    category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                genai_types.SafetySetting(
                    category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                )
            ]

        # Add default thinking config
        if self._is_support_thinking_model():
            if thinking_config:
                config.thinking_config = thinking_config
            else:
                config.thinking_config = genai_types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=-1 # Dynamic thinking
                )
        elif thinking_config:
            msg = f"Model '{self.model_id}' does not support thinking config."
            raise errors.InvalidArgumentError(msg)

        return config

    def _prepare_messages(
        self,
        messages: list[types.MessageOrDict],
    ) -> list[genai_types.Content]:
        self._validate_file_parts_in_messages(messages)

        return self._messages_to_contents(messages)

    def _messages_to_contents(self, messages: list[types.MessageOrDict]) -> list[genai_types.Content]:
        """Converts a list of Message to a list of Google generative model Content objects.
        """
        p_messages = normalize_messages(messages)

        contents: list[genai_types.Content] = []
        for message in p_messages:
            role = message.role
            parts = message.parts

            content_parts: list[genai_types.Part] = []
            for part in parts:
                kind = part.kind
                if isinstance(part, types.TextPart):
                    text = part.text
                    content_parts.append(genai_types.Part.from_text(text=text))
                elif isinstance(part, types.FilePart):
                    file_info = part.file
                    content_parts.append(genai_types.Part.from_bytes(
                        data=base64.b64decode(file_info.bytes.encode("utf-8")),
                        mime_type=file_info.mime_type
                    ))
                else:
                    msg = f"Part kind '{kind}' is not supported yet."
                    raise NotImplementedError(msg)

            if role == "user":
                contents.append(genai_types.Content(
                    role="user",
                    parts=content_parts
                ))
            elif role == "assistant":
                contents.append(genai_types.Content(
                    role="model",
                    parts=content_parts
                ))
            else:
                msg = f"Message role '{role}' is not supported yet."
                raise NotImplementedError(msg)
        return contents

    def _parse_response(
            self,
            response: genai_types.GenerateContentResponse,
            response_type: types.ResponseType,
        ) -> types.GenerateResponse:
        if response is None:
            msg = "No response received from the vertex language model."
            raise errors.APIError(msg)
        candidates = response.candidates
        if not candidates or len(candidates) == 0:
            msg = "No candidates found in the response."
            raise errors.APIError(msg)
        candidate = candidates[0]
        content = candidate.content
        if not content:
            msg = "No content parts found in the response candidate."
            raise errors.APIError(msg)
        content_parts = content.parts
        if not content_parts or len(content_parts) == 0:
            msg = "No parts found in the response content."
            raise errors.APIError(msg)

        if response_type == types.ResponseType.JSON:
            return types.GenerateResponse(
                message=types.Message(
                    role=types.Role.ASSISTANT,
                    parts=self._get_reasoning_and_data_parts(content_parts),
                    message_id=response.response_id,
                ),
                metadata=types.GenerateResponseMetadata(
                    model_id=response.model_version or self.model_id,
                    usage=types.LanguageModelResponseUsage(
                        prompt_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata and response.usage_metadata.prompt_token_count else 0,
                        completion_tokens=(response.usage_metadata.thoughts_token_count + response.usage_metadata.candidates_token_count) if response.usage_metadata and response.usage_metadata.thoughts_token_count and response.usage_metadata.candidates_token_count else 0,
                    ),
                    finish_reason=candidate.finish_reason,
                    confidence=self._calculate_confidence(
                        candidate.avg_logprobs,
                        response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                    )
                )
            )
        if response_type == types.ResponseType.TEXT:
            return types.GenerateResponse(
                message=types.Message(
                    role=types.Role.ASSISTANT,
                    parts=self._get_reasoning_and_text_parts(content_parts),
                    message_id=response.response_id,
                ),
                metadata=types.GenerateResponseMetadata(
                    model_id=response.model_version or self.model_id,
                    usage=types.LanguageModelResponseUsage(
                        prompt_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata and response.usage_metadata.prompt_token_count else 0,
                        completion_tokens=(response.usage_metadata.thoughts_token_count + response.usage_metadata.candidates_token_count) if response.usage_metadata and response.usage_metadata.thoughts_token_count and response.usage_metadata.candidates_token_count else 0,
                    ),
                    finish_reason=candidate.finish_reason,
                    confidence=self._calculate_confidence(
                        candidate.avg_logprobs,
                        response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                    )
                )
            )

        return None

    def _get_reasoning_and_text_parts(self, content_part: list[genai_types.Part]) -> list[types.Part]:
        """Extracts reasoning (thoughts) and text parts from a list of content parts."""
        parts: list[types.Part] = []
        for part in content_part:
            if not part.text:
                continue
            if part.thought:
                parts.append(
                    types.ReasoningPart(
                        reasoning=part.text
                    )
                )
            else:
                parts.append(
                    types.TextPart(
                        text=part.text
                    )
                )
        return parts

    def _get_reasoning_and_data_parts(self, content_part: list[genai_types.Part]) -> list[types.Part]:
        """Extracts reasoning (thoughts) and data parts (Expect the text is a JSON string) from a list of content parts."""
        parts: list[types.Part] = []
        for part in content_part:
            if not part.text:
                continue
            if part.thought:
                parts.append(
                    types.ReasoningPart(
                        reasoning=part.text
                    )
                )
            else:
                try:
                    corrected_json_string = part.text.replace("```json", "").replace("```", "").strip()
                    json_object = json.loads(corrected_json_string)
                except json.JSONDecodeError as e:
                    msg = "JSON parsing error in Vertex language model response."
                    raise errors.ParseError(msg, text=part.text) from e
                parts.append(types.DataPart(data=json_object))
        return parts

    def _parse_stream(self, stream: Iterator[genai_types.GenerateContentResponse]) -> Iterator[types.StreamResponse]:
        """Parses the streaming response from the Vertex language model and yields StreamResponse objects."""
        first_chunk = True
        last_chunk = None

        for chunk in stream:
            # The google-genai SDK automatically aggregates the metadata in the final chunk.
            # Save the latest chunk to access this metadata after the loop.
            last_chunk = chunk

            # Extract parts from the current chunk to yield immediately
            parts: list[types.Part] = []
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                parts = self._get_reasoning_and_text_parts(chunk.candidates[0].content.parts)

            if not parts:
                continue

            yield types.StreamResponse(
                message=types.Message(
                    role=types.Role.ASSISTANT,
                    parts=parts,
                    message_id=chunk.response_id,
                ),
                append=not first_chunk,
            )
            if first_chunk:
                first_chunk = False

        # After the loop, the last chunk contains all the final metadata.
        # We yield a final chunk with this metadata.
        if last_chunk and last_chunk.candidates:
            candidate = last_chunk.candidates[0]
            usage = last_chunk.usage_metadata
            yield types.StreamResponse(
                message=types.Message(
                    role=types.Role.ASSISTANT,
                    parts=[],
                    message_id=last_chunk.response_id,
                ),
                append=True,
                last_chunk=True,
                metadata=types.GenerateResponseMetadata(
                    model_id=last_chunk.model_version,
                    usage=types.LanguageModelResponseUsage(
                        prompt_tokens=usage.prompt_token_count if usage else 0,
                        completion_tokens=(getattr(usage, "thoughts_token_count", 0) + getattr(usage, "candidates_token_count", 0)) if usage else 0,
                    ),
                    finish_reason=candidate.finish_reason,
                    confidence=self._calculate_confidence(candidate.avg_logprobs, usage.candidates_token_count if usage else 0),
                ),
            )

    def _calculate_confidence(
        self,
        avg_logprobs: float | None,
        candidates_token_count: int | None
    ):
        """Calculates the confidence score based on logprobs.

        Args:
            avg_logprobs (float | None): The average log probabilities of the generated tokens.
            candidates_token_count (int | None): The number of tokens in the candidate response.

        Returns:
            float | None: The confidence score or None if logprobs is not available.
        """
        return math.exp(avg_logprobs * candidates_token_count) if avg_logprobs and candidates_token_count else None

    def _validate_file_parts_in_messages(self, messages: list[types.MessageOrDict]) -> None:
        """Validate all FileParts in messages for model compatibility.

        Args:
            messages: List of messages to validate

        Raises:
            InvalidArgumentError: If any FilePart is incompatible with the model
        """
        normalized_messages = normalize_messages(messages)
        for message in normalized_messages:
            for part in message.parts:
                if isinstance(part, types.FilePart):
                    self._validate_file_part_for_model(part, self.model_id)

def handle_error(operation, *args, **kwargs):
    """Handles errors raised during vertex language model operations."""
    try:
        return operation(*args, **kwargs)
    except genai_errors.UnknownFunctionCallArgumentError as e:
        msg = "Unknown function call argument error when calling Vertex AI API."
        logger.exception(msg)
        raise errors.InvalidArgumentError(msg) from e
    except genai_errors.UnsupportedFunctionError as e:
        msg = "Unsupported function error when calling Vertex AI API."
        logger.exception(msg)
        raise errors.InvalidArgumentError(msg) from e
    except genai_errors.FunctionInvocationError as e:
        msg = "Function cannot invoke with the given arguments when calling Vertex AI API."
        logger.exception(msg)
        raise errors.APIError(msg) from e
    except genai_errors.APIError as e:
        msg = f"API error when calling Vertex AI API. {e.message}"
        logger.exception(msg)
        raise errors.APIError(msg, e.status, e.response) from e
    except Exception as e:
        msg = "Unexpected error when calling Vertex AI API."
        logger.exception(msg)
        raise errors.APIError(msg) from e
