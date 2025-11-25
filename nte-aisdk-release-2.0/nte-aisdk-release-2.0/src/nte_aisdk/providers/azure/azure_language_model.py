from __future__ import annotations

import json
import logging
import math
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessageChunk, BaseMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from openai._exceptions import APIConnectionError, APIResponseValidationError, APIStatusError
from pydantic import SecretStr

from nte_aisdk import errors, types
from nte_aisdk.language_model import LanguageModel
from nte_aisdk.utils.types import convert_to_langchain_messages, normalize_messages

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nte_aisdk.providers.azure import AzureProvider

logger = logging.getLogger(__name__)

class AzureLanguageModel(LanguageModel):
    provider: AzureProvider
    models: list[AzureChatOpenAI]
    openai_models: list[AzureOpenAI]

    def __init__(
        self,
        azure_deployment: str,
        model_name: str,
        provider: AzureProvider,
    ):
        """Initializes the Azure language model.

        Args:
            azure_deployment (str): The deployment name of the language model on Azure.
            model_name (str): The name of the model, e.g. "gpt-4o", "DeepSeek-R1". You can find the model name of your deployment in the Azure AI Foundry.
            provider (AzureProvider): The Azure provider that manages the connection to Azure AI services.
        """
        super().__init__(model_name, provider)
        self.azure_deployment = azure_deployment

        # Check conflict if user provides both provider configuration and instances.
        if ((provider.azure_endpoint or provider.api_key or provider.api_version) and
                provider.instances):
            msg = "You cannot provide both provider configuration and instances. Please provide either one."
            raise errors.InvalidArgumentError(msg)

        instance_configs: list[types.AzureInstanceConfig] = []

        # Create a single instance config, with the provider configuration provided.
        if provider.azure_endpoint and provider.api_key and provider.api_version:
            instance_configs = [
                types.AzureInstanceConfig(
                    azure_endpoint=provider.azure_endpoint,
                    api_key=provider.api_key,
                    api_version=provider.api_version
                )
            ]

        # If the provider has multiple instances, we will append the models for each instance.
        if provider.instances:
            instance_configs = [
                types.AzureInstanceConfig.model_validate(instance_config)
                if isinstance(instance_config, dict) else instance_config
                for instance_config in provider.instances
            ]

        # Create a list of AzureChatOpenAI models for each instance configuration, no matter it's single or multiple instances.
        self.models = [
            self._create_model_instance(instance_config)
            for instance_config in instance_configs
        ]

        self.openai_models = [
            self._create_openai_model_instance(instance_config)
            for instance_config in instance_configs
        ]

    @property
    def model(self) -> AzureChatOpenAI:
        return super().model

    @property
    def openai_model(self) -> AzureOpenAI:
        return super().openai_model

    def do_generate(
            self,
            messages: list[types.MessageOrDict],
            response_type: types.ResponseType = types.ResponseType.TEXT,
            response_schema: dict[Any, Any] | None = None,
            instructions: str | None = None,
        ) -> types.GenerateResponse:
        """Generates content using the Azure language model."""
        langchain_messages = self._prepare_messages(
            messages,
            instructions=instructions
        )

        if response_type == types.ResponseType.JSON:
            if self._is_deepseek_reasoning_model():
                # DeepSeek R1 models does not support JSON output with schema.
                # When using DeepSeek R1 models and expect JSON output, user need to specify they want the output in JSON in their prompt.
                response = handle_error(
                    self.model.with_structured_output(
                        response_schema,
                        method="json_mode",
                        include_raw=True
                    ).invoke,
                    langchain_messages
                )
            else:
                # For other models, we need to ensure the response schema is provided for the JSON structured output.
                if response_schema is None:
                    msg = "Response schema must be provided when response type is JSON."
                    raise errors.InvalidArgumentError(msg)

                response = handle_error(
                    self.model.with_structured_output(
                        response_schema,
                        method="json_schema",
                        include_raw=True
                    ).invoke,
                    langchain_messages
                )
        elif response_type == types.ResponseType.TEXT:
            response = handle_error(
                self.model.invoke,
                langchain_messages
            )
        else:
            msg = f"Unsupported response type: {response_type}. Supported types are TEXT and JSON."
            raise errors.InvalidArgumentError(msg)

        return self._parse_response(response, response_type)

    def do_stream(
            self,
            messages: list[types.MessageOrDict],
            response_type: types.ResponseType = types.ResponseType.TEXT,
            response_schema: dict[Any, Any] | None = None,
            instructions: str | None = None,
        ) -> Iterator[types.StreamResponse]:
        """Stream content using the Azure language model."""
        langchain_messages = self._prepare_messages(
            messages,
            instructions=instructions
        )
        if response_type == types.ResponseType.JSON:
            if self._is_deepseek_reasoning_model():
                # DeepSeek R1 models does not support JSON output with schema.
                # When using DeepSeek R1 models and expect JSON output, user need to specify they want the output in JSON in their prompt.
                stream = handle_error(
                    self.model.bind(
                        response_format={"type": "json_object"},
                        stream_options={
                            "include_usage": True,
                        }
                    ).stream,
                    langchain_messages,
                )
            else:
                # For other models, we need to ensure the response schema is provided for the JSON structured output.
                if response_schema is None:
                    msg = "Response schema must be provided when response type is JSON."
                    raise errors.InvalidArgumentError(msg)

                # Langchain stream does not working properly with structured output, so we use the OpenAI stream API directly.
                response_format = _convert_to_openai_response_format(response_schema, strict=True)
                stream = handle_error(
                    self.model.bind(
                        response_format=response_format,
                        stream_options={
                            "include_usage": True,
                        }
                    ).stream,
                    langchain_messages,
                )
        elif response_type == types.ResponseType.TEXT:
            # Include usage information in the stream with stream_options
            # The usage information will include in the last chunk after the chunk include finish_reason
            stream = handle_error(
                self.model.bind(
                    stream_options={
                        "include_usage": True,
                    }
                ).stream,
                langchain_messages,
            )
        else:
            msg = f"Unsupported response type: {response_type}. Supported types are TEXT and JSON."
            raise errors.InvalidArgumentError(msg)

        return self._parse_stream(stream)

    def do_batch_submit(self, jsonl_content: str) -> str:
        """Submit a batch job to Azure OpenAI and return batch ID.

        Args:
            jsonl_content (str): The JSONL content for the batch job
        Returns:
            str: Batch job ID for tracking
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            temp_file_path = Path(f.name)

        try:
            with temp_file_path.open("rb") as f:
                batch_input_file = handle_error(
                    self.openai_model.files.create,
                    file=f,
                    purpose="batch"
                )

            batch_job = handle_error(
                self.openai_model.batches.create,
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            return batch_job.id
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()

    def do_batch_retrieve(self, batch_id: str) -> dict:
        """Retrieve current batch status and results if completed.

        Args:
            batch_id (str): The batch job ID to retrieve

        Returns:
            dict: Batch status information with format:
                {
                    "status": "queued" | "in_progress" | "completed" | "failed" | "expired" | "cancelled",
                    "results": [...] | None  # Only present if status is "completed"
                }
        """
        batch_job = handle_error(
            self.openai_model.batches.retrieve,
            batch_id
        )
        response = {
            "status": batch_job.status
        }
        if batch_job.status == "completed":
            # Download results directly
            file_id = batch_job.output_file_id or batch_job.error_file_id
            if not file_id:
                error_msg = f"No output file for batch {batch_id}"
                raise RuntimeError(error_msg)
            file_response = handle_error(
                self.openai_model.files.content,
                file_id
            )
            results = []
            for line in file_response.text.strip().split("\n"):
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            response["results"] = results
        elif batch_job.status in ["failed", "expired", "cancelled"]:
            response["error"] = f"Batch {batch_id} failed: {batch_job.status}"

        return response

    def _is_o_series_model(self) -> bool:
        return bool(re.match(r"^o\d*-.+", self.model_id.lower()))

    def _is_deepseek_reasoning_model(self) -> bool:
        return self.model_id.lower().startswith("deepseek-r1")

    def _is_gpt_5_series_model(self) -> bool:
        return self.model_id.lower().startswith("gpt-5")

    def _create_model_instance(self, instance_config: types.AzureInstanceConfig) -> AzureChatOpenAI:
        if self._is_o_series_model() or self._is_gpt_5_series_model():
            return AzureChatOpenAI(
                azure_endpoint=instance_config.azure_endpoint,
                azure_deployment=self.azure_deployment,
                api_key=SecretStr(instance_config.api_key),
                api_version=instance_config.api_version,
                reasoning_effort="medium",
            )
        if self._is_deepseek_reasoning_model():
            return AzureChatOpenAI(
                azure_endpoint=instance_config.azure_endpoint,
                api_key=SecretStr(instance_config.api_key),
                api_version=instance_config.api_version,
                model=self.azure_deployment,
                temperature=0,
            )
        return AzureChatOpenAI(
            azure_endpoint=instance_config.azure_endpoint,
            azure_deployment=self.azure_deployment,
            api_key=SecretStr(instance_config.api_key),
            api_version=instance_config.api_version,
            temperature=0,
            logprobs=True,
        )

    def _create_openai_model_instance(self, instance_config: types.AzureInstanceConfig) -> AzureOpenAI:
        return AzureOpenAI(
            api_version=instance_config.api_version,
            azure_endpoint=instance_config.azure_endpoint,
            api_key=instance_config.api_key
        )

    def _prepare_messages(
        self,
        messages: list[types.MessageOrDict],
        instructions: str | None = None,
    ) -> list[BaseMessage]:
        # Validate FileParts in messages for model compatibility
        self._validate_file_parts_in_messages(messages)

        langchain_messages = convert_to_langchain_messages(messages)

        # If instructions are provided, prepend them as a system message.
        if instructions:
            langchain_messages.insert(0, SystemMessage(content=instructions))
        return langchain_messages

    def _parse_response(
            self,
            response,
            response_type: types.ResponseType,
        ) -> types.GenerateResponse:
        if response is None:
            msg = "Response is None. Please check the response from the model."
            raise errors.APIError(msg)

        if response_type == types.ResponseType.JSON:
            metadata = response["raw"].response_metadata

            return types.GenerateResponse(
                message=types.Message(
                    role=types.Role.ASSISTANT,
                    parts=self._get_reasoning_and_data_parts(response),
                    message_id=response["raw"].id,
                ),
                metadata=types.GenerateResponseMetadata(
                    model_id=metadata.get("model_name"),
                    usage=types.LanguageModelResponseUsage(
                        prompt_tokens=metadata.get("token_usage", {}).get("prompt_tokens"),
                        completion_tokens=metadata.get("token_usage", {}).get("completion_tokens"),
                    ),
                    finish_reason=metadata.get("finish_reason"),
                    confidence=self._calculate_confidence(
                        metadata.get("logprobs"))
                )
            )
        if response_type == types.ResponseType.TEXT:
            metadata = response.response_metadata
            text = response.content.strip()
            parts = self._get_reasoning_and_text_parts(text)

            return types.GenerateResponse(
                message=types.Message(
                    role=types.Role.ASSISTANT,
                    parts=parts,
                    message_id=response.id,
                ),
                metadata=types.GenerateResponseMetadata(
                    model_id=metadata.get("model_name"),
                    usage=types.LanguageModelResponseUsage(
                        prompt_tokens=metadata.get("token_usage", {}).get("prompt_tokens"),
                        completion_tokens=metadata.get("token_usage", {}).get("completion_tokens"),
                    ),
                    finish_reason=metadata.get("finish_reason"),
                    confidence=self._calculate_confidence(
                        metadata.get("logprobs"))
                )
            )

        return None

    def _get_reasoning_and_text_parts(self, text: str) -> list[types.Part]:
        """Extracts reasoning and text parts from the response text.

        Args:
            text (str): The response text from the model.

        Returns:
            list[types.Part]: A list containing reasoning and text parts.
            If reasoning is not found, it returns a list with only the text part.
        """
        if self._is_deepseek_reasoning_model():
            # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/reasoning_parsers/deepseek_r1_reasoning_parser.py
            think_start_token = "<think>"  # noqa: S105
            think_end_token = "</think>"  # noqa: S105
            reasoning_regex = re.compile(
                rf"{think_start_token}(.*?){think_end_token}", re.DOTALL)
            if think_end_token not in text:
                # If the think_end_token is not in the text, it means no reasoning part is found.
                return [types.TextPart(text=text)]

            # Add a start token if it's missing to keep compatibility.
            if think_start_token not in text:
                text = f"{think_start_token}{text}"

            # Use a regex to find the reasoning content
            reasoning_content = reasoning_regex.findall(text)[0]
            end_index = len(
                f"{think_start_token}{reasoning_content}{think_end_token}"
            )
            text_content = text[end_index:]
            return [
                types.ReasoningPart(reasoning=reasoning_content),
                types.TextPart(text=text_content)
            ]

        # For other models, we assume the response is just text without reasoning.
        return [types.TextPart(text=text)]

    def _get_reasoning_and_data_parts(self, response) -> list[types.Part]:
        """Extracts reasoning and data parts from the response text.
        """
        if self._is_deepseek_reasoning_model():
            text = response["raw"].content.strip()
            # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/reasoning_parsers/deepseek_r1_reasoning_parser.py
            think_start_token = "<think>"  # noqa: S105
            think_end_token = "</think>"  # noqa: S105
            reasoning_regex = re.compile(
                rf"{think_start_token}(.*?){think_end_token}", re.DOTALL)
            if think_end_token not in text:
                return [types.TextPart(text=text)]

            # Add a start token if it's missing to keep compatibility.
            if think_start_token not in text:
                text = f"{think_start_token}{text}"

            # Use a regex to find the reasoning content
            reasoning_content = reasoning_regex.findall(text)[0]
            end_index = len(
                f"{think_start_token}{reasoning_content}{think_end_token}"
            )
            text_content = text[end_index:]
            try:
                json_object = json.loads(text_content)
            except json.JSONDecodeError as e:
                msg = "JSON parsing error in Azure language model response."
                raise errors.ParseError(msg, text=text_content) from e
            return [
                types.ReasoningPart(reasoning=reasoning_content),
                types.DataPart(data=json_object)
            ]

        # For other models, we assume the json response is from the parsed structured output.
        if response["parsing_error"] is not None or "parsed" not in response or response["parsed"] is None:
            msg = f"JSON parsing error in Azure language model response. {response['parsing_error']}"
            raise errors.ParseError(msg, text=str(response["raw"].content))
        return [types.DataPart(data=response["parsed"])]

    def _parse_stream(self, stream: Iterator[AIMessageChunk]) -> Iterator[types.StreamResponse]:
        metadata = {}
        first_chunk = True

        # State tracking for Deepseek reasoning tags
        in_thinking = False
        buffer = ""  # Buffer for partial tags

        for chunk in stream:
            # Process content
            content = str(chunk.content) if hasattr(chunk, "content") else ""

            if content:
                if self._is_deepseek_reasoning_model():
                    # Process content with the reasoning in DeepSeek models
                    buffer += content
                    responses, buffer, in_thinking, first_chunk = self._process_deepseek_buffer(
                        chunk.id,
                        buffer,
                        in_thinking=in_thinking,
                        first_chunk=first_chunk
                    )
                    yield from responses
                else:
                    # Non-Deepseek models: process normally
                    yield types.StreamResponse(
                        message=types.Message(
                            role=types.Role.ASSISTANT,
                            parts=[types.TextPart(text=content)],
                            message_id=chunk.id
                        ),
                        append=not first_chunk,
                    )
                    if first_chunk:
                        first_chunk = False

            # Handle any remaining buffer content for Deepseek
            if self._is_deepseek_reasoning_model() and buffer:
                part = (
                    types.ReasoningPart(reasoning=buffer)
                    if in_thinking else
                    types.TextPart(text=buffer)
                )
                yield types.StreamResponse(
                    message=types.Message(
                        role=types.Role.ASSISTANT,
                        parts=[part],
                        message_id=chunk.id
                    ),
                    append=not first_chunk
                )

            # Prepare to yield the last chunk with finish reason and usage in metadata
            # Yield with the last chunk (Last chunk should be the chunk with usage_metadata)
            if chunk.response_metadata.get("finish_reason"):
                metadata["finish_reason"] = str(chunk.response_metadata.get("finish_reason"))
            if chunk.response_metadata.get("model_name"):
                metadata["model_name"] = str(chunk.response_metadata.get("model_name"))
            if chunk.usage_metadata:
                yield types.StreamResponse(
                    message=types.Message(
                        role=types.Role.ASSISTANT,
                        parts=[],
                        message_id=chunk.id
                    ),
                    append=True,
                    last_chunk=True,
                    metadata=types.GenerateResponseMetadata(
                        model_id=metadata.get("model_name", self.model_id), # Model name is empty from the stream chunk response. Fallback to self.model_id
                        usage=types.LanguageModelResponseUsage(
                            prompt_tokens=chunk.usage_metadata.get("input_tokens", 0) if chunk.usage_metadata else 0,
                            completion_tokens=chunk.usage_metadata.get("output_tokens", 0) if chunk.usage_metadata else 0,
                        ),
                        finish_reason=metadata.get("finish_reason"),
                        confidence=None # Skip calculation of confidence for streaming
                    )
                )

    def _process_deepseek_buffer(
        self,
        chunk_id: str | None,
        buffer: str,
        *,
        in_thinking: bool,
        first_chunk: bool,
    ) -> tuple[Iterator[types.StreamResponse], str, bool, bool]:
        """Processes the buffer for Deepseek reasoning tags (<think> and </think>).

        Args:
            chunk_id (str): The ID of the current chunk.
            buffer (str): The current buffer content.
            in_thinking (bool): Whether we are currently inside a <think> tag.
            first_chunk (bool): Whether this is the first chunk being processed.

        Returns:
            tuple[Iterator[types.StreamResponse], str, bool]: A tuple containing:
                - An iterator of StreamResponse parts extracted from the buffer.
                - The remaining buffer after processing.
                - The updated in_thinking state.
                - The updated first_chunk state.
        """
        responses = []
        while buffer:
            if not in_thinking:
                # Look for opening tag
                think_start = buffer.find("<think>")
                if think_start != -1: # Found opening tag
                    # Yield text content before thinking tag
                    if think_start > 0:
                        before_think = buffer[:think_start]
                        responses.append(
                            types.StreamResponse(
                                message=types.Message(
                                    role=types.Role.ASSISTANT,
                                    parts=[types.TextPart(text=before_think)],
                                    message_id=chunk_id
                                ),
                                append=not first_chunk,
                            )
                        )
                        if first_chunk:
                            first_chunk = False

                    in_thinking = True # Enter thinking mode
                    buffer = buffer[think_start + 7:]  # Remove "<think>"
                else:
                    # No opening tag found, yield as regular content
                    responses.append(
                        types.StreamResponse(
                            message=types.Message(
                                role=types.Role.ASSISTANT,
                                parts=[types.TextPart(text=buffer)],
                                message_id=chunk_id
                            ),
                            append=not first_chunk,
                        )
                    )
                    if first_chunk:
                        first_chunk = False
                    buffer = ""
            else:
                # Look for closing tag
                think_end = buffer.find("</think>")
                if think_end != -1: # Found closing tag
                    # Yield reasoning content before closing tag
                    reasoning_content = buffer[:think_end]
                    if reasoning_content:
                        responses.append(
                            types.StreamResponse(
                                message=types.Message(
                                    role=types.Role.ASSISTANT,
                                    parts=[types.ReasoningPart(reasoning=reasoning_content)],
                                    message_id=chunk_id
                                ),
                                append=not first_chunk,
                            )
                        )
                        if first_chunk:
                            first_chunk = False

                    in_thinking = False # Exit thinking mode
                    buffer = buffer[think_end + 8:]  # Remove "</think>"
                else:
                    # No closing tag yet, yield as reasoning
                    if buffer:
                        responses.append(
                            types.StreamResponse(
                                message=types.Message(
                                    role=types.Role.ASSISTANT,
                                    parts=[types.ReasoningPart(reasoning=buffer)],
                                    message_id=chunk_id
                                ),
                                append=not first_chunk,
                            )
                        )
                        if first_chunk:
                            first_chunk = False
                    buffer = ""
        return iter(responses), buffer, in_thinking, first_chunk

    def _calculate_confidence(
        self,
        logprobs: dict[str, Any] | None,
    ):
        """Calculates the confidence score based on logprobs.

        Args:
            logprobs (dict[str, Any] | None): The logprobs from the response metadata.

        Returns:
            float | None: The confidence score or None if logprobs is not available.
        """
        if logprobs is None:
            return None

        total_log_prob = 0.0
        for content in logprobs.get("content", []):
            total_log_prob += content.get("logprob", 0.0)
        return math.exp(total_log_prob) if total_log_prob else None

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
    """Handles errors raised during Azure language model operations."""
    try:
        return operation(*args, **kwargs)
    except APIResponseValidationError as e:
        msg = f"Response validation error when calling Azure OpenAI API. {e.message}"
        logger.exception(msg)
        raise errors.APIError(msg, e.status_code) from e
    except APIConnectionError as e:
        msg = f"Connection error when calling Azure OpenAI API. {e.message}"
        logger.exception(msg)
        raise errors.APIError(msg) from e
    except APIStatusError as e:
        msg = f"API status error when calling Azure OpenAI API. {e.message}"
        logger.exception(msg)
        raise errors.APIError(msg, e.status_code) from e
    except Exception as e:
        msg = "Unexpected error when calling Azure OpenAI API."
        logger.exception(msg)
        raise errors.APIError(msg) from e

# Copy of the _convert_to_openai_response_format internal function from langchain
# This is used to convert various schema formats into OpenAI's structured output format.
# So we can do streaming with structured output, directly with OpenAI API.

from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.utils.pydantic import is_basemodel_subclass


def _convert_to_openai_response_format(schema: dict[str, Any], *, strict: bool | None = None) -> dict | Any:
    """Convert a schema to OpenAI response format.

    This function mimics the behavior of LangChain's private _convert_to_openai_response_format
    function to convert various schema formats into OpenAI's structured output format.

    Args:
        schema: The schema to convert. Can be:
            - A Pydantic BaseModel class
            - A dict with OpenAI response format structure
            - A dict with function-like schema structure
            - Any other dict schema
        strict: Whether to enforce strict schema validation

    Returns:
        A properly formatted OpenAI response format dict or BaseModel
    """
    # If it's a Pydantic BaseModel class, return it directly
    if isinstance(schema, type) and is_basemodel_subclass(schema):
        return schema
    # If it's already in OpenAI response format
    if (
        isinstance(schema, dict)
        and "json_schema" in schema
        and schema.get("type") == "json_schema"
    ):
        response_format = schema
    # If it's a function-like schema with name and schema keys
    elif isinstance(schema, dict) and "name" in schema and "schema" in schema:
        response_format = {"type": "json_schema", "json_schema": schema}
    # Convert other dict schemas
    elif strict is None:
        strict = schema["strict"] if isinstance(schema, dict) and isinstance(schema.get("strict"), bool) else False

    # Convert to OpenAI function format first
    function = convert_to_openai_function(schema, strict=strict)
    # Transform function format to json_schema format
    function["schema"] = function.pop("parameters")
    response_format = {"type": "json_schema", "json_schema": function}

    # Validate strict parameter consistency
    if strict is not None and strict != response_format["json_schema"].get("strict"):
        msg = (
            f"Output schema already has 'strict' value set to "
            f"{response_format['json_schema']['strict']} but 'strict' also passed in "
            f"as {strict}. Please make sure that 'strict' is only specified in one place."
        )
        raise ValueError(msg)

    return response_format
