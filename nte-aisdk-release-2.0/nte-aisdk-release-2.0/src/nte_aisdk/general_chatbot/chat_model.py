import logging
from collections.abc import Iterator

from google.genai import types as genai_types

from nte_aisdk import errors, types
from nte_aisdk.feature import Feature
from nte_aisdk.language_model import LanguageModel
from nte_aisdk.providers.azure import AzureLanguageModel
from nte_aisdk.providers.vertex.vertex_language_model import VertexLanguageModel
from nte_aisdk.utils import ensure_arguments, normalize_messages

logger = logging.getLogger(__name__)

class GeneralChatModel(Feature):
    def __init__(
            self,
            language_model: LanguageModel,
            logger: logging.Logger = logger,
        ):
        super().__init__()
        self._language_model = language_model
        self._logger = logger
        self.INSTRUCTIONS_PROMPT = "You are AI assistant who help user to answer their question."

    @ensure_arguments
    def chat(
        self,
        messages: list[types.MessageOrDict],
        # TODO: Convert into a customizable parameters type for all generate content config
        *,
        safety_settings: list[genai_types.SafetySetting] | None = None,
        thinking_config: genai_types.ThinkingConfig | None = None,
    ) -> types.GenerateResponse:
        response = self._language_model.do_generate(
            messages=messages,
            instructions=self.INSTRUCTIONS_PROMPT,
            **({"safety_settings": safety_settings} if safety_settings else {}),
            **({"thinking_config": thinking_config} if thinking_config else {})
        )

        p_messages = normalize_messages(messages)
        self._log({
            "model_name": response.metadata.model_id,
            "question": p_messages[-1].text if p_messages else "",
            "answer": response.message.text,
            "reasoning": response.message.reasoning,
            "history": [message.model_dump(mode="json") for message in p_messages],
            "token_usage": response.metadata.usage.model_dump(mode="json"),
        })

        # Add tokens to context, for rate limiting
        self._add_prompt_tokens_to_context(response.metadata.usage.prompt_tokens)
        self._add_completion_tokens_to_context(response.metadata.usage.completion_tokens)
        return response

    @ensure_arguments
    def chat_stream(self, messages: list[types.MessageOrDict]) -> Iterator[types.StreamResponse]:
        # Temporarily restrict streaming to Azure and Vertex language models
        if not isinstance(self._language_model, (AzureLanguageModel, VertexLanguageModel)):
            msg = "Streaming is only supported for Azure and Vertex models at the moment."
            raise errors.InvalidArgumentError(msg)

        p_messages = normalize_messages(messages)
        accumulated_text = ""
        accumulated_reasoning = ""

        for chunk in self._language_model.do_stream(
            messages,
            instructions=self.INSTRUCTIONS_PROMPT,
        ):
            if chunk.last_chunk and chunk.metadata is not None:
                # Yield final response with metadata
                yield chunk

                # Add tokens count for rate limiting
                self._add_prompt_tokens_to_context(chunk.metadata.usage.prompt_tokens)
                self._add_completion_tokens_to_context(chunk.metadata.usage.completion_tokens)

                # Log the final result
                self._log({
                    "model_name": chunk.metadata.model_id,
                    "question": p_messages[-1].text if p_messages else "",
                    "answer": accumulated_text,
                    "reasoning": accumulated_reasoning,
                    "history": [message.model_dump(mode="json") for message in p_messages[:-1]],
                    "token_usage": chunk.metadata.usage.model_dump(mode="json"),
                })
            else:
                # Accumulate content for final logging
                if chunk.message.text:
                    accumulated_text += chunk.message.text
                if chunk.message.reasoning:
                    accumulated_reasoning += chunk.message.reasoning

                # Yield intermediate chunks as-is
                yield chunk

    def _log(
            self,
            data: dict,
        ):
        user = self._get_user()

        self._logger.info(
            "New message on general chat",
            extra={
                "props": {  # for json_logging to print out the extra proeprty
                    "data": {
                        "user": user,
                        "input": {
                            "question": data["question"],
                        },
                        "output": {
                            "answer": data["answer"],
                            "reasoning": data.get("reasoning"),
                            "history": data["history"]
                        },
                        "model": {
                            "name": data.get("model_name")
                        },
                        "metadata": {
                            "category": "general",
                        },
                        "usage": {
                            "prompt_tokens": data["token_usage"]["prompt_tokens"],
                            "completion_tokens": data["token_usage"]["completion_tokens"],
                            "total_tokens": data["token_usage"]["prompt_tokens"]
                            + data["token_usage"]["completion_tokens"],
                        },
                        "correlation_id": _get_correlation_id(),
                    }
                }
            },
        )

def _get_correlation_id():
    """Helper function to get correlation ID, with fallback if json_logging is not available."""
    try:
        import json_logging # type: ignore  # noqa: I001, PGH003, PLC0415
        return json_logging.get_correlation_id()
    except ImportError:
        json_logging = None
        logger.warning("json_logging is not installed, logging will not include extra properties.")

    return None
