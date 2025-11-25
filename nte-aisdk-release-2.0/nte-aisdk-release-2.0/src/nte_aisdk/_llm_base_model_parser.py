from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any, Literal

from .azure_openai import AzureOpenAIModelConfig
from .deepseek import DeepseekModelConfig
from .exception import SDKException
from .llama_ai.model_config import LlamaModelConfig
from .vertex_ai.model_config import VertexAIModelConfig

if TYPE_CHECKING:
    from .llm_base_model import LlmBaseModel

import logging

logger = logging.getLogger(__name__)

class LlmBaseModelParser:
    def __init__(self, model: LlmBaseModel):
        self.model = model

    def parse(self, response, response_type: Literal["text", "json"] = "text") -> dict[str, Any]:
        parser = self._get_parser(self.model.model_config)
        return parser(response, response_type)

    def _parse_model_azureai_result(self, response, response_type: Literal["text", "json"]) -> dict[str, Any]:
        def _parse_azure_chat_open_ai(response, response_type: Literal["text", "json"]) -> dict[str, Any]:
            raw_response = response["raw"]["raw"] if response_type == "json" else response["raw"]
            metadata = raw_response.response_metadata
            # # Retrieve metadata
            confidence = None
            if metadata.get("logprobs") is not None:
                total_log_prob = 0.0
                for content in metadata["logprobs"]["content"]:
                    total_log_prob += content["logprob"]
                confidence = math.exp(total_log_prob)

            if response_type == "json":
                # Extract the entities from the response
                if not response["raw"]["parsing_error"]:
                    return {
                        "data": response["raw"]["parsed"],
                        "confidence": confidence,
                        "prompt_tokens": response["prompt_tokens"],
                        "completion_tokens": response["completion_tokens"],
                    }
                return {
                    "text": str(response["raw"]["parsed"]),
                    "confidence": confidence,
                    "prompt_tokens": response["prompt_tokens"],
                    "completion_tokens": response["completion_tokens"],
                }

            # Extract the entities from the response
            text = response["raw"].content.strip()
            return {
                "text": text,
                "confidence": confidence,
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
            }

        # Call the inner_function with provided arguments
        return _parse_azure_chat_open_ai(response, response_type)


    def _parse_model_vertexai_result(self, response, response_type: Literal["text", "json"]) -> dict[str, Any]:
        def _parse_vertexai(response, response_type: Literal["text", "json"]) -> dict[str, Any]:
            confidence = math.exp(response["raw"].candidates[0].avg_logprobs*response["raw"].usage_metadata.candidates_token_count)
            text = response["raw"].candidates[0].content.parts[0].text
            # Extract the entities from the response
            if response_type == "json":
                try:
                    corrected_json_string = text.replace("```json", "").replace("```", "").strip()
                    json_object = json.loads(corrected_json_string)
                except json.JSONDecodeError:
                    # Handle the case where text is not a valid JSON string
                    # Keep the original value of result["text"]
                    return {
                        "text": text,
                        "confidence": confidence,
                        "prompt_tokens": response["prompt_tokens"],
                        "completion_tokens": response["completion_tokens"],
                    }
                else:
                    return {
                        "data": json_object,
                        "confidence": confidence,
                        "prompt_tokens": response["prompt_tokens"],
                        "completion_tokens": response["completion_tokens"],
                    }
            else:
                return {
                    "text": text,
                    "confidence": confidence,
                    "prompt_tokens": response["prompt_tokens"],
                    "completion_tokens": response["completion_tokens"],
                }

        # Call the inner_function with provided arguments
        return _parse_vertexai(response, response_type)

    def _parse_model_llama_result(self, response, response_type: Literal["text", "json"]) -> dict[str, Any]:
        def _parse_llama(response, response_type: Literal["text", "json"]) -> dict[str, Any]:
            text = response["raw"].choices[0].message.content
            if response_type == "json":
                try:
                    corrected_json_string = text.strip()
                    json_object = json.loads(corrected_json_string)
                except json.JSONDecodeError:
                    # Handle the case where text is not a valid JSON string
                    # Keep the original value of result["text"]
                    return {
                        "text": text,
                        "prompt_tokens": response["prompt_tokens"],
                        "completion_tokens": response["completion_tokens"],
                    }
                else:
                    return {
                        "data": json_object,
                        "prompt_tokens": response["prompt_tokens"],
                        "completion_tokens": response["completion_tokens"],
                    }
            return {
                "text": text,
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
            }

        # Call the inner_function with provided arguments
        return _parse_llama(response, response_type)

    def _parse_model_deepseek_result(self, response, response_type: Literal["text", "json"]) -> dict[str, Any]:
        def _parse_deepseek(response, response_type: Literal["text", "json"]) -> dict[str, Any]:
            raw_response = response["raw"]

            # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/reasoning_parsers/deepseek_r1_reasoning_parser.py
            text = raw_response.content.strip()
            think_start_token = "<think>"  # noqa: S105
            think_end_token = "</think>"  # noqa: S105
            reasoning_regex = re.compile(
                rf"{think_start_token}(.*?){think_end_token}", re.DOTALL)
            if think_end_token not in text:
                return {
                    "data": {},
                    "prompt_tokens": response["prompt_tokens"],
                    "completion_tokens": response["completion_tokens"],
                }
            else:
                # Add a start token if it's missing to keep compatibility.
                if think_start_token not in text:
                    text = f"{think_start_token}{text}"
                # Use a regex to find the reasoning content
                reasoning_content = reasoning_regex.findall(text)[0]
                end_index = len(
                    f"{think_start_token}{reasoning_content}{think_end_token}"
                )
                final_output = text[end_index:]

                if response_type == "json":
                    # Extract the entities from the response
                    if len(final_output) == 0:
                        return {
                            "reasoning": reasoning_content,
                            "data": {},
                            "prompt_tokens": response["prompt_tokens"],
                            "completion_tokens": response["completion_tokens"],
                        }

                    return {
                        "reasoning": reasoning_content,
                        "data": json.loads(final_output),
                        "prompt_tokens": response["prompt_tokens"],
                        "completion_tokens": response["completion_tokens"],
                    }

                # Extract the entities from the response
                return {
                    "text": final_output,
                    "reasoning": reasoning_content,
                    "prompt_tokens": response["prompt_tokens"],
                    "completion_tokens": response["completion_tokens"],
                }

        # Call the inner_function with provided arguments
        return _parse_deepseek(response, response_type)

    def _get_parser(self, llm_model):
        if isinstance(llm_model, AzureOpenAIModelConfig):
            return self._parse_model_azureai_result
        if isinstance(llm_model, VertexAIModelConfig):
            return self._parse_model_vertexai_result
        if isinstance(llm_model, LlamaModelConfig):
            return self._parse_model_llama_result
        if isinstance(llm_model, DeepseekModelConfig):
            return self._parse_model_deepseek_result
        logger.error("Unsupported model type. Please provide a valid model type - GPT, VertexAI or Llama model")
        msg = "Unsupported model type. Please provide a valid model type - GPT, VertexAI or Llama model"
        raise SDKException(msg)
