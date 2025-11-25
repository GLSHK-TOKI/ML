import contextlib
import copy
import logging
from typing import Any, Dict, Literal, Optional, Type, TypeGuard, Union

from google import genai
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.genai import types
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_openai import AzureChatOpenAI
import openai
from pydantic import SecretStr

from ._llm_base_model_parser import LlmBaseModelParser
from .azure_openai import (
    AzureChatOpenAILoadBalancer,
    AzureOpenAIInstanceConfig,
    AzureOpenAIModelConfig,
    AzureOpenAIReasoningModelConfig,
)
from .deepseek import DeepseekInstanceConfig, DeepseekLoadBalancer, DeepseekModelConfig
from .exception import LlmModelError, ensure_arguments, handle_llm_model_operation
from .instance_config import BaseInstanceConfig
from .llama_ai import LlamaInstanceConfig, LlamaLoadBalancer, LlamaModelConfig
from .model_config import BaseModelConfig
from .utils import num_tokens_from_messages
from .vertex_ai import VertexAIInstanceConfig, VertexAILoadBalancer, VertexAIModelConfig, VertexAIReasoningModelConfig

# Check if Flask is available
try:
    from flask import g
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    g = None # type: ignore[assignment] # Fallback for non-Flask environments

logger = logging.getLogger(__name__)

class LlmBaseModel:
    """LlmBaseModel is a base class for all generative models in the AI SDK that use large language models (LLMs).

    Args:
        instance_configs (list[BaseInstanceConfig]): List of instance configurations of llm that used to generate the result.
        model_config (BaseModelConfig): Model configuration of llm that used to generate the result.
        response_type (Literal["text", "json"], optional): Response type. Defaults to "text".
        safety_settings (dict[HarmBlockThreshold], optional): Dictionary for safety settings configuration of the Vertex AI model. There are 4 configurable settings:
            1. "hate_speech": Defaults to None
            2. "dangerous_content": Defaults to None
            3. "sexually_explicit": Defaults to None
            4. "harassment": Defaults to None
    """
    instance_configs: list[BaseInstanceConfig]
    model_config: BaseModelConfig

    @ensure_arguments
    def __init__(
            self,
            instance_configs: list[BaseInstanceConfig],
            model_config: BaseModelConfig,
            safety_settings: list[types.SafetySetting] | None = None
        ):
        if safety_settings is None:
            self.safety_settings = [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                )
            ]
        else:
            if not (isinstance(model_config, VertexAIModelConfig) and self._is_vertex_instance_list(instance_configs)):
                msg = "Safety settings are only supported for Vertex AI models."
                logger.error(msg)
                raise LlmModelError(400, msg)
            self.safety_settings = safety_settings

        self.instance_configs = instance_configs
        self.model_config = model_config
        self.parser = LlmBaseModelParser(self)

        if isinstance(self.model_config, AzureOpenAIModelConfig) and self._is_azure_instance_list(self.instance_configs):
            self._lb = self._init_azure_model(self.instance_configs, self.model_config)
        elif isinstance(self.model_config, VertexAIModelConfig) and self._is_vertex_instance_list(self.instance_configs):
            self._lb = self._init_vertex_model(self.instance_configs, self.model_config)
        elif isinstance(self.model_config, LlamaModelConfig) and self._is_llama_instance_list(self.instance_configs):
            self._lb = self._init_llama_model(self.instance_configs)
        elif isinstance(self.model_config, DeepseekModelConfig) and self._is_deepseek_instance_list(self.instance_configs):
            self._lb = self._init_deepseek_model(self.instance_configs, self.model_config)

        else:
            logger.error("Unsupported model configuration")
            msg = "Unsupported model configuration"
            raise LlmModelError(400, msg)

    def model_invoke(
            self,
            messages: list[BaseMessage],
            response_type: Literal["text", "json"] = "text",
            response_schema: dict[str, Any] | None = None,
            *,
            skip_token_count: bool = False,
            stream: bool = False
        ):
        if isinstance(self.model_config, AzureOpenAIModelConfig):
            prompt_tokens = 0 if skip_token_count else num_tokens_from_messages(messages)
            self._add_prompt_tokens(prompt_tokens)
            llm = self._lb.get_instance()
            
            if stream:
                if response_type == "json":
                    # JSON streaming by json_schema
                    response_format = _convert_to_openai_response_format(response_schema, strict=True)
                    llm = llm.bind(response_format=response_format)
                response = handle_llm_model_operation(llm.stream, messages)
                return {
                    "raw": response,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 0, # Will be calculated after streaming completes
                }

            if response_type == "json":
                llm = llm.with_structured_output(
                    response_schema,
                    method="json_schema",
                    include_raw=True
                )
                response = handle_llm_model_operation(llm.invoke, messages)
                ai_message = response["raw"]
            else:
                response = handle_llm_model_operation(llm.invoke, messages)
                ai_message = response
            completion_tokens = 0 if skip_token_count else num_tokens_from_messages([ai_message])
            self._add_completion_tokens(completion_tokens)
            return {
                "raw": response,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        if isinstance(self.model_config, VertexAIModelConfig):
            generation_config: dict[str, Any] = {
                "temperature": 0,
                "top_p": 0.8,
                "top_k": 1,
                "safety_settings": self.safety_settings
            }
            if isinstance(self.model_config, VertexAIReasoningModelConfig):
                generation_config["thinking_config"] = {
                    "thinking_budget": self.model_config.thinking_budget,
                }

            if response_type == "json":
                # Function to convert all "type" keys into "type_" keys
                def _convert_type_keys(obj):
                    if isinstance(obj, dict):
                        for key in list(obj.keys()):
                            value = obj.pop(key)
                            if key == "type":
                                obj["type_"] = value
                            else:
                                obj[key] = value
                            _convert_type_keys(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            _convert_type_keys(item)

                generation_config["response_mime_type"] = "application/json"
                if response_schema is not None:
                    generation_config["response_schema"] = _convert_type_keys(copy.deepcopy(response_schema))

            contents = self._messages_to_contents(messages)
            response = handle_llm_model_operation(
                self._lb.get_instance().models.generate_content,
                model=self.model_config.model_name,
                contents=contents,
                config=generation_config,
            )
            self._handle_vertex_response_no_content(response)

            prompt_tokens = 0 if skip_token_count else response.usage_metadata.prompt_token_count
            completion_tokens = 0 if skip_token_count else response.usage_metadata.candidates_token_count
            self._add_prompt_tokens(prompt_tokens)
            self._add_completion_tokens(completion_tokens)

            return {
                "raw": response,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        if isinstance(self.model_config, LlamaModelConfig):
            contents = self._messages_to_contents_llama(messages)
            response = handle_llm_model_operation(self._lb.get_instance().chat.completions.create, model=self.model_config.model_name, messages=contents)
            prompt_tokens = 0 if skip_token_count else response.usage.prompt_tokens
            completion_tokens = 0 if skip_token_count else response.usage.completion_tokens
            self._add_prompt_tokens(prompt_tokens)
            self._add_completion_tokens(completion_tokens)

            return {
                "raw": response,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        if isinstance(self.model_config, DeepseekModelConfig):
            prompt_tokens = 0 if skip_token_count else num_tokens_from_messages(messages)
            self._add_prompt_tokens(prompt_tokens)
            llm = self._lb.get_instance()
            # DeepSeek R1 does not support structured output
            if stream:
                response = handle_llm_model_operation(llm.stream, messages)
                return {
                    "raw": response,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 0, # Will be calculated after streaming completes
                }
            response = handle_llm_model_operation(llm.invoke, messages)
            ai_message = response
            completion_tokens = 0 if skip_token_count else num_tokens_from_messages([ai_message])
            self._add_completion_tokens(completion_tokens)
            return {
                "raw": response,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        return None

    def model_stream(self, messages: list[BaseMessage]):
        """
        Stream responses from the model, yielding processed chunks.
        This method encapsulates all streaming logic and chunk processing.
        """
        if not isinstance(self.model_config, (AzureOpenAIModelConfig, DeepseekModelConfig)):
            raise TypeError("Streaming is not supported for this model.")
        
        # Get streaming response
        response = self.model_invoke(messages, stream=True)
        prompt_tokens = response.get("prompt_tokens", 0)
        
        # Process streaming chunks
        accumulated_content = ""
        accumulated_reasoning = ""
        finish_reason = None
        
        # State tracking for Deepseek reasoning tags
        is_deepseek = isinstance(self.model_config, DeepseekModelConfig)
        in_thinking = False
        buffer = ""  # Buffer for partial tags
        
        try:
            for chunk in response["raw"]:
                # Process content
                content = chunk.content if hasattr(chunk, "content") else ""
                
                if content:
                    if is_deepseek:
                        # Process content character by character for Deepseek reasoning detection
                        buffer += content
                        while buffer:
                            if not in_thinking:
                                # Look for opening tag
                                think_start = buffer.find("<think>")
                                if think_start != -1:
                                    # Yield content before thinking tag
                                    if think_start > 0:
                                        before_think = buffer[:think_start]
                                        accumulated_content += before_think
                                        yield {
                                            "type": "text",
                                            "delta": before_think
                                        }
                                    
                                    # Enter thinking mode
                                    in_thinking = True
                                    buffer = buffer[think_start + 7:]  # Remove "<think>"
                                else:
                                    # No opening tag found, yield as regular content
                                    accumulated_content += buffer
                                    yield {
                                        "type": "text",
                                        "delta": buffer
                                    }
                                    buffer = ""
                            else:
                                # Look for closing tag
                                think_end = buffer.find("</think>")
                                if think_end != -1:
                                    # Extract reasoning content
                                    reasoning_content = buffer[:think_end]
                                    if reasoning_content:
                                        accumulated_reasoning += reasoning_content
                                        yield {
                                            "type": "reasoning",
                                            "delta": reasoning_content
                                        }
                                    
                                    # Exit thinking mode
                                    in_thinking = False
                                    buffer = buffer[think_end + 8:]  # Remove "</think>"
                                else:
                                    # No closing tag yet, yield as reasoning
                                    if buffer:
                                        accumulated_reasoning += buffer
                                        yield {
                                            "type": "reasoning",
                                            "delta": buffer
                                        }
                                    buffer = ""
                    else:
                        # Non-Deepseek models: process normally
                        accumulated_content += content
                        yield {
                            "type": "text",
                            "delta": content
                        }
                
                # Check for finish_reason
                if (hasattr(chunk, "response_metadata") and 
                    isinstance(chunk.response_metadata, dict) and
                    "finish_reason" in chunk.response_metadata):
                    finish_reason = chunk.response_metadata["finish_reason"]
            
            # Handle any remaining buffer content for Deepseek
            if is_deepseek and buffer:
                if in_thinking:
                    accumulated_reasoning += buffer
                    yield {
                        "type": "reasoning",
                        "delta": buffer
                    }
                else:
                    accumulated_content += buffer
                    yield {
                        "type": "text",
                        "delta": buffer
                    }
            
            # Calculate final token usage
            total_content = accumulated_content + accumulated_reasoning
            completion_tokens = num_tokens_from_messages([AIMessage(content=total_content)])
            self._add_completion_tokens(completion_tokens)
            
            # Yield final chunk with complete information
            yield {
                "type": "finish",
                "delta": "",
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            raise e

    def _init_azure_model(
            self,
            instance_configs: list[AzureOpenAIInstanceConfig],
            model_config: AzureOpenAIModelConfig
        ):
        models = [
            AzureChatOpenAI(
                azure_endpoint=instance_config.azure_endpoint,
                azure_deployment=model_config.azure_deployment,
                api_key=SecretStr(instance_config.api_key),
                api_version=model_config.api_version,
                reasoning_effort="medium"
            )
            if isinstance(model_config, AzureOpenAIReasoningModelConfig)
            else AzureChatOpenAI(
                azure_endpoint=instance_config.azure_endpoint,
                azure_deployment=model_config.azure_deployment,
                api_key=SecretStr(instance_config.api_key),
                api_version=model_config.api_version,
                temperature=0,
                logprobs=True
            )
            for instance_config in instance_configs
        ]

        return AzureChatOpenAILoadBalancer(models)

    def _init_vertex_model(
            self,
            instance_configs: list[VertexAIInstanceConfig],
            model_config: VertexAIModelConfig
        ):
        models = []
        for instance_config in instance_configs:
            client = genai.Client(
                vertexai=True,
                project=instance_config.project,
                location=instance_config.location,
                credentials=instance_config.init_auth()
            )
            models.append(client)

        return VertexAILoadBalancer(models)

    def _init_llama_model(
            self,
            instance_configs: list[LlamaInstanceConfig]
        ):
        models = []
        creds = []
        for instance_config in instance_configs:
            credentials = instance_config.init_auth()
            auth_request = GoogleAuthRequest()

            credentials.refresh(auth_request)

            models.append(openai.OpenAI(
                base_url=f"https://{instance_config.location}-aiplatform.googleapis.com/v1/projects/{instance_config.project}/locations/{instance_config.location}/endpoints/openapi",
                api_key=credentials.token,
            ))
            creds.append(credentials)

        return LlamaLoadBalancer(models, creds)

    def _init_deepseek_model(
            self,
            instance_configs: list[DeepseekInstanceConfig],
            model_config: DeepseekModelConfig
        ):
        models = [
            AzureChatOpenAI(
                azure_endpoint=instance_config.azure_endpoint,
                api_key=SecretStr(instance_config.api_key),
                api_version=model_config.api_version,
                model_name=model_config.model_name,
                temperature=0,
            )
            for instance_config in instance_configs
        ]

        return DeepseekLoadBalancer(models)


    def _is_azure_instance_list(self, lst: list[Any]) -> TypeGuard[list[AzureOpenAIInstanceConfig]]:
        return all(isinstance(x, AzureOpenAIInstanceConfig) for x in lst)

    def _is_vertex_instance_list(self, lst: list[Any]) -> TypeGuard[list[VertexAIInstanceConfig]]:
        return all(isinstance(x, VertexAIInstanceConfig) for x in lst)

    def _is_llama_instance_list(self, lst: list[Any]) -> TypeGuard[list[LlamaInstanceConfig]]:
        return all(isinstance(x, LlamaInstanceConfig) for x in lst)

    def _is_deepseek_instance_list(self, lst: list[Any]) -> TypeGuard[list[DeepseekInstanceConfig]]:
        return all(isinstance(x, DeepseekInstanceConfig) for x in lst)

    def _messages_to_contents(self, messages: list[BaseMessage]) -> list[types.Content]:
        """Converts a list of Langchain BaseMessage objects to a list of Google generative model Content objects.
        Implementation here currently support text-only parts. And system instruction will be converted to user role.
        Ref: https://github.com/langchain-ai/langchain-google/blob/main/libs/genai/langchain_google_genai/chat_models.py
        """
        contents: list[types.Content] = []
        for i, message in enumerate(messages):
            if isinstance(message, AIMessage):
                role = "model"
                parts = [types.Part.from_text(text=str(message.content))]
                contents.append(types.Content(role=role, parts=parts))
            elif isinstance(message, HumanMessage | SystemMessage):
                role = "user"
                parts = [types.Part.from_text(text=str(message.content))]
                contents.append(types.Content(role=role, parts=parts))
            else:
                msg = f"Unexpected message with type {type(message)} at the position {i}."
                raise LlmModelError(400, msg)
        return contents

    def _messages_to_contents_llama(self, messages: list[BaseMessage]) -> list[dict]:
        """Converts a list of Langchain BaseMessage objects to a list of Llama3 Model Content objects.
        """
        output = []
        for i, message in enumerate(messages):
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                msg = f"Unexpected message with type {type(message)} at the position {i}."
                raise LlmModelError(400, msg)

            output.append({"role": role, "content": message.content})
        return output

    def _handle_vertex_response_no_content(self, response: types.GenerateContentResponse):
        candidates = getattr(response, "candidates", None)
        if not candidates or not isinstance(candidates, list) or len(candidates) == 0:
            msg = "VertexAI Model - Response has no candidates."
            logger.exception(msg)
            raise LlmModelError(400, msg)
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not content or not isinstance(parts, list) or len(parts) == 0:
            finish_reason = getattr(getattr(candidate, "finish_reason", None), "name", "Unknown")
            msg = f"VertexAI Model - Response does not have content. Finish Reason: {finish_reason}"
            logger.exception(msg)
            raise LlmModelError(400, msg)

    def _add_tokens(self, token_type: str, token_count: int):
        """Update token count for the specified type, supporting Flask or non-Flask environments.

        Args:
            token_type (str): Type of token ('prompt' or 'completion').
            token_count (int): Number of tokens to add.
        """
        attr_name = f"nte_aisdk_{token_type}_tokens"
        if FLASK_AVAILABLE and g is not None:
            # Flask environment: Use Flask's g
            with contextlib.suppress(RuntimeError): # Suppress RuntimeError if g is accessed outside a Flask app context
                setattr(g, attr_name, getattr(g, attr_name, 0) + token_count)
        else:
            # Non-Flask environment
            pass

    def _add_prompt_tokens(self, prompt_tokens: int):
        """Add prompt tokens (wrapper for add_tokens)."""
        self._add_tokens("prompt", prompt_tokens)

    def _add_completion_tokens(self, completion_tokens: int):
        """Add completion tokens (wrapper for add_tokens)."""
        self._add_tokens("completion", completion_tokens)


# Copy of the _convert_to_openai_response_format internal function from langchain
# This is used to convert various schema formats into OpenAI's structured output format.
# So we can do streaming with structured output, directly with OpenAI API.
def _convert_to_openai_response_format(schema: Union[Dict[str, Any], Type], *, strict: Optional[bool] = None) -> Union[Dict, Any]:
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
    else:
        if strict is None:
            if isinstance(schema, dict) and isinstance(schema.get("strict"), bool):
                strict = schema["strict"]
            else:
                strict = False
                
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