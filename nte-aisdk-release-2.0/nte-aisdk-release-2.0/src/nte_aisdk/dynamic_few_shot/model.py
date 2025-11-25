from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nte_aisdk.providers.vertex import VertexLanguageModel

if TYPE_CHECKING:
    from .. import SearchStrategy
    from ..language_model import LanguageModel
    from .example_store._base_example_store import BaseExampleStore
    from .prompt_template import DynamicFewShotPromptTemplate

import json
import logging

from google.genai import types as genai_types

from nte_aisdk import errors, types
from nte_aisdk.exception import throw_if_example_invalid_field, throw_if_example_missing_field
from nte_aisdk.feature import Feature
from nte_aisdk.utils import ensure_arguments
from nte_aisdk.vertex_ai.model_config import VertexAIModelConfig

logger = logging.getLogger(__name__)

class DynamicFewShotModel(Feature):
    prompt_template: DynamicFewShotPromptTemplate
    example_store: BaseExampleStore  # Update type hint
    search_strategy: SearchStrategy
    _language_model: LanguageModel

    @ensure_arguments
    def __init__(
            self,
            language_model: LanguageModel,
            prompt_template: DynamicFewShotPromptTemplate,
            example_store: BaseExampleStore,  # Update type hint
            search_strategy: SearchStrategy,
        ):
        super().__init__()
        self._language_model = language_model
        self.prompt_template = prompt_template
        self.example_store = example_store
        self.search_strategy = search_strategy

        # Perform the response type and schema checks
        self._validate_response_type_and_schema()

        self.search_strategy.set_example_store(example_store)

    @ensure_arguments
    def generate(
        self,
        query: str,
        environment: str | None= None,
        supplementary_inputs: dict[str, Any] | None = None,
        space_id: str | None = None,
        # TODO: Convert into a customizable parameters type for all generate content config
        *,
        safety_settings: list[genai_types.SafetySetting] | None = None,
        thinking_config: genai_types.ThinkingConfig | None = None,
    ) -> types.DynamicFewShotResponse:
        """Generate a response using the model with dynamic few-shot examples.

        Args:
            query (str): The input query that will perform embedding search for relevant examples.
            environment (str | None): The environment to use for the example store. Defaults to None if using DynamicFewShotSimpleExampleStore.
            supplementary_inputs (dict[str, Any] | None): Additional inputs for giving more hints in the prompt template. Defaults to None.
            space_id (str | None): Put the space id for a space-aware generation.
                It will only use the examples associate with this space id if the index contains examples from multiple spaces. Defaults to None.

        Returns:
            types.DynamicFewShotResponse: The generated response and metadata including referenced examples.
        """
        # Convert string input to Message internally for consistency with v2 architecture
        message = types.Message(
            role=types.Role.USER,
            parts=[types.TextPart(text=query)]
        )

        # Check if example store uses environments
        if self.example_store.uses_environments:
            if environment not in ("live", "staging"):
                logger.error("Invalid environment value: %s.", environment)
                msg = f"Invalid environment value: {environment}."
                raise errors.InvalidArgumentError(msg)

            # Raise exception if space_id is used, DynamicFewShotSimpleExampleStore does not support space_id
            if space_id is not None:
                logger.error("DynamicFewShotSimpleExampleStore does not support space_id.")
                msg = "DynamicFewShotSimpleExampleStore does not support space_id."
                raise errors.InvalidArgumentError(msg)
        else:
            # If example store doesn't use environments, set to None
            environment = None

        input_field = self.prompt_template.field_mapping.input
        output_field = self.prompt_template.field_mapping.output
        supplementary_input_fields = list(self.prompt_template.field_mapping.get_supplementary_inputs().values()) if supplementary_inputs else None

        examples = self.search_strategy.search(
            query=query,
            input_field=input_field,
            output_field=output_field,
            num_examples=self.prompt_template.num_examples,
            environment=environment,
            supplementary_input_fields=supplementary_input_fields,
            space_id=space_id,
        )
        return self._generate_with_examples(
            message,
            examples,
            supplementary_inputs,
            safety_settings=safety_settings,
            thinking_config=thinking_config
        )

    @ensure_arguments
    def preview_generate(
        self,
        query: str,
        added_examples: list[dict[str, Any]],
        removed_ids: list[str],
        environment: str | None = None,
        supplementary_inputs: dict[str, Any] | None = None,
        space_id: str | None = None,
        # TODO: Convert into a customizable parameters type for all generate content config
        *,
        safety_settings: list[genai_types.SafetySetting] | None = None,
        thinking_config: genai_types.ThinkingConfig | None = None,
    ) -> types.DynamicFewShotResponse:
        """Preview the generation with updated examples and removed IDs.

        Args:
            query (str): The input query that will perform embedding search for relevant examples.
            added_examples (list[dict[str, Any]]): The examples to add to the search in this generation. It will not be saved to the example store.
            removed_ids (list[str]): The IDs of the examples to remove from the search in this generation. It will not be saved to the example store.
            environment (str | None): The environment to use for the example store. Defaults to None if using DynamicFewShotSimpleExampleStore.
            supplementary_inputs (dict[str, Any] | None): Additional inputs for giving more hints in the prompt template. Defaults to None.
            space_id (str | None): Put the space id for a space-aware generation.
                It will only use the examples associate with this space id if the index contains examples from multiple spaces. Defaults to None.

        Returns:
        types.DynamicFewShotResponse: The generated response and metadata including referenced examples.
        """
        # Convert string input to Message internally for consistency with v2 architecture
        message = types.Message(
            role=types.Role.USER,
            parts=[types.TextPart(text=query)]
        )

        # Check if example store uses environments
        if self.example_store.uses_environments:
            if environment not in ("live", "staging"):
                logger.error("Invalid environment value: %s.", environment)
                msg = f"Invalid environment value: {environment}."
                raise errors.InvalidArgumentError(msg)

            # Raise exception if space_id is used, DynamicFewShotSimpleExampleStore does not support space_id
            if space_id is not None:
                logger.error("DynamicFewShotSimpleExampleStore does not support space_id.")
                msg = "DynamicFewShotSimpleExampleStore does not support space_id."
                raise errors.InvalidArgumentError(msg)
        else:
            # If example store doesn't use environments, set to None
            environment = None

        input_field = self.prompt_template.field_mapping.input
        output_field = self.prompt_template.field_mapping.output
        supplementary_input_fields = list(self.prompt_template.field_mapping.get_supplementary_inputs().values()) if supplementary_inputs else None

        if removed_ids is None:
            removed_ids = []

        for example in added_examples:
            throw_if_example_missing_field(example, [input_field, output_field])
            throw_if_example_invalid_field(example)

        examples = self.search_strategy.preview_search(
            query=query,
            input_field=input_field,
            output_field=output_field,
            num_examples=self.prompt_template.num_examples,
            added_examples=added_examples,
            removed_ids=removed_ids,
            environment=environment,
            supplementary_input_fields=supplementary_input_fields,
            space_id=space_id,
        )
        return self._generate_with_examples(
            message,
            examples,
            supplementary_inputs,
            safety_settings=safety_settings,
            thinking_config=thinking_config
        )

    def _generate_with_examples(
            self,
            message: types.Message,
            examples: list[dict[str, Any]],
            supplementary_inputs: dict[str, Any] | None = None,
            safety_settings: list[genai_types.SafetySetting] | None = None,
            thinking_config: genai_types.ThinkingConfig | None = None,
        ) -> types.DynamicFewShotResponse:
        # Extract query text from the message
        query = message.text or ""

        is_gemini_model = isinstance(self._language_model, VertexLanguageModel) and "gemini" in self._language_model.model_id.lower()

        prompt = self.prompt_template.get_dynamic_few_shot_prompt(
            examples,
            query,
            supplementary_inputs=supplementary_inputs,
            is_gemini_model=is_gemini_model
        )

        # Create message for the new LanguageModel interface
        prompt_message = types.Message(
            role=types.Role.USER,
            parts=[types.TextPart(text=prompt)]
        )

        response_type = types.ResponseType.JSON if self.prompt_template.response_type == "json" else types.ResponseType.TEXT
        response_schema = self.prompt_template.response_schema.model_json_schema() if self.prompt_template.response_schema else None

        # Disable safety settings for Gemini models
        if is_gemini_model and safety_settings is None:
                safety_settings=[
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE
                    )
                ]
        else:
            safety_settings = None

        llm_response = self._language_model.do_generate(
            messages=[prompt_message],
            response_type=response_type,
            response_schema=response_schema,
            **({"safety_settings": safety_settings} if safety_settings else {}),
            **({"thinking_config": thinking_config} if thinking_config else {})
        )

        # Add token usage tracking for rate limiting
        self._add_prompt_tokens_to_context(llm_response.metadata.usage.prompt_tokens)
        self._add_completion_tokens_to_context(llm_response.metadata.usage.completion_tokens)

        # Parse the response based on response type
        if self.prompt_template.response_type == "json":
            # For JSON responses, extract the text content (JSON as string)
            result_text = llm_response.message.text or ""
            # If the message has structured data, convert it to JSON string
            if llm_response.message.data:
                result_text = json.dumps(llm_response.message.data, ensure_ascii=False)
        else:
            # For text responses, clean up the output by removing prefix
            text = llm_response.message.text or ""
            result_text = self._remove_prefix(text.strip(), self.prompt_template.prompt_field.output)

        return types.DynamicFewShotResponse(
            text=result_text,
            examples=examples,
            metadata=types.GenerateResponseMetadata(
                model_id=llm_response.metadata.model_id,
                usage=types.LanguageModelResponseUsage(
                    prompt_tokens=llm_response.metadata.usage.prompt_tokens,
                    completion_tokens=llm_response.metadata.usage.completion_tokens
                ),
                confidence=llm_response.metadata.confidence,
                finish_reason=llm_response.metadata.finish_reason
            )
        )

    def _remove_prefix(self, text: str, prefix: str) -> str:
        """Remove the prefix from the generated text if present."""
        if text.startswith(prefix):
            text = text[len(prefix):]
            return text.lstrip(":").strip()
        return text

    def _validate_response_type_and_schema(self):
        if self.prompt_template.response_type == "text" and self.prompt_template.response_schema:
            msg = "response_type = text cannot pass response_schema"
            raise errors.InvalidArgumentError(msg)
        if self.prompt_template.response_type == "json" and not self.prompt_template.response_schema:
            msg = "response_type = json required to pass response_schema"
            raise errors.InvalidArgumentError(msg)
        if self.prompt_template.response_type == "json" and self.prompt_template.response_schema:
            self._validate_prompt_template()

    def _validate_prompt_template(self):
        # Get the fields from the response schema
        response_schema_fields = self.prompt_template.response_schema.model_fields.keys() if self.prompt_template.response_schema else []

        # Check if any of the fields match with input or output in prompt_field
        if not any(field in response_schema_fields for field in [self.prompt_template.prompt_field.input, self.prompt_template.prompt_field.output]):
            msg = "The variable in ResponseSchema class does not match with the value in DynamicFewShotPromptField class"
            raise errors.InvalidArgumentError(msg)