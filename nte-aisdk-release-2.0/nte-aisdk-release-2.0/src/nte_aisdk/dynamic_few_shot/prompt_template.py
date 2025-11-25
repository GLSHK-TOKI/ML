import json
import logging
from typing import Any, Literal

from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from pydantic import BaseModel

from nte_aisdk import errors
from nte_aisdk.utils import ensure_arguments

from ..model_config import BaseModelConfig
from ..vertex_ai import VertexAIModelConfig
from .example_field_mapping import DynamicFewShotExampleFieldMapping, DynamicFewShotPromptField

logger = logging.getLogger(__name__)

class DynamicFewShotPromptTemplate:
    """DynamicFewShotPromptTemplate is a class for dynamic few shot prompt template generation.

    Args:
        instruction_text: This is the instruction text for the prompt template.
        field_mapping: A mapping of input and output fields used to construct Elasticsearch queries
            and generate prompts. This specifies which fields to use for similarity search and how to map
            the search results to the prompt template.

                <Example>
                `field_mapping = {'input': 'comment', 'output': 'category'}`
                This means that the search strategy will use 'comment' and 'category' fields in Elasticsearch queries. The search results are then
                formatted into the prompt template as shown below:

                    Example:
                        If Elasticsearch returns:
                        [{'comment': 'comment 1', 'category': 'chat', '_id': 'xxx'},
                        {'comment': 'comment 2', 'category': 'others', '_id': 'yyy'}]

                        The generated prompt will be:
                        comment: {comment 1}
                        category: chat

                        comment: {comment 2}
                        category: others
                </Example>
        prompt_field: An optional parameter to customize the prefix of the prompt template.
            If not provided, the field values from `field_mapping` will be used.

                <Example>
                    if the prompt_field is {"input": "question", "output": "answer"},
                    The prompt template will be:

                    question: {comment 1}
                    answer: chat

                    question: {comment 2}
                    answer: others
                </Example>

        num_examples: The number of examples to generate for the prompt template. The examples will be ranked by similarity.
    """  # noqa: D214
    instruction_text: str
    field_mapping: DynamicFewShotExampleFieldMapping
    prompt_field: DynamicFewShotPromptField
    num_examples: int
    response_type: Literal["text", "json"]
    response_schema: type[BaseModel] | None = None

    @ensure_arguments
    def __init__(
            self,
            instruction_text: str,
            field_mapping: DynamicFewShotExampleFieldMapping,
            num_examples: int,
            response_type: Literal["text", "json"],
            prompt_field: DynamicFewShotPromptField | None = None,
            response_schema: type[BaseModel] | None = None
        ):
        self.instruction_text = instruction_text
        self.field_mapping = field_mapping
        self.response_type = response_type
        self.response_schema = response_schema
        if prompt_field is None:
            self.prompt_field = DynamicFewShotPromptField(
                input=field_mapping.input,
                output=field_mapping.output,
                **field_mapping.get_supplementary_inputs()
            )
        else:
            self.prompt_field = prompt_field
        self.num_examples = num_examples



    def get_dynamic_few_shot_prompt(
            self,
            dynamic_examples : list,
            query : str,
            supplementary_inputs: dict[str, Any] | None = None,
            *,
            is_gemini_model: bool = False,
        ) -> str:
        self._throw_if_supplementary_input_missing_field_mapping(supplementary_inputs)

        template_text = f"{self.prompt_field.input}: {{{self.field_mapping.input}}}\n"
        if self.field_mapping.get_supplementary_inputs():
            for k, v in self.field_mapping.get_supplementary_inputs().items():
                template_text += f"{getattr(self.prompt_field, k)}: {{{v}}}\n"

        template_text += f"{self.prompt_field.output}: {{{self.field_mapping.output}}}"

        example_prompt = PromptTemplate(
            input_variables=[self.field_mapping.input, self.field_mapping.output],
            template=template_text,
        )

        if self.response_type == "json":
            instruction_text = self.instruction_text
            json_schema = self.response_schema.model_json_schema() if self.response_schema else None
            response_schema = json.dumps(json_schema["properties"]) if json_schema else None
            instruction_text += "Output a response strictly in a valid JSON format."
            if response_schema:
                instruction_text += f" The response should be a JSON object with the following schema:\n<JSONSchema>{response_schema}</JSONSchema>"
        else:
            instruction_text = self.instruction_text

        prefix = "<Example>" if is_gemini_model else "## Example\n"
        suffix = "</Example>\n" if is_gemini_model else ""

        suffix += f"{self.prompt_field.input}: {{input}}"
        for k in self.field_mapping.get_supplementary_inputs():
            suffix += f"\n{getattr(self.prompt_field, k)}: {{{k}}}"

        few_shot_prompt = FewShotPromptTemplate(
            examples=dynamic_examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", *self.field_mapping.get_supplementary_inputs().keys()],
        )
        formatter_prompt = few_shot_prompt.format(
            input=query,
            **supplementary_inputs if supplementary_inputs else {}
        )

        prompt_template = f"{instruction_text}\n{formatter_prompt}\n"

        logger.debug(prompt_template)
        return prompt_template

    def _throw_if_supplementary_input_missing_field_mapping(self, supplementary_inputs: dict[str, Any] | None):
        if not supplementary_inputs:
            supplementary_inputs = {}

        for k in self.field_mapping.get_supplementary_inputs():
            if k not in supplementary_inputs:
                msg = f"Field mapping supplementary field '{k}' is missing in the supplementary inputs"
                raise errors.InvalidArgumentError(msg)

        for k in supplementary_inputs:
            if k not in self.field_mapping.get_supplementary_inputs():
                msg = f"Supplementary input field '{k}' is not in the field mapping"
                raise errors.InvalidArgumentError(msg)

