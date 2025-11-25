from __future__ import annotations

import base64
import re
from typing import TYPE_CHECKING, ClassVar

from nte_aisdk import types
from nte_aisdk.feature import Feature
from nte_aisdk.pii_detection.pii_detector_batches import PIIDetectorBatches
from nte_aisdk.utils import normalize_message
from nte_aisdk.utils.errors import ensure_arguments

if TYPE_CHECKING:
    from nte_aisdk.language_model import LanguageModel

class PIIDetector(Feature):
    """A class for detecting and handling PII in text.

    Args:
        language_model (LanguageModel): Language model instance for PII detection.
        config (types.PIIDetectionConfig): Configuration for PII detection, including categories of PII and description to detect.
    """

    _PII_RESPONSE_SCHEMA: ClassVar[dict] = {
        "title": "pii_response",
        "description": "The response schema for the pii detection.",
        "type": "object",
        "properties": {
            "pii": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "The PII category from the markdown table"
                        },
                        "text": {
                            "type": "string",
                            "description": "The exact raw text from the markdown table"
                        },
                    },
                    "additionalProperties": False,
                    "required": ["entity", "text"]
                },
            },
        },
        "required": ["pii"],
        "additionalProperties": False
    }
    language_model: LanguageModel
    config: types.PIIDetectionConfig

    @ensure_arguments
    def __init__(self,
            language_model: LanguageModel,
            config: types.PIIDetectionConfigOrDict
        ):
        super().__init__()
        self.language_model = language_model
        self.config = config if isinstance(config, types.PIIDetectionConfig) else types.PIIDetectionConfig.model_validate(config) # Normalize the PII config
        self.batches = PIIDetectorBatches(self)

    def find_in_base64(self, pii_text: str, original_text: str) -> list[int]:
        """Find PII text within Base64-encoded strings in the original text.

        Args:
            pii_text: The PII text to search for (decoded)
            original_text: The original text that may contain Base64-encoded data

        Returns:
            List of character positions where the Base64 string containing the PII starts
        """
        locations = []
        # Pattern to match Base64 strings (at least 20 chars to avoid false positives)
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'

        for match in re.finditer(base64_pattern, original_text):
            base64_str = match.group()
            try:
                # Try to decode the input
                decoded_bytes = base64.b64decode(base64_str, validate=True)
                decoded_text = decoded_bytes.decode('utf-8', errors='ignore')
                if pii_text.lower() in decoded_text.lower():
                    locations.append(match.start())
            except Exception:  # noqa: S112
                # If decoding fails, skip this match
                continue
        return locations

    def detect(self, text: str, *, enable_location_mark: bool = True, context: str | None = None) -> types.PIIDetectionResponse:
        """Detect PII in the given text.

        Args:
            text (str): Text content to scan for PII
            enable_location_mark (bool): Keyword argument. Whether to include the location of detected PII in the result
            context (str | None): Optional context or description about the input text to help the LLM better understand what to look for (e.g., "This is a log file from a payment service", "Customer service conversation transcript")

        Returns:
            types.PIIDetectionResponse: A response object with the detected PII items and metadata.
        """
        # Convert string input to Message internally
        message = types.Message(
            role=types.Role.USER,
            parts=[types.TextPart(text=text)]
        )
        token_counter = self.SingleFunctionTokenCounter()

        # Normalize the input message
        p_text_message = normalize_message(message)

        # Extract the text content from the message
        text_content = p_text_message.text or ""

        # Prepare the PII detection prompt
        prompt_message = self._prepare_pii_detection_prompt(p_text_message, context=context)

        # Generate the markdown response using the language model
        markdown_response = self.language_model.do_generate(
            messages=[prompt_message],
            instructions=self.get_pii_detection_instructions()
        )

        # Track token usage
        self._add_prompt_tokens_to_context(markdown_response.metadata.usage.prompt_tokens)
        self._add_completion_tokens_to_context(markdown_response.metadata.usage.completion_tokens)
        token_counter.add_prompt_tokens(markdown_response.metadata.usage.prompt_tokens)
        token_counter.add_completion_tokens(markdown_response.metadata.usage.completion_tokens)

        # Prepare JSON conversion prompt
        json_message = self._prepare_pii_markdown_to_json_prompt(markdown_response.message.text or "")

        # Generate JSON response
        json_response = self.language_model.do_generate(
            messages=[json_message],
            instructions='You are given the PII and senstive data detection result in Markdown table. Your task is to convert the entire table and process **all** rows into a JSON format. Preserve the exact text as it appears in the Markdown table.\nJSON schema below:\npii:[{"entity":string, "text": string}]',
            response_type="json",
            response_schema=self._PII_RESPONSE_SCHEMA
        )

        # Track token usage
        self._add_prompt_tokens_to_context(json_response.metadata.usage.prompt_tokens)
        self._add_completion_tokens_to_context(json_response.metadata.usage.completion_tokens)
        token_counter.add_prompt_tokens(json_response.metadata.usage.prompt_tokens)
        token_counter.add_completion_tokens(json_response.metadata.usage.completion_tokens)

        # Extract PII results from the response
        pii_result_list = json_response.message.data.get("pii", []) if json_response.message.data else []

        # Find locations of each PII text in the input
        pii_final_result = []
        for item in pii_result_list:
            parts = [re.escape(part) for part in item["text"].split() if part]
            pattern = r"\s+".join(parts)
            matches = [m.start() for m in re.finditer(pattern, text_content)]
            if matches:
                for match_location in matches:
                    pii_item = types.PIIItem(
                        entity=item["entity"],
                        text=item["text"],
                        location=match_location if enable_location_mark else None
                    )
                    pii_final_result.append(pii_item)
            else:
                # Fallback: Try to find PII within Base64-encoded strings
                base64_locations = self.find_in_base64(item["text"], text_content) if enable_location_mark else []
                if base64_locations:
                    for match_location in base64_locations:
                        pii_item = types.PIIItem(
                            entity=item["entity"],
                            text=item["text"],
                            location=match_location
                        )
                        pii_final_result.append(pii_item)
                else:
                    # No location found in plain text or Base64
                    pii_item = types.PIIItem(
                        entity=item["entity"],
                        text=item["text"],
                        location=None
                    )
                    pii_final_result.append(pii_item)

        return types.PIIDetectionResponse(
            data=pii_final_result,
            usage=token_counter.to_usage(),
            confidence=json_response.metadata.confidence
        )

    def mask(self, text: str, detection_result: list[types.PIIItem]) -> types.PIIMaskingResponse:
        """Mask the found PII in the given text.

        Args:
            text (str): Text content to mask PII entities in
            detection_result: List of detected PII items from previous detect step

        Returns:
            types.PIIMaskingResponse: A response object with the masked text and metadata.
        """
        # Convert string input to Message internally
        message = types.Message(
            role=types.Role.USER,
            parts=[types.TextPart(text=text)]
        )

        # Normalize the input message
        p_text_message = normalize_message(message)

        pii_masking_sys_prompt = f"""
        You are given the <USER-INPUT> and the <REPLACEMENT-MAP>.
        Your task is to replace ALL occurrences of text found in the <REPLACEMENT-MAP> with their corresponding entity names throughout the ENTIRE input.

        CRITICAL RULES:
        1. Process the ENTIRE input from start to finish - do not stop early
        2. Preserve ALL text not in the <REPLACEMENT-MAP> exactly as is
        3. Preserve ALL whitespace, blank lines, newlines, and formatting
        4. Do not change any typos or spacing
        5. Replace EVERY occurrence of each PII text, even if it appears multiple times
        6. Continue processing even after blank lines or section breaks
        7. Output only processed text - DO NOT include the <USER-INPUT> tags in your response

        <REPLACEMENT-MAP>
        {self.create_mask_map(detection_result)}
        </REPLACEMENT-MAP>
        """

        pii2mask_user_prompt = f"<USER-INPUT>{p_text_message.text}</USER-INPUT>"

        # Create messages using the new types
        user_message = types.Message(
            role=types.Role.USER,
            parts=[types.TextPart(text=pii2mask_user_prompt)]
        )

        response = self.language_model.do_generate(
            messages=[user_message],
            instructions=pii_masking_sys_prompt
        )

        # Track token usage
        self._add_prompt_tokens_to_context(response.metadata.usage.prompt_tokens)
        self._add_completion_tokens_to_context(response.metadata.usage.completion_tokens)

        # Create usage statistics
        usage = types.LanguageModelResponseUsage(
            prompt_tokens=response.metadata.usage.prompt_tokens,
            completion_tokens=response.metadata.usage.completion_tokens
        )

        return types.PIIMaskingResponse(
            text=response.message.text or "",
            usage=usage,
            confidence=response.metadata.confidence
        )

    def create_mask_map(self, detection_result: list[types.PIIItem]) -> str:
        """Helper method to create the mask map string"""
        mask_map = ""
        for pii_item in detection_result:
            line = f"{pii_item.text} -> [{pii_item.entity}]\n"
            mask_map += line
        return mask_map

    def get_pii_detection_instructions(self) -> str:
        """Get the system instructions for PII detection."""
        # Format the PII categories with their names and definitions
        pii_categories_name_formatted = []
        pii_categories_definition_formatted = []

        for category in self.config.categories:
            # Add the category name to the names list
            category_str = f"- {category.name} \n"
            pii_categories_name_formatted.append(category_str)

            # Process definition if it exists
            if category.definition:
                # Convert name to uppercase and replace spaces with hyphens
                tag_name = category.name.upper().replace(" ", "-")

                # Format according to the XML-like structure
                category_str = f"<{tag_name}-PATTERN>\n{category.definition}\n</{tag_name}-PATTERN>"
                pii_categories_definition_formatted.append(category_str)

        names_section = "".join(pii_categories_name_formatted)
        definitions_section = "\n\n".join(pii_categories_definition_formatted)
        formatted_pii_cat = f"{names_section}\n{definitions_section}"

        return f"""You are given the <USER-INPUT> that might contain PII or sensitive data listed in the <PII-SENSITIVE-CAT> session. You are asked to detect the PII or sensitive data from it.
<PII-SENSITIVE-CAT>
{formatted_pii_cat}
</PII-SENSITIVE-CAT>

<INSTRUCTIONS>
1. Scan the entire <USER-INPUT>, including multiline text, to detect **all** items of PII listed in <PII-SENSITIVE-CAT>.
2. For each potential match, verify that its category is in <PII-SENSITIVE-CAT> before including it in the output.
3. Ignore any data that does not match a category in <PII-SENSITIVE-CAT>, even if it appears to be sensitive.
4. Only report PII that can be directly identified as matching a category in <PII-SENSITIVE-CAT> without requiring contextual inference from surrounding text (e.g., do not combine data with context above or below to infer PII).
5. Preserve the exact text of each detected item as it appears in <USER-INPUT>.
6. Output only the markdown table, with no additional explanations or comments.
7. If any base64 strings is found in <USER-INPUT> and its also listed as a PII category (such as token, credentials, etc.) in <PII-SENSITIVE-CAT>, then DO NOT decode the base64 strings, and report the base64 string as is if it matches any PII category. Ignore the decoded content.
8. Decode any base64 strings in <USER-INPUT> ONLY IF base64 is not listed as a PII category (such as token, credentials, etc.) in <PII-SENSITIVE-CAT>. IF ONLY base64 IS NOT a PII category, decode it and check for any potential match with <PII-SENSITIVE-CAT> inside the decoded content.
</INSTRUCTIONS>

<OUTPUT-FORMAT>
Generate the pairs of the PII and the raw text you found in markdown table
<OUTPUT-FORMAT>"""

    def _prepare_pii_detection_prompt(self, text_message: types.Message, context: str | None = None) -> types.Message:
        """Prepare the PII detection prompt message.

        Args:
            text_message (types.Message): The message object containing the text to scan for PII
            context (str | None): Optional context about the input to help the LLM understand better

        Returns:
            types.Message: The prepared message for PII detection
        """
        # Build the prompt with optional context
        if context:
            pii_detection_user_prompt = f"<CONTEXT>\n{context}\n</CONTEXT>\n\n<USER-INPUT>\n{text_message.text}\n</USER-INPUT>. Think it step by step"
        else:
            pii_detection_user_prompt = f"<USER-INPUT>\n{text_message.text}\n</USER-INPUT>. Think it step by step"

        return types.Message(
            role=types.Role.USER,
            parts=[types.TextPart(text=pii_detection_user_prompt)]
        )

    def _prepare_pii_markdown_to_json_prompt(self, result: str) -> types.Message:
        """Prepare the markdown to JSON conversion prompt message.

        Args:
            result (str): The markdown result from PII detection

        Returns:
            types.Message: The prepared message for JSON conversion
        """
        return types.Message(
            role=types.Role.USER,
            parts=[types.TextPart(text=result)]
        )
