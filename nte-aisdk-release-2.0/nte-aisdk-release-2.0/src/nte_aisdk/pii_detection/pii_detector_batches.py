from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from nte_aisdk import types
from nte_aisdk import errors
from nte_aisdk.providers.azure import AzureLanguageModel

if TYPE_CHECKING:
    from nte_aisdk.pii_detection.pii_detector import PIIDetector

class PIIDetectorBatches:
    """Handles batch operations for PII detection and masking using OpenAI's Batch API.

    This class provides structured batch processing capabilities with a two-stage detection pipeline:
    1. Markdown generation for human-readable analysis
    2. JSON conversion for structured data processing
    3. PII masking with entity replacement

    All methods return structured Pydantic models for type safety and consistency.
    """

    def __init__(self, pii_detector: PIIDetector):
        """Initialize the batch handler with dependency injection.

        Args:
            pii_detector (PIIDetector): The parent PIIDetector instance
            language_model: Language model instance (injected for better testability)
            detection_instructions: PII detection instructions (injected)
            mask_map_creator: Function to create mask maps (injected)
        """
        self._pii_detector = pii_detector
        self._language_model = pii_detector.language_model

        if not isinstance(self._language_model, AzureLanguageModel):
            msg = "Batch API is only supported for Azure OpenAI models at the moment."
            raise errors.InvalidArgumentError(msg)

        self._get_pii_detection_instructions = pii_detector.get_pii_detection_instructions
        self._create_mask_map = pii_detector.create_mask_map
        self._find_in_base64 = pii_detector.find_in_base64

    def submit_detect(self, input_texts: list[str]) -> str:
        """Submit a batch job for PII detection from input texts.

        Args:
            input_texts (list[str]): List of texts to detect PII in

        Returns:
            str: Batch job ID for tracking
        """
        # Create batch JSONL content for markdown generation
        jsonl_content = self._create_batch_jsonl_for_markdown(input_texts)

        # Submit the batch job
        return self._language_model.do_batch_submit(jsonl_content=jsonl_content)

    def submit_detect_to_json(self, markdown_results: types.BatchMarkdownResponse) -> str:
        """Submit a batch job for converting markdown PII results to JSON.

        Args:
            markdown_results (BatchMarkdownResponse): Structured markdown results from the markdown detection batch

        Returns:
            str: Batch job ID for tracking
        """
        jsonl_content = self._create_batch_for_json_conversion(markdown_results)
        return self._language_model.do_batch_submit(jsonl_content=jsonl_content)

    def submit_mask(self, detection_response: types.BatchPIIDetectionResponse, input_texts: list[str]) -> str:
        """Submit a batch job for PII masking from structured detection response and original text source.

        Args:
            detection_response (BatchPIIDetectionResponse): The structured detection response from submit_detect_to_json
            input_texts (list[str]): List of original texts corresponding to detection results

        Returns:
            str: Batch job ID for tracking
        """
        # Extract PII detection results and align with original texts
        detection_results = []
        for batch_result in detection_response.results:
            text_index = batch_result.index
            # Ensure we have the corresponding original text
            if text_index < len(input_texts):
                # Convert PIIItem objects to list for masking processing
                pii_items = batch_result.detection_response.data
                detection_results.append(pii_items)
            else:
                # Handle missing text case
                detection_results.append([])

        # Create batch JSONL content for masking
        jsonl_content = self._create_batch_jsonl_for_masking(input_texts, detection_results)

        # Submit the batch job
        return self._language_model.do_batch_submit(jsonl_content=jsonl_content)

    def retrieve(self, batch_id: str, *, original_texts: list[str] | None = None, enable_location_mark: bool = True) -> dict | types.BatchMarkdownResponse | types.BatchPIIDetectionResponse | types.BatchPIIMaskResponse:
        """Retrieve current batch status and automatically process results if completed.

        This method automatically detects the batch type based on custom_id patterns
        and processes the results accordingly:
        - Markdown batches (custom_ids: markdown_*) -> BatchMarkdownResponse with structured results
        - To-JSON batches (custom_ids: to_json_*) -> BatchPIIDetectionResponse object
        - Masking batches (custom_ids: mask_*) -> BatchPIIMaskResponse object

        Args:
            batch_id (str): The batch job ID to retrieve
            original_texts (list[str], optional): Original texts for to-json batch processing
            enable_location_mark (bool): Whether to include PII location info (to-json batches only)

        Returns:
            Union[dict, BatchMarkdownResponse, BatchPIIDetectionResponse, BatchPIIMaskResponse]:
            - For incomplete batches: dict with status info
            - For markdown batches: BatchMarkdownResponse with structured markdown results
            - For to-json batches: BatchPIIDetectionResponse with structured results
            - For masking batches: BatchPIIMaskResponse with structured results
        """
        # Get raw batch status and results
        raw_response = self._language_model.do_batch_retrieve(batch_id=batch_id)

        # If not completed, return status as-is
        if raw_response.get("status") != "completed" or not raw_response.get("results"):
            return raw_response

        raw_results = raw_response["results"]
        if not raw_results:
            return raw_response

        # Detect batch type by checking first result's custom_id
        first_custom_id = raw_results[0].get("custom_id", "")

        if first_custom_id.startswith("markdown_"):
            return self._process_markdown_results(raw_results)
        if first_custom_id.startswith("to_json_"):
            return self._process_to_json_results(raw_results, original_texts, enable_location_mark=enable_location_mark)
        if first_custom_id.startswith("mask_"):
            return self._process_masking_results(raw_results)

        # Unknown batch type - return raw results
        return {
            "status": "completed",
            "batch_type": "unknown",
            "results": raw_results
        }

    # Helper methods for creating JSONL content
    def _create_batch_jsonl_for_markdown(self, texts: list[str]) -> str:
        """Create JSONL content for markdown generation batch."""
        lines = []
        for i, text in enumerate(texts):
            prompt = f"<USER-INPUT>\n{text}\n</USER-INPUT>. Think it step by step"
            request = {
                "custom_id": f"markdown_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self._language_model.azure_deployment,
                    "messages": [
                        {"role": "system", "content": self._get_pii_detection_instructions()},
                        {"role": "user", "content": prompt}
                    ]
                }
            }
            lines.append(json.dumps(request))
        return "\n".join(lines)

    def _create_batch_for_json_conversion(self, markdown_results: types.BatchMarkdownResponse) -> str:
        """Create JSONL content for JSON conversion batch.

        Args:
            markdown_results: Structured markdown response from batch detection

        Returns:
            str: JSONL formatted content for batch submission
        """
        lines = []
        for batch_result in markdown_results.results:
            index = batch_result.index
            markdown_content = batch_result.markdown_response.data

            request = {
                "custom_id": f"to_json_{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self._language_model.azure_deployment,
                    "messages": [
                        {"role": "system", "content": 'Convert markdown table to JSON format. Schema: pii:[{"entity":string, "text": string}]'},
                        {"role": "user", "content": markdown_content}
                    ],
                    "response_format": {"type": "json_object"}
                }
            }
            lines.append(json.dumps(request))
        return "\n".join(lines)

    def _create_batch_jsonl_for_masking(self, texts: list[str], detection_results: list[list[types.PIIItem]]) -> str:
        """Create JSONL content for PII masking batch."""
        lines = []

        # Handle mismatched array lengths gracefully
        min_length = min(len(texts), len(detection_results))
        for i in range(min_length):
            text = texts[i]
            pii_items = detection_results[i]
            mask_map = self._create_mask_map(pii_items)

            masking_prompt = f"""
            You are given the <USER-INPUT> and the <REPLACEMENT-MAP>.
            Your task is to replace the text with entity name using the <REPLACEMENT-MAP>.
            All text not found in the <REPLACEMENT-MAP> should be preserved as is. Do not change any typo or spacing in the text.

            <REPLACEMENT-MAP>
            {mask_map}
            </REPLACEMENT-MAP>
            """

            user_input = f"<USER-INPUT>{text}</USER-INPUT>"

            request = {
                "custom_id": f"mask_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self._language_model.azure_deployment,
                    "messages": [
                        {"role": "system", "content": masking_prompt},
                        {"role": "user", "content": user_input}
                    ]
                }
            }
            lines.append(json.dumps(request))
        return "\n".join(lines)

    def _process_markdown_results(self, raw_results: list[dict]) -> types.BatchMarkdownResponse:
        """Process markdown batch results into BatchMarkdownResponse.

        Args:
            raw_results: List of raw batch result objects

        Returns:
            BatchMarkdownResponse: Structured markdown response with results for each text
        """
        # Map results by custom_id to handle out-of-order responses
        result_map = {result.get("custom_id", ""): result for result in raw_results
                     if result.get("custom_id", "").startswith("markdown_")}

        batch_results = []

        # Process results in order by index
        for i in range(len(result_map)):
            result = result_map.get(f"markdown_{i}")
            markdown_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            confidence = None

            if result and result.get("response", {}).get("body", {}).get("choices"):
                markdown_text = result["response"]["body"]["choices"][0]["message"]["content"]
                usage = result["response"]["body"].get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

            # Create MarkdownResponse for this text
            markdown_response = types.MarkdownResponse(
                data=markdown_text,
                usage=types.LanguageModelResponseUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                ),
                confidence=confidence
            )

            # Create BatchMarkdownResult for this text
            batch_result = types.BatchMarkdownResult(
                index=i,
                markdown_response=markdown_response
            )
            batch_results.append(batch_result)

        # Return structured BatchMarkdownResponse
        return types.BatchMarkdownResponse(
            status="completed",
            results=batch_results
        )

    def _process_to_json_results(self, raw_results: list[dict], original_texts: list[str], *, enable_location_mark: bool = True) -> dict | types.BatchPIIDetectionResponse:
        """Process to-json batch results into BatchPIIDetectionResponse.

        Args:
            raw_results: List of raw batch result objects
            original_texts: Original texts for processing
            enable_location_mark: Whether to include PII location info

        Returns:
            BatchPIIDetectionResponse or dict with error info
        """
        if not original_texts:
            return {
                "status": "completed",
                "batch_type": "to_json",
                "error": "original_texts required for to_json batch processing"
            }

        # Map results by custom_id to handle out-of-order responses
        result_map = {result.get("custom_id", ""): result for result in raw_results
                     if result.get("custom_id", "").startswith("to_json_")}

        batch_results = []

        for i, text_content in enumerate(original_texts):
            pii_items = []  # PIIItem objects for this specific text
            prompt_tokens = 0
            completion_tokens = 0
            confidence = None

            result = result_map.get(f"to_json_{i}")

            if result and result.get("response", {}).get("body", {}).get("choices"):
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                usage = result["response"]["body"].get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                try:
                    pii_data = json.loads(content)
                    # Extract confidence if available (set to None if cannot determine)
                    confidence = pii_data.get("confidence", None)

                    for item in pii_data.get("pii", []):
                        # Use regex pattern matching for multi-word PII with flexible whitespace
                        parts = [re.escape(part) for part in item["text"].split() if part]
                        pattern = r"\s+".join(parts)
                        matches = [m.start() for m in re.finditer(pattern, text_content)]

                        if matches:
                            # Create PIIItem for each occurrence found in plain text
                            for match_location in matches:
                                pii_item = types.PIIItem(
                                    entity=item["entity"],
                                    text=item["text"].strip(),
                                    location=match_location if enable_location_mark else None
                                )
                                pii_items.append(pii_item)
                        else:
                            # Fallback: Try to find PII within Base64-encoded strings
                            base64_locations = self._find_in_base64(item["text"].strip(), text_content) if enable_location_mark else []
                            if base64_locations:
                                # Create PIIItem for each occurrence found in Base64
                                for match_location in base64_locations:
                                    pii_item = types.PIIItem(
                                        entity=item["entity"],
                                        text=item["text"].strip(),
                                        location=match_location
                                    )
                                    pii_items.append(pii_item)
                            else:
                                # No location found in plain text or Base64
                                pii_item = types.PIIItem(
                                    entity=item["entity"],
                                    text=item["text"].strip(),
                                    location=None
                                )
                                pii_items.append(pii_item)
                except (json.JSONDecodeError, KeyError):
                    pass

            # Create PIIDetectionResponse for this text
            detection_response = types.PIIDetectionResponse(
                data=pii_items,
                usage=types.LanguageModelResponseUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                ),
                confidence=confidence
            )

            # Create BatchPIIDetectionResult for this text
            batch_result = types.BatchPIIDetectionResult(
                index=i,
                detection_response=detection_response
            )
            batch_results.append(batch_result)

        # Return structured BatchPIIDetectionResponse as proper API response
        return types.BatchPIIDetectionResponse(
            status="completed",
            results=batch_results
        )

    def _process_masking_results(self, raw_results: list[dict]) -> types.BatchPIIMaskResponse:
        """Process masking batch results into BatchPIIMaskResponse.

        Args:
            raw_results: List of raw batch result objects

        Returns:
            BatchPIIMaskResponse: Structured masking response
        """
        result_map = {result.get("custom_id", ""): result for result in raw_results
                     if result.get("custom_id", "").startswith("mask_")}

        batch_results = []
        for i in range(len(result_map)):
            result = result_map.get(f"mask_{i}")
            masked_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            confidence = None

            if result and result.get("response", {}).get("body", {}).get("choices"):
                masked_text = result["response"]["body"]["choices"][0]["message"]["content"]
                usage = result["response"]["body"].get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Extract confidence if available (set to None if cannot determine)
                # Note: Masking typically doesn't have confidence, but keeping for consistency
                confidence = None

            # Create PIIMaskingResponse for this text
            mask_response = types.PIIMaskingResponse(
                text=masked_text,
                usage=types.LanguageModelResponseUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                ),
                confidence=confidence
            )

            # Create BatchPIIMaskResult for this text
            batch_result = types.BatchPIIMaskResult(
                index=i,
                mask_response=mask_response
            )
            batch_results.append(batch_result)

        # Return new structured BatchPIIMaskResponse
        return types.BatchPIIMaskResponse(
            status="completed",
            results=batch_results
        )

