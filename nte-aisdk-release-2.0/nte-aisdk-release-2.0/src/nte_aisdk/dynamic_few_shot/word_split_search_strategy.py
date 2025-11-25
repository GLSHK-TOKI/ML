from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from nte_aisdk import errors

from ..search_strategy import SearchStrategy
from .example_store import DynamicFewShotExampleStore, DynamicFewShotSimpleExampleStore

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

class DynamicFewShotWordSplitSearchStrategy(SearchStrategy):

    def __init__(self):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", ";", ",", " "],
        chunk_size=2,
        chunk_overlap=0,
        length_function=self._len_fun,  # Reference to instance method
        is_separator_regex=False
        )

    def search(
            self,
            query: str,
            input_field: str,
            output_field: str,
            num_examples: int,
            environment: Optional[str] = None,
            supplementary_input_fields: list[str] | None = None,
            space_id: str | None = None
        ):
        """Search for examples using word-split search strategy.

        Args:
            query: The query string
            input_field: Field name for the input
            output_field: Field name for the output
            num_examples: Number of examples to retrieve
            environment: Optional environment name (e.g. "live" or "staging")
            supplementary_input_fields: Optional additional input fields to include
            space_id: Optional space ID for space-aware search

        Returns:
            List of examples that match the query
        """
        # Get the appropriate search function
        search_func = self._get_search_function(environment, is_preview=False)

        # Call word_split_search with the resolved search function
        return self._word_split_search(
            query=query,
            input_field=input_field,
            output_field=output_field,
            num_examples=num_examples,
            search_function=search_func,
            supplementary_input_fields=supplementary_input_fields,
            space_id=space_id
        )

    def preview_search(
            self,
            query: str,
            input_field: str,
            output_field: str,
            num_examples: int,
            added_examples: list[dict[str, Any]],
            removed_ids: list[str],
            environment: Optional[str] = None,
            supplementary_input_fields: list[str] | None = None,
            space_id: str | None = None
        ):
        """Preview search results with consideration for updated examples and removed IDs.

        Args:
            query: The query string
            input_field: Field name for the input
            output_field: Field name for the output
            num_examples: Number of examples to retrieve
            added_examples: Examples to add or update
            removed_ids: IDs of examples to remove
            environment: Optional environment name (e.g. "live" or "staging")
            supplementary_input_fields: Optional additional input fields to include
            space_id: Optional space ID for space-aware search
        Returns:
            List of examples that match the query with updates applied
        """
        # Get the appropriate preview search function
        preview_search_func = self._get_search_function(environment, is_preview=True)

        # Call word_split_search with the resolved search function
        return self._word_split_search(
            query=query,
            input_field=input_field,
            output_field=output_field,
            num_examples=num_examples,
            search_function=preview_search_func,
            added_examples=added_examples,
            removed_ids=removed_ids,
            supplementary_input_fields=supplementary_input_fields,
            space_id=space_id
        )

    def _get_search_function(self, environment: Optional[str], *, is_preview: bool) -> Callable:
        """Get the appropriate search function based on the example store type and environment.

        Args:
            environment: The environment to search in ("live" or "staging"), or None
            is_preview: Whether to use the preview search function

        Returns:
            Callable: The search function to use

        Raises:
            SDKException: If environment is required but invalid, or if example store is not set
        """
        if self._example_store is None:
            method_name = "preview_search" if is_preview else "search"
            msg = f"The 'example store' is not set before calling the {method_name} method."
            logger.error(msg)
            raise errors.InvalidArgumentError(msg)

        # Check if example store uses environments
        # Type assertion for simple example store
        if (
            isinstance(self._example_store, DynamicFewShotSimpleExampleStore)
            and hasattr(self._example_store, "uses_environments")
            and not self._example_store.uses_environments
        ):
            # Environment not needed - return appropriate method directly from the store
            if is_preview:
                return self._example_store.preview_search
            return self._example_store.search

        # Example store uses environments - validate environment parameter
        # Type assertion for example store
        if isinstance(self._example_store, DynamicFewShotExampleStore):
            if environment is None:
                msg = "Environment parameter is required for this example store"
                logger.error(msg)
                raise errors.InvalidArgumentError(msg)
            # Return appropriate environment-specific function
            if environment == "live":
                return self._example_store.live.preview_search if is_preview else self._example_store.live.search
            if environment == "staging":
                return self._example_store.staging.preview_search if is_preview else self._example_store.staging.search
        # Invalid environment provided
        msg = f"Invalid environment value: {environment}."
        logger.error(msg)
        raise errors.InvalidArgumentError(msg)

    # Rest of the class methods remain the same
    def _len_fun(self, text):
        """Helper function to count words in text"""
        return len(text.strip().split())

    def _split(self, input_text):
        """Split text into chunks for processing."""
        if not input_text:
            return []
        return self.text_splitter.split_text(input_text)

    def _remove_duplicates(self, json_array, field):
        """Remove duplicate items from a list based on a specific field."""
        if not json_array:
            return []
        return list({item[field]: item for item in json_array}.values())

    def _remove_duplicates_from_another_list(self, array_to_be_remove, array_value_to_check, field):
        sub_json_values = {item[field] for item in array_value_to_check}
        array_to_be_remove = [item for item in array_to_be_remove if item[field] not in sub_json_values]
        return array_to_be_remove

    def _sort_examples_by_score(self, examples):
        """Sort examples in descending order based on their score."""
        return sorted(examples, key=lambda x: x.get("_score", 0), reverse=True)

    def _word_split_search(
            self,
            query: str,
            input_field: str,
            output_field: str,
            num_examples: int,
            search_function,
            added_examples: list[dict[str, Any]] | None = None,
            removed_ids: list[str] | None = None,
            supplementary_input_fields: list[str] | None = None,
            space_id: str | None = None
        ):
        """Search for similar examples based on the query in the index of current environment."""
        if not query:
            return None

        if self._example_store is None:
            logger.error("The 'example store' is not set before calling the word_split_search method.")
            msg = "The 'example store' is not set before calling the word_split_search method."
            raise errors.InvalidArgumentError(msg)

        # Call search function appropriately based on its signature
        is_preview_search = (search_function.__name__ == "preview_search")
        if is_preview_search:
            # For preview search
            examples = search_function(
                query,
                input_field,
                output_field,
                num_examples,
                added_examples or [],
                removed_ids or [],
                supplementary_input_fields=supplementary_input_fields,
                **{"space_id": space_id} if space_id is not None else {},
            )
        else:
            # For regular search
            examples = search_function(
                query,
                input_field,
                output_field,
                num_examples,
                supplementary_input_fields,
                **{"space_id": space_id} if space_id is not None else {},
            )

        if not examples:
            return []

        # Split the query into words
        sub_text_list = self._split(query)

        # Collect examples from each subquery
        sub_text_examples = []
        for sub_text in sub_text_list:
            if is_preview_search:
                results = search_function(
                    sub_text,
                    input_field,
                    output_field,
                    num_examples,
                    added_examples or [],
                    removed_ids or [],
                    supplementary_input_fields=supplementary_input_fields,
                    **{"space_id": space_id} if space_id is not None else {},
                )
            else:
                results = search_function(
                    sub_text,
                    input_field,
                    output_field,
                    num_examples,
                    supplementary_input_fields,
                    **{"space_id": space_id} if space_id is not None else {},
                )

            if results:
                sub_text_examples.extend(results)

        # Rest of your method remains the same
        if not sub_text_examples:
            return examples[:num_examples]# Return only main results if no sub-results

        # Remove duplicates from sub-results
        sub_text_examples = self._remove_duplicates(sub_text_examples, input_field)

        # Remove any examples that are already in the main results
        sub_text_examples = self._remove_duplicates_from_another_list(
            sub_text_examples, examples, input_field
        )

        examples = self._sort_examples_by_score(examples)
        sub_text_examples = self._sort_examples_by_score(sub_text_examples)

        # Calculate how many examples to take from each source
        examples_count = (num_examples + 1) // 2  # Ceiling division to handle odd numbers
        sub_text_count = num_examples - examples_count  # Remaining count for sub_text_examples

        # Create combined examples with proper distribution
        combined_examples = examples[:examples_count] + sub_text_examples[:sub_text_count]

        # Sort and limit results
        sorted_examples = self._sort_examples_by_score(combined_examples)
        fields_to_keep = ["_id", input_field, output_field, *(supplementary_input_fields or []), "_score"]
        return [{k: v for k, v in example.items() if k in fields_to_keep} for example in sorted_examples]