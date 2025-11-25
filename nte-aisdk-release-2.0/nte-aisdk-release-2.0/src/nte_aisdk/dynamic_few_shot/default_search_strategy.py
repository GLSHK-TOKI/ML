import logging
from typing import Any

from nte_aisdk import errors

from ..search_strategy import SearchStrategy
from .example_store import DynamicFewShotExampleStore, DynamicFewShotSimpleExampleStore

logger = logging.getLogger(__name__)

class DynamicFewShotDefaultSearchStrategy(SearchStrategy):

    def __init__(self):
        super().__init__()

    def search(
            self,
            query: str,
            input_field: str,
            output_field: str,
            num_examples: int,
            environment: str | None = None,
            supplementary_input_fields: list[str] | None = None,
            space_id: str | None = None
        ):
        if self._example_store is None:
            logger.error("The 'example store' is not set before calling the search method.")
            msg = "The 'example store' is not set before calling the search method."
            raise errors.InvalidArgumentError(msg)

        # Type assertion for simple example store
        if (
            isinstance(self._example_store, DynamicFewShotSimpleExampleStore)
            and hasattr(self._example_store, "uses_environments")
            and not self._example_store.uses_environments
        ):
            return self._example_store.search(
                query=query,
                input_field=input_field,
                output_field=output_field,
                num_examples=num_examples,
                supplementary_input_fields=supplementary_input_fields,
                space_id=space_id
            )

        # Type assertion for example store
        if isinstance(self._example_store, DynamicFewShotExampleStore):
            if environment == "live":
                return self._example_store.live.search(
                    query=query,
                    input_field=input_field,
                    output_field=output_field,
                    num_examples=num_examples,
                    supplementary_input_fields=supplementary_input_fields
                )
            if environment == "staging":
                return self._example_store.staging.search(
                    query=query,
                    input_field=input_field,
                    output_field=output_field,
                    num_examples=num_examples,
                    supplementary_input_fields=supplementary_input_fields
                )
        logger.error("Invalid environment value: %s .", environment)
        msg = f"Invalid environment value: {environment}."
        raise errors.InvalidArgumentError(msg)

    def preview_search(
            self,
            query: str,
            input_field: str,
            output_field: str,
            num_examples: int,
            added_examples: list[dict[str, Any]],
            removed_ids: list[str],
            environment: str | None = None,
            supplementary_input_fields: list[str] | None = None,
            space_id: str | None = None
        ):
        if self._example_store is None:
            logger.error("The 'example store' is not set before calling the preview_search method.")
            msg = "The 'example store' is not set before calling the preview_search method."
            raise errors.InvalidArgumentError(msg)

        # Type assertion for example store
        if isinstance(self._example_store, DynamicFewShotExampleStore):
            if environment == "live":
                return self._example_store.live.preview_search(
                    query=query,
                    input_field=input_field,
                    output_field=output_field,
                    num_examples=num_examples,
                    added_examples=added_examples,
                    removed_ids=removed_ids,
                    supplementary_input_fields=supplementary_input_fields
                )
            if environment == "staging":
                return self._example_store.staging.preview_search(
                    query=query,
                    input_field=input_field,
                    output_field=output_field,
                    num_examples=num_examples,
                    added_examples=added_examples,
                    removed_ids=removed_ids,
                    supplementary_input_fields=supplementary_input_fields
                )

        # Type assertion for simple example store
        if (
            isinstance(self._example_store, DynamicFewShotSimpleExampleStore)
            and hasattr(self._example_store, "uses_environments")
            and not self._example_store.uses_environments
        ):
            return self._example_store.preview_search(
                query=query,
                input_field=input_field,
                output_field=output_field,
                num_examples=num_examples,
                added_examples=added_examples,
                removed_ids=removed_ids,
                supplementary_input_fields=supplementary_input_fields,
                space_id=space_id
            )
        logger.error("Invalid environment value: %s .", environment)
        msg = f"Invalid environment value: {environment}."
        raise errors.InvalidArgumentError(msg)