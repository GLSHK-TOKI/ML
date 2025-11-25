from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dynamic_few_shot.example_store._base_example_store import BaseExampleStore

class SearchStrategy(ABC):
    """SearchStrategy is an abstract class for search strategies."""

    _example_store: BaseExampleStore | None = None

    def __init__(self):
        self._example_store = None

    def set_example_store(self, example_store: BaseExampleStore):
        self._example_store = example_store

    @abstractmethod
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
        """Search for examples using a specific strategy."""

    @abstractmethod
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
        """Preview search results with updated examples and removed IDs."""