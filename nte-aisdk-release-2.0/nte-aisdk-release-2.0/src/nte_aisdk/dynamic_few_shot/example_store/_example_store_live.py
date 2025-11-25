from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import DynamicFewShotExampleStore

from ._example_store_env import DynamicFewShotExampleStoreEnv
from ._example_store_examples_live import DynamicFewShotExampleStoreExamplesLive


class DynamicFewShotExampleStoreLive(DynamicFewShotExampleStoreEnv):
    """DynamicFewShotExampleStoreLive is a submodule for DynamicFewShotExampleStore.

    This class is responsible for performing actions on staging index.
    """

    def __init__(self, example_store: DynamicFewShotExampleStore):
        super().__init__(example_store)
        self.examples = DynamicFewShotExampleStoreExamplesLive(self, self._example_store)