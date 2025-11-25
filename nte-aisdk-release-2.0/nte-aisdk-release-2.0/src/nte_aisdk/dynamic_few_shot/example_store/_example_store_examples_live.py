from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._example_store_env import DynamicFewShotExampleStoreEnv
    from .example_store import DynamicFewShotExampleStore

from ._example_store_examples import DynamicFewShotExampleStoreExamples


class DynamicFewShotExampleStoreExamplesLive(DynamicFewShotExampleStoreExamples):
    """DynamicFewShotExampleStoreExamplesLive is for performing crud examples on live index"""

    def __init__(
            self,
            env: DynamicFewShotExampleStoreEnv,
            example_store: DynamicFewShotExampleStore
        ):
        super().__init__(env, example_store)