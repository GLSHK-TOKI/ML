from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._example_store_env import DynamicFewShotExampleStoreEnv
    from .example_store import DynamicFewShotExampleStore

import logging

from nte_aisdk.utils import ensure_arguments, handle_elasticsearch_error

from ..._constants import DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX

logger = logging.getLogger(__name__)

class DynamicFewShotExampleStoreExamples:
    """DynamicFewShotExamples is a submodule for DynamicFewShotExampleStore.

    This class is responsible for crud for examples in the vector store.
    """

    def __init__(
            self,
            env: DynamicFewShotExampleStoreEnv,
            example_store: DynamicFewShotExampleStore
        ):
        self._env = env
        self._example_store = example_store

    def _compose_return_doc(self, example: dict):
        doc = {}
        for field in example["_source"]:
            # Embedding field is excluded from the search query
            doc[field] = example["_source"][field]
        doc["_id"] = example["_id"]
        return doc

    @ensure_arguments
    def get(self, doc_id: str):
        response = handle_elasticsearch_error(
            self._example_store.es.get,
            index=self._env.get_index_name(),
            id=doc_id,
            source_excludes=[f"*{DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX}"]
        )
        logger.debug(response)
        return {
            "status": "success",
            "examples": self._compose_return_doc(dict(response))
        }

    @ensure_arguments
    def get_many(self, doc_ids: list[str]):
        response = handle_elasticsearch_error(
            self._example_store.es.mget,
            index=self._env.get_index_name(),
            body={"ids": doc_ids},
            source_excludes=[f"*{DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX}"]
        )
        logger.debug(response)
        return {
            "status": "success",
            "examples": [self._compose_return_doc(res) for res in response["docs"]]
        }

    @ensure_arguments
    def get_all(self, size=9999, search_after=None):
        """Retrieve all examples from the example store.

        Args:
            size (int): The maximum number of examples to retrieve in a call. Default is 9999.
            search_after (str): The value to start the search after. Default is None.

        Returns:
            dict: A dictionary containing the status and retrieved examples.
        """
        search_params = {}
        if size:
            search_params["size"] = size
        if search_after:
            search_params["search_after"] = search_after
        response = handle_elasticsearch_error(
            self._example_store.es.search,
            index=self._env.get_index_name(),
            body={
                "query": {
                    "match_all": {}
                },
                "sort": [
                    {
                        "last_updated_time": {
                            "order": "desc"
                        }
                    }
                ],
                "_source": {
                    "excludes": [f"*{DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX}"]
                }
            },
            **search_params
        )

        logger.debug(response)

        return {
            "status": "success",
            "examples": [self._compose_return_doc(example) for example in response["hits"]["hits"]],
            "search_after": response["hits"]["hits"][-1]["sort"] if response["hits"]["hits"] else None
        }

    @ensure_arguments
    def get_by_field(self, field_name: str, value: str):
        response = handle_elasticsearch_error(
            self._example_store.es.search,
            index=self._env.get_index_name(),
            body={
                "query": {
                    "term": {
                        f"{field_name}.keyword": value
                    },
                    "_source": {
                        "excludes": [f"*{DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX}"]
                    }
                }
            }
        )

        logger.debug(response)

        return {
            "status": "success",
            "examples": [self._compose_return_doc(example) for example in response["hits"]["hits"]]
        }