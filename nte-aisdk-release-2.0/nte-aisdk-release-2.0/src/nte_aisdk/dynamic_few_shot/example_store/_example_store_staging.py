from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from elasticsearch.helpers import bulk

if TYPE_CHECKING:
    from .. import DynamicFewShotExampleStore

from nte_aisdk import types
from nte_aisdk.utils import ensure_arguments, handle_elasticsearch_error

from ..._constants import (
    DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX,
    DYNAMIC_FEW_SHOT_INDEX_STATES_DOCUMENT_ID,
    DYNAMIC_FEW_SHOT_INDEX_STATES_SUFFIX,
)
from ._example_store_env import DynamicFewShotExampleStoreEnv
from ._example_store_examples_staging import DynamicFewShotExampleStoreExamplesStaging

logger = logging.getLogger(__name__)

class DynamicFewShotExampleStoreStaging(DynamicFewShotExampleStoreEnv):
    """DynamicFewShotExampleStoreStaging is a submodule for DynamicFewShotExampleStore.

    This class is responsible for performing actions on staging index.
    """

    def __init__(self, example_store: DynamicFewShotExampleStore):
        super().__init__(example_store)
        self.examples = DynamicFewShotExampleStoreExamplesStaging(self, self._example_store)

    def publish(self, size=9999, search_after=None):
        """Publish the examples on staging index to the live ElasticSearch index.
        1. Reindex staging documents to the passive index
        2. Switch the live index states to the passive index
        """
        client = self._example_store.es
        response_get = handle_elasticsearch_error(
            client.get,
            index = f"{self._example_store.index_prefix}{DYNAMIC_FEW_SHOT_INDEX_STATES_SUFFIX}",
            id = DYNAMIC_FEW_SHOT_INDEX_STATES_DOCUMENT_ID
        )
        live_index = response_get["_source"]["live"]["index"]
        staging_index = response_get["_source"]["staging"]["index"]
        inactive_index = response_get["_source"]["inactive"]["index"]
        if (not search_after):
            response = handle_elasticsearch_error(
                client.delete_by_query,
                index= inactive_index,
                body= {
                    "query": {
                        "match_all": {}
                    }
                }
            )

        search_params = {}
        if size:
            search_params["size"] = size
        if search_after:
            search_params["search_after"] = search_after
        response = handle_elasticsearch_error(
            client.search,
            index= staging_index,
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
                ]
            },
            **search_params
        )
        action_bulk: list[dict] = []

        # Index each row of the DataFrame as a separate document
        for res in response["hits"]["hits"]:
            data = {
                "_op_type": "create",
                "_index": inactive_index,
                "_id": res["_id"],
            }
            data.update(res["_source"])
            action_bulk.append(data)

        bulk(client, actions=action_bulk, chunk_size=100)

        body={
            "live": {
                "index": inactive_index,
                "embedding_model": {
                    "azure_deployment": self.get_embedding_model_config().azure_deployment,
                    "api_version": self.get_embedding_model_config().api_version,
                },
            },
            "staging": {
                "index": staging_index,
                "embedding_model": {
                    "azure_deployment": self.get_embedding_model_config().azure_deployment,
                    "api_version": self.get_embedding_model_config().api_version,
                },
            },
            "inactive": {
                "index": live_index,
                "embedding_model": {
                    "azure_deployment": response_get["_source"]["live"]["embedding_model"]["azure_deployment"],
                    "api_version": response_get["_source"]["live"]["embedding_model"]["api_version"],
                },
            }
        }
        if (len(response["hits"]["hits"])<size):
            response = handle_elasticsearch_error(
                self._example_store.es.update,
                index=f"{self._example_store.index_prefix}{DYNAMIC_FEW_SHOT_INDEX_STATES_SUFFIX}",
                id=DYNAMIC_FEW_SHOT_INDEX_STATES_DOCUMENT_ID,
                refresh=True,
                body={"doc": body},
            )

            # Update the shared embedding model to sync with the new live index configuration
            staging_config = self.get_embedding_model_config()
            self._update_embedding_model(staging_config.azure_deployment, staging_config.api_version)

            return {
                    "status": "success",
                    "search_after": None,
            }

        logger.debug(response)
        return {
                "status": "success",
                "search_after": response["hits"]["hits"][len(response["hits"]["hits"])-1]["sort"]
        }

    def _update_embedding_model(self, azure_deployment: str, api_version: str):
        model_config = types.AzureModelConfigWithAPIVersion(
            azure_deployment=azure_deployment,
            api_version=api_version
        )
        self.embedding_model = self._init_embedding_model(model_config)

    def _contains_embedding_key(self, document: dict):
        return [key for key in document if DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX in key]

    @ensure_arguments
    def reindex_embeddings(self, azure_deployment: str, api_version: str, size=9999,search_after=None):
        """Update the embeddings model for the examples for the first batch"""
        self._update_embedding_model(azure_deployment, api_version)
        client = self._example_store.es
        response_get = handle_elasticsearch_error(
            client.get,
            index = f"{self._example_store.index_prefix}{DYNAMIC_FEW_SHOT_INDEX_STATES_SUFFIX}",
                        id = DYNAMIC_FEW_SHOT_INDEX_STATES_DOCUMENT_ID
        )
        search_params = {}
        if size:
            search_params["size"] = size
        if search_after:
            search_params["search_after"] = search_after
        staging_index = response_get["_source"]["staging"]["index"]
        response = handle_elasticsearch_error(
            client.search,
            index= staging_index,
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
                ]
            },
            **search_params
            )
        action_bulk: list[dict] = []
        if (len(response["hits"]["hits"])<size):
            search_after=None
        else:
            search_after=response["hits"]["hits"][len(response["hits"]["hits"])-1]["sort"]
        # Index each row of the DataFrame as a separate document
        for res in response["hits"]["hits"]:
            doc = res["_source"]
            keys: list[str] = []
            keys = self._contains_embedding_key(doc)
            updated_embeddings = {}
            for key in keys:
                updated_embeddings[key] = self.embedding_model.do_embed(doc[key.removesuffix(DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX)]).embedding
            doc.update(updated_embeddings)
            data = {
                "_op_type": "update",
                "_index": staging_index,
                "_id": res["_id"],
            }
            data.update({"_source": {"doc": doc}})
            action_bulk.append(data)

        bulk(client, actions=action_bulk, chunk_size=100)

        body={
            "staging": {
                "index": staging_index,
                "embedding_model": {
                    "azure_deployment": azure_deployment,
                    "api_version": api_version,
                },
            },
        }

        response = handle_elasticsearch_error(
            self._example_store.es.update,
            index=f"{self._example_store.index_prefix}{DYNAMIC_FEW_SHOT_INDEX_STATES_SUFFIX}",
            id=DYNAMIC_FEW_SHOT_INDEX_STATES_DOCUMENT_ID,
            body={"doc": body}
        )

        logger.debug(response)
        return {
            "status": "success",
            "search_after": search_after
        }