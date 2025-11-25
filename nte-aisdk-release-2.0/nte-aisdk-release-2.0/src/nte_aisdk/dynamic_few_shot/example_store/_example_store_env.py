from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from nte_aisdk import errors, types
from nte_aisdk._constants import (
    DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX,
    DYNAMIC_FEW_SHOT_INDEX_STATES_DOCUMENT_ID,
    DYNAMIC_FEW_SHOT_INDEX_STATES_SUFFIX,
)
from nte_aisdk.providers.azure import AzureProvider
from nte_aisdk.utils import handle_elasticsearch_error

if TYPE_CHECKING:
    from nte_aisdk.text_embedding_model import TextEmbeddingModel

    from .. import DynamicFewShotExampleStore


logger = logging.getLogger(__name__)

class DynamicFewShotExampleStoreEnv:
    """DynamicFewShotExampleStoreEnv is an base class for staging/live example store.
    """
    _example_store: DynamicFewShotExampleStore
    embedding_model: TextEmbeddingModel

    def __init__(self, example_store: DynamicFewShotExampleStore):
        self._environment = self.__class__.__name__.replace("DynamicFewShotExampleStore", "").lower()
        self._example_store = example_store
        self.embedding_model = self._init_embedding_model(self.get_embedding_model_config())
        self._preview_search_input = ""
        self._preview_search_output = ""

    def _init_embedding_model(self, model_config: types.AzureModelConfigWithAPIVersion) -> TextEmbeddingModel:
        if not model_config:
            msg = "Model configuration not found in states index."
            raise errors.SDKError(msg)

        # Create an embedding model
        # Normalise the instance configs into Pydantic models
        # Use the config from states index and the instance(s) config from parameters
        p_embedding_instance_configs = [
            types.AzureInstanceConfigWithoutAPIVersion.model_validate(instance_config)
            if isinstance(instance_config, dict)
            else instance_config
            for instance_config in self._example_store.embedding_instance_configs
        ]
        provider = AzureProvider(
            instances=[
                types.AzureInstanceConfig(
                    azure_endpoint=instance_config.azure_endpoint,
                    api_key=instance_config.api_key,
                    api_version=model_config.api_version,
                )
                for instance_config in p_embedding_instance_configs
            ]
        )
        return provider.create_embedding_model(
            azure_deployment=model_config.azure_deployment,
            # MARK: The deployment name might not be the model name, but usually we encourage keeping them the same or including the model name in the deployment name.
            model_name=model_config.azure_deployment
        )

    def get_index_name(self):
        """Return the name of the current environment (staging/live) index."""
        states = handle_elasticsearch_error(
            self._example_store.es.get,
            index=f"{self._example_store.index_prefix}{DYNAMIC_FEW_SHOT_INDEX_STATES_SUFFIX}",
            id=DYNAMIC_FEW_SHOT_INDEX_STATES_DOCUMENT_ID
        )["_source"]
        logger.debug(states)
        return states[self._environment]["index"]

    def get_embedding_model_config(self):
        """Return the embedding deployment name and api version for the examples from states index."""
        states = handle_elasticsearch_error(
            self._example_store.es.get,
            index=f"{self._example_store.index_prefix}{DYNAMIC_FEW_SHOT_INDEX_STATES_SUFFIX}",
            id=DYNAMIC_FEW_SHOT_INDEX_STATES_DOCUMENT_ID
        )["_source"]
        response = states[self._environment]["embedding_model"]
        logger.debug(response)
        return types.AzureModelConfigWithAPIVersion(**response)

    def search(
            self,
            query: str,
            input_field: str,
            output_field: str,
            num_examples: int,
            supplementary_input_fields: list[str] | None = None,
            excluded_ids: list[str] | None = None,
        ):
        """Search for similar examples based on the query in the index of current environment.

        Args:
            query (str): The query string
            input_field (str): The field name of the input in the examples
            output_field (str): The field name of the output in the examples
            num_examples (int): The number of examples to return
            supplementary_input_fields (list[str], optional): List of additional input fields to include in results
            excluded_ids (list[str], optional): List of ids to be excluded from the elasticsearch search.
                                                Defaults to [] for non-preview search.

        Returns:
            list[dict[str, Any]]: List of examples that are similar to the query
        """
        if query is None or query == "":
            return None

        if supplementary_input_fields is None:
            supplementary_input_fields = []

        embedding_field = f"{input_field}{DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX}"
        embeddings = self.embedding_model.do_embed(query).embedding

        if excluded_ids is None:
            excluded_ids = []

        search_query = {
            "query": {
                "function_score": {
                "query": {
                    "bool": {
                    "must": [
                        {
                        "match_all": {}
                        },
                        {
                        "exists": {
                            "field": embedding_field
                        }
                        },
                        {
                        "exists": {
                            "field": input_field
                        }
                        },
                        {
                        "exists": {
                            "field": output_field
                        }
                        }
                    ],
                    "must_not": [
                        {"ids": {"values": excluded_ids}}
                    ]
                    }
                },
                "functions": [
                    {
                    "filter": {
                        "bool": {
                        "must": [
                            {
                            "exists": {
                                "field": embedding_field
                            }
                            }
                        ]
                        }
                    },
                    "weight": 1000
                    }
                ],
                "boost": 1,
                "score_mode": "sum",
                "boost_mode": "multiply"
                }
            },
            "knn": {
                "field": embedding_field,
                "query_vector": embeddings,
                "k": 100,
                "num_candidates": 100,
                "boost": 100,
                "filter": [{"ids": {"values": excluded_ids}}]
            },
            "_source": [input_field, output_field, *supplementary_input_fields],
            "size": num_examples # Searches for the top k similar embeddings candidates (set in k :), but only returns the top J results (size: J).
        }

        try :
            result = handle_elasticsearch_error(
                self._example_store.es.search,
                index=self.get_index_name(),
                body = search_query
            )
        except errors.APIError as e: #return error when there is no index_mapping match or no document
            resp = self._example_store.es.count(
                index=self.get_index_name()
            )
            if (resp["count"] == 0):
                msg = "API error in Elasticsearch, no documents found or field does not exist in index mapping in the index"
                raise errors.APIError(msg) from e
            raise errors.APIError(str(e)) from e
        if(len(result["hits"]["hits"]) == 0): #check if no document in the index
            msg = "API error in Elasticsearch, no documents in the index or no search result found"
            raise errors.APIError(msg)
        es_examples = result["hits"]["hits"]
        #Transform ES format to customized format
        examples = []
        for es_example in es_examples:
            json_data = self.trim_dict_by_keys(es_example["_source"],[input_field, output_field, *supplementary_input_fields])
            json_data["_id"] = es_example["_id"]
            json_data["_score"] = es_example["_score"]
            examples.append(json_data)

        logger.debug(examples)

        return examples

    def trim_dict_by_keys(self, dictionary, keys):
        return {k: v for k, v in dictionary.items() if k in keys}

    # Preview search result based on the query and preview changes on examples
    def preview_search(
            self,
            query: str,
            input_field: str,
            output_field: str,
            num_examples: int,
            added_examples: list[dict[str, Any]],
            removed_ids: list[str],
            supplementary_input_fields: list[str] | None = None,
        ):
        """Get the search result based on the query and preview changes on examples.
        preview the search result by score() method for the examples.

        Args:
            query (str): query string
            input_field (str): The field name of the input in the examples
            output_field (str): The field name of the output in the examples
            num_examples (int): The number of examples to return
            added_examples (list[dict[str, Any]]): list of to be added/updated examples
            removed_ids (list[str]): list of to be removed examples id
            supplementary_input_fields (list[str], optional): list of supplementary input fields to include
        """
        self._preview_search_input = input_field
        self._preview_search_output = output_field

        if supplementary_input_fields is None:
            supplementary_input_fields = []

        self._preview_supplementary_input_fields = supplementary_input_fields

        # Get the examples from Elastic search without removed examples id in elastic search query
        indexed_examples = self.search(
            query, input_field, output_field, num_examples, supplementary_input_fields, removed_ids
        )
        # Cosine similarity function with existing examples and updated examples
        indexed_scored_examples = self._compose_scored_examples(query, indexed_examples)
        updated_scored_examples = self._compose_scored_examples(query, added_examples)
        # Combine and pick top K results as prompt template
        combined_results = self._combine_results(indexed_scored_examples, updated_scored_examples)
        top_results = combined_results[:num_examples]
        # Create the list of fields to keep based on the searching parameter
        fields_to_keep = ["_id", input_field, output_field, *supplementary_input_fields, "_score"]

        # Return only the required fields
        return [{k: v for k, v in d.items() if k in fields_to_keep} for d in top_results]

    def _compose_scored_examples(self, query, examples):
        similarities = []
        query_embedding = self.embedding_model.do_embed(query).embedding
        for example in examples:
            es_input = example[self._preview_search_input]
            input_embedding = self.embedding_model.do_embed(es_input).embedding
            sim_score = self._cosine_similarity(np.array([query_embedding]), np.array([input_embedding]))

            score_dict = {
                self._preview_search_input: es_input,
                self._preview_search_output: example[self._preview_search_output],
                **{k: example[k] for k in self._preview_supplementary_input_fields if k in example},
                "_score": sim_score
            }

            # Add _id to the result for indexed examples
            if "_id" in example:
                score_dict["_id"] = example["_id"]

            similarities.append(score_dict)
        return similarities

    def _combine_results(self, es_results, context_similarities):
        all_results = context_similarities + es_results
        seen = set()
        unique_results = []

        for result in all_results:
            identifier = (
                result[self._preview_search_input],
                result[self._preview_search_output],
                *(result[k] for k in self._preview_supplementary_input_fields if k in result)
            )
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append(result)

        return sorted(unique_results, key=lambda x: x["_score"], reverse=True)

    def _cosine_similarity(self, vec1: npt.NDArray, vec2: npt.NDArray):
        dot_product = np.dot(vec1.flatten(), vec2.flatten())
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)