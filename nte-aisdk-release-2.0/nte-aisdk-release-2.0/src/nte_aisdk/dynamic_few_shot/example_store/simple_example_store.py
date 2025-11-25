from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from elasticsearch.helpers import bulk

from nte_aisdk import errors
from nte_aisdk.utils import ensure_arguments, handle_elasticsearch_error

from ..._constants import (
    DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX,
)
from ...vector_store import VectorStore
from ._base_example_store import BaseExampleStore
from ._simple_example_store_examples import DynamicFewShotSimpleExampleStoreExamples

if TYPE_CHECKING:
    from nte_aisdk.text_embedding_model import TextEmbeddingModel

logger = logging.getLogger(__name__)


class DynamicFewShotSimpleExampleStore(VectorStore, BaseExampleStore):
    """A simpler implementation of DynamicFewShotExampleStore with basic functionality."""
    @property
    def uses_environments(self) -> bool:
        return False

    @ensure_arguments
    def __init__(
            self,
            host: str,
            basic_auth: str | tuple[str, str],
            embedding_model: TextEmbeddingModel,
            index_name: str,
        ):
        """Initialize the DynamicFewShotSimpleExampleStore.

        Args:
            host: Elasticsearch host URL
            basic_auth: Authentication credentials
            embedding_model: Azure embedding model for creating embeddings
            index_name: Name for the Elasticsearch index
        """
        super().__init__(host, basic_auth)
        self.embedding_model = embedding_model
        self.index_name = index_name
        self._preview_search_input = ""
        self._preview_search_output = ""
        self.examples = DynamicFewShotSimpleExampleStoreExamples(self)

    #
    # Core functionality
    #
    def _cosine_similarity(self, vec1: npt.NDArray, vec2: npt.NDArray) -> float:
        try:
            vec1_flat = vec1.flatten()
            vec2_flat = vec2.flatten()
            dot_product = np.dot(vec1_flat, vec2_flat)
            norm_a = np.linalg.norm(vec1_flat)
            norm_b = np.linalg.norm(vec2_flat)

            if norm_a == 0 or norm_b == 0:
                msg = f"Vector norm cannot be zero: {norm_a}, {norm_b}"
                raise ValueError(msg)  # noqa: TRY301

            return float(dot_product / (norm_a * norm_b))
        except ValueError:
            logger.exception("Error calculating cosine similarity: %s")
            raise

    def search(
            self,
            query: str,
            input_field: str,
            output_field: str,
            num_examples: int,
            supplementary_input_fields: list[str] | None = None,
            excluded_ids: list[str] | None = None,
            space_id: str | None = None,
        ) -> list[dict[str, Any]] | None:

        if not query:
            return None

        if supplementary_input_fields is None:
            supplementary_input_fields = []

        if not self.embedding_model:
            msg = "Embedding model is not initialized."
            raise errors.InvalidArgumentError(msg)

        embeddings = self.embedding_model.do_embed(query).embedding

        # Ensure excluded_ids is a list
        if excluded_ids is None or not isinstance(excluded_ids, list):
            excluded_ids = []

        # Build search query
        search_query = self._build_search_query(
            embeddings,
            input_field,
            output_field,
            num_examples,
            excluded_ids,
            supplementary_input_fields,
            space_id
        )

        try :
            result = handle_elasticsearch_error(
                self.es.search,
                index=self.index_name,
                body = search_query
            )
        except errors.APIError as e: #return error when there is no index_mapping match or no document
            resp = self.es.count(
                index=self.index_name
            )
            if (resp["count"] == 0):
                msg = "API error in Elasticsearch, no documents found or field does not exist in index mapping in the index"
                raise errors.APIError(msg, e.status_code) from e
            raise errors.APIError(str(e), e.status_code) from e
        if(len(result["hits"]["hits"]) == 0): #check if no document in the index
            msg = "API error in Elasticsearch, no documents in the index or no search result found"
            raise errors.APIError(msg, 400)
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

    def _build_search_query(
            self,
            embeddings: list[float],
            input_field: str,
            output_field: str,
            num_examples: int,
            excluded_ids: list[str],
            supplementary_input_fields: list[str] | None = None,
            space_id: str | None = None
        ) -> dict[str, Any]:
        if supplementary_input_fields is None:
            supplementary_input_fields = []

        must_not_clause = []
        if excluded_ids:
            must_not_clause.append({"ids": {"values": excluded_ids}})

        filter_clause: list = []
        if space_id:
            filter_clause.append({"term": {"space_id.keyword": space_id}})
        if excluded_ids:
            filter_clause.append({"ids": {"values": excluded_ids}})

        embedding_field = f"{input_field}{DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX}"
        return {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"space_id.keyword": space_id}} if space_id else {"match_all": {}},
                                {"exists": {"field": embedding_field}},
                                {"exists": {"field": input_field}},
                                {"exists": {"field": output_field}},
                                *[{"exists": {"field": field}} for field in supplementary_input_fields]
                            ],
                            "must_not": must_not_clause
                        }
                    },
                    "functions": [
                        {
                            "filter": {
                                "bool": {
                                    "must": [
                                        {"exists": {"field": embedding_field}}
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
                "filter": filter_clause,
            },
            "_source": [input_field, output_field, *supplementary_input_fields],
            "size": num_examples
        }

    #
    # Preview search methods
    #
    def preview_search(
            self,
            query: str,
            input_field: str,
            output_field: str,
            num_examples: int,
            added_examples: list[dict[str, Any]],
            removed_ids: list[str],
            supplementary_input_fields: list[str] | None = None,
            space_id: str | None = None,
        ) -> list[dict[str, Any]]:
        """Preview search results with updated/removed examples.

        Args:
            query: Query string
            input_field: Input field name
            output_field: Output field name
            num_examples: Number of examples to return
            added_examples: Examples to add/update
            removed_ids: Example IDs to remove
            supplementary_input_fields: Optional. Additional input fields if your examples set have
            space_id: Optional. Space id for filtering if multiple spaces are used in single index
        Returns:
            Preview search results
        """
        self._preview_search_input = input_field
        self._preview_search_output = output_field

        if supplementary_input_fields is None:
            supplementary_input_fields = []

        self._preview_supplementary_input_fields = supplementary_input_fields

        # Ensure removed_ids is a list
        if not isinstance(removed_ids, list):
            removed_ids = []
        # Get examples from Elasticsearch, excluding removed IDs
        try:
            indexed_examples = self.search(query, input_field, output_field, num_examples, supplementary_input_fields, removed_ids, space_id)
        except errors.APIError:
            # If search fails, use empty list for indexed examples
            indexed_examples = []

        # Score examples using cosine similarity
        indexed_scored_examples = self._compose_scored_examples(query, indexed_examples or [])
        updated_scored_examples = self._compose_scored_examples(query, added_examples or [])
        # Combine and get top results
        combined_results = self._combine_results(indexed_scored_examples, updated_scored_examples)
        top_results = combined_results[:num_examples]
        fields_to_keep = ["_id", input_field, output_field, *supplementary_input_fields, "_score"]

        return [{k: v for k, v in d.items() if k in fields_to_keep} for d in top_results]

    def _compose_scored_examples(self, query: str, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Calculate similarity scores for examples.

        Args:
            query: Query string
            examples: List of examples to score

        Returns:
            Examples with similarity scores
        """
        if not examples:
            return []
        if not self.embedding_model:
            return []
        similarities = []
        query_embedding = self.embedding_model.do_embed(query).embedding
        for example in examples:
            # Skip if example doesn't have required field
            if self._preview_search_input not in example:
                continue
            es_input = example[self._preview_search_input]
            input_embedding = self.embedding_model.do_embed(es_input).embedding
            # Calculate similarity score
            sim_score = self._cosine_similarity(
                np.array([query_embedding]),
                np.array([input_embedding])
            )

            # Create result dictionary
            score_dict = {
                self._preview_search_input: es_input,
                self._preview_search_output: example.get(self._preview_search_output, ""),
                **{k: example[k] for k in self._preview_supplementary_input_fields if k in example},
                "_score": sim_score
            }

            # Add _id and last_updated_time if present
            if "_id" in example:
                score_dict["_id"] = example["_id"]
            if "last_updated_time" in example:
                score_dict["last_updated_time"] = example["last_updated_time"]

            similarities.append(score_dict)
        return similarities

    def _combine_results(
            self,
            es_results: list[dict[str, Any]],
            context_similarities: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
        """Combine and deduplicate results from different sources.

        Args:
            es_results: Results from Elasticsearch
            context_similarities: Results from updated examples

        Returns:
            Combined and sorted unique results
        """
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

    #
    # Reindexing methods
    #
    def _contains_embedding_keys(self, document: dict[str, Any]) -> list[str]:
        """Find keys in a document that contain embeddings.

        Args:
            document: Document to search for embedding keys

        Returns:
            List of keys that contain embeddings
        """
        return [key for key in document if key.endswith(DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX)]


    @ensure_arguments
    def reindex_embeddings(self,size=9999,search_after=None):
        """Update the embeddings model for the examples for the first batch"""
        client = self.es
        search_params = {}
        if size:
            search_params["size"] = size
        if search_after:
            search_params["search_after"] = search_after
        response = handle_elasticsearch_error(
            client.search,
            index= self.index_name,
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
            keys = self._contains_embedding_keys(doc)
            updated_embeddings = {}
            for key in keys:
                updated_embeddings[key] = self.embedding_model.do_embed(doc[key.removesuffix(DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX)]).embedding
            doc.update(updated_embeddings)
            data = {
                "_op_type": "update",
                "_index": self.index_name,
                "_id": res["_id"],
            }
            data.update({"_source": {"doc": doc}})
            action_bulk.append(data)
        bulk(client, actions=action_bulk, chunk_size=100)

        logger.debug(response)
        return {
            "status": "success",
            "search_after": search_after
        }