import logging
from collections.abc import Iterable, Sequence
from typing import Any

from nte_aisdk import types
from nte_aisdk.multimodal_embedding_model import MultimodalEmbeddingModel
from nte_aisdk.text_embedding_model import TextEmbeddingModel
from nte_aisdk.utils import ensure_arguments, handle_elasticsearch_error

from ..vector_store import VectorStore
from ._constants import (
    KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
)

logger = logging.getLogger(__name__)


class KnowledgeBaseStore(VectorStore):
    """KnowledgeBaseStore is a class that extends VectorStore for knowledge base chatbot usecase.

    Args:
        embedding_model: The embedding model to use for the knowledge base store.
        index_prefix: The prefix of the ElasticSearch index for this knowledge base store.
                Two indices should be created with the following names:
                    - {index_prefix}docs
                    - {index_prefix}states
    """

    _embedding_model: TextEmbeddingModel
    index_prefix: str

    @ensure_arguments
    def __init__(
        self,
        host: str,
        basic_auth: str | tuple[str, str],
        embedding_model: TextEmbeddingModel,
        index_prefix: str,
        multimodal_embedding_model: MultimodalEmbeddingModel | None = None,
    ):
        super().__init__(host, basic_auth)
        self._embedding_model = embedding_model
        self.index_prefix = index_prefix
        self._multimodal_embedding_model = multimodal_embedding_model

    @ensure_arguments
    def search(
        self,
        query: str,
        collection_id: str,
        size: int = 30,
        threshold: float = 1.5
    ) -> list[dict[str, Any]]:
        """Search the contexts from the knowledge base store.

        Args:
            query: The query string to search the contexts.
            collection_id: The id of the collection to search the contexts from.
            size: The number of contexts chunks to return. Default is 30.
            threshold: The threshold of min similarity score of retrieval. Default is 1.5.

        Returns:
            A list of search results

        """
        if not query:
            return []

        embeddings = self._embedding_model.do_embed(query).embedding

        search_query = {
            "script_score": {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"parentId.keyword": collection_id}}
                        ],
                        "filter": [
                            {"exists": {"field": "embeddings"}}
                        ]
                    }
                },
                "min_score": threshold,
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1",
                    "params": {"query_vector": embeddings},
                },
            }
        }

        result = handle_elasticsearch_error(
            self.es.search,
            index=self.index_prefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
            size=size,
            query=search_query,
            track_total_hits=True
        )

        keys = ["id", "parentId", "n_token", "content", "collection", "last_updated_time", "meta"]
        es_sources = result["hits"]["hits"]
        return [
            {**self._trim_dict_by_keys(es_example["_source"], keys), "_id": es_example["_id"], "_score": es_example["_score"]}
            for es_example in es_sources
        ]

    @ensure_arguments
    def search_multimodal(
        self,
        query: str,
        query_image: str | None,
        collection_id: str,
        *,
        size_image: int = 20,
        threshold: float = 1.5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search for relevant image content using vector similarity.

        Args:
            query: The user's text query.
            query_image: Optional base64 encoded image string for image-based search.
            collection_id: The id of the collection to search within.
            size_image: Number of top image results to return.
            threshold: The minimum similarity score threshold.

        Returns:
            dict: Contains only "image_results" list with image search results.
                  Text results should be fetched separately using the search() method.
        """
        results: dict[str, list[dict[str, Any]]] = {"image_results": []}

        # Search Image Content
        image_embedding = self._get_multimodal_embedding(query, query_image)
        if image_embedding is not None:
            results["image_results"] = self._search_by_type(
                collection_id, image_embedding, "image", size_image, threshold, "image_embedding"
            )

        return results

    def _get_multimodal_embedding(self, query: str, query_image: str | None) -> list[float] | None:
        """Get embedding for multimodal search using either text or image.

        This method creates embeddings with the following priority:
        - If query_image is provided: Use image for embedding (preferred for image search)
        - If no image is provided: Fallback to text query for embedding

        Args:
            query: Text query to use as fallback if no image provided
            query_image: Optional base64 encoded image string (takes priority if provided)

        Returns:
            Embedding vector or None if not available
        """
        if self._multimodal_embedding_model is None:
            logger.warning("Multimodal embedding model is not configured")
            return None

        if not hasattr(self._multimodal_embedding_model, "do_embed_multimodal"):
            logger.warning("Multimodal embedding model does not support do_embed_multimodal")
            return None

        # Create content using proper TextPart or FilePart objects
        # Priority: Use image if provided, otherwise fallback to text query
        content: list[types.TextPart | types.FilePart] = []
        if query_image:
            # Create FilePart for image
            file_part = types.FilePart(
                file=types.FileWithBytes(
                    name="query_image.jpg",
                    mime_type="image/jpeg",
                    bytes=query_image
                )
            )
            content=[file_part]
        elif query:
            # Create TextPart for text query as fallback
            text_part = types.TextPart(text=query)
            content = [text_part]
        else:
            logger.warning("No valid content (text or image) provided for multimodal embedding")
            return None
        response = self._multimodal_embedding_model.do_embed_multimodal(content=content)
        return response.embedding

    def _search_by_type(  # noqa: PLR0913
        self,
        collection_id: str,
        embedding: list[float],
        chunk_type: str,
        size: int,
        threshold: float,
        embedding_field: str
    ) -> list[dict[str, Any]]:
        """Perform search by chunk type."""
        search_query = {
            "script_score": {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"parentId.keyword": collection_id}},
                            {"term": {"chunk_type": chunk_type}}
                        ]
                    }
                },
                "script": {
                    "source": f"cosineSimilarity(params.query_vector, '{embedding_field}') + 1",
                    "params": {"query_vector": embedding}
                },
                "min_score": threshold
            }
        }

        result = handle_elasticsearch_error(
            self.es.search,
            index=self.index_prefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
            size=size,
            query=search_query,
            track_total_hits=True
        )

        search_results = []
        for es_hit in result["hits"]["hits"]:

            keys = ["id", "parentId", "n_token", "content", "collection", "last_updated_time", "meta"]

            result_data = self._trim_dict_by_keys(
                es_hit["_source"],
                keys
            )
            result_data["_id"] = es_hit["_id"]
            result_data["_score"] = es_hit["_score"]
            search_results.append(result_data)

        return search_results

    @staticmethod
    def _trim_dict_by_keys(dictionary: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
        return {k: dictionary[k] for k in keys if k in dictionary}
