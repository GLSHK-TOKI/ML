from __future__ import annotations

import logging
from datetime import UTC, datetime
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from nte_aisdk import errors
from nte_aisdk.exception import check_example_fields
from nte_aisdk.utils import ensure_arguments, handle_elasticsearch_error

from ..._constants import DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX

if TYPE_CHECKING:
    from .simple_example_store import DynamicFewShotSimpleExampleStore

logger = logging.getLogger(__name__)


class DynamicFewShotSimpleExampleStoreExamples:
    """Handle CRUD operations for examples in the DynamicFewShotSimpleExampleStore."""

    def __init__(
            self,
            example_store : DynamicFewShotSimpleExampleStore
    ):
        """Initialize the examples manager.

        Args:
            example_store: The parent example store
        """
        self._example_store = example_store

    def _compose_return_doc(self, example: dict[str, Any]) -> dict[str, Any]:
        """Extract non-embedding fields from an Elasticsearch document."""
        if not example or "_source" not in example:
            return {"_id": example.get("_id", "")}

        doc = {}
        for field in example["_source"]:
            # Embedding field is excluded from the search query
            doc[field] = example["_source"][field]

        doc["_id"] = example["_id"]
        return doc

    def _generate_example_details(
            self,
            example: dict[str, Any],
            input_fields: list[str],
            space_id: str | None = None
        ) -> dict[str, Any]:
        """Generate embeddings and other details for an example."""
        # Create embeddings using the example store's method
        example_doc: dict[str, Any] = {}
        formatted_datetime = datetime.now(UTC).isoformat()  # Store as ISO string, not float

        # Convert all fields in `example` to strings for shared column name index mapping compatibility
        for key, value in example.items():
            if isinstance(value, str):
                example_doc[key] = value
            else:
                example_doc[key] = str(value)

        # Generate embeddings for each input field to the example document
        for input_field in input_fields:
            if input_field not in example:
                msg = f"Input field '{input_field}' missing from example"
                raise errors.InvalidArgumentError(msg)

            embedding_field_name = f"{input_field}{DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX}"

            # Use the embedding model directly
            embeddings = self._example_store.embedding_model.do_embed(example_doc[input_field]).embedding  # type: ignore[attr-defined]

            if not embeddings:
                msg = f"Failed to create embeddings for field: {input_field}"
                raise errors.APIError(msg)

            example_doc[embedding_field_name] = embeddings

        # Add example data and metadata
        example_doc["last_updated_time"] = formatted_datetime
        if space_id is not None:
            example_doc["space_id"] = space_id

        return example_doc

    @ensure_arguments
    def get(self, doc_id: str) -> dict[str, Any]:
        """Get a single example by ID."""
        response = handle_elasticsearch_error(
            self._example_store.es.get,
            index=self._example_store.index_name,  # Use example_store's method
            id=doc_id,
            source_excludes=[f"*{DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX}"]
        )

        logger.debug("Get example response: %s", response)
        return {
            "status": "success",
            "example": self._compose_return_doc(dict(response))
        }

    @ensure_arguments
    def get_many(self, doc_ids: list[str]) -> dict[str, Any]:
        """Get multiple examples by their IDs."""
        if not doc_ids:
            return {"status": "success", "examples": []}

        response = handle_elasticsearch_error(
            self._example_store.es.mget,
            index=self._example_store.index_name,  # Use example_store's method
            body={"ids": doc_ids},
            source_excludes=[f"*{DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX}"]
        )

        logger.debug("Get many examples response: %s", response)
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
            index=self._example_store.index_name,
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
    def get_by_field(self, field_name: str, value: str) -> dict[str, Any]:
        """Get examples matching a specific field value."""
        response = handle_elasticsearch_error(
            self._example_store.es.search,
            index=self._example_store.index_name,  # Use example_store's method
            body={
                "query": {"term": {f"{field_name}.keyword": value}},
                "_source": {
                    "excludes": [f"*{DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX}"]
                }
            }
        )

        logger.debug("Get by field response: %s", response)

        return {
            "status": "success",
            "examples": [self._compose_return_doc(example) for example in response["hits"]["hits"]],
            "total": response["hits"]["total"]["value"] if "total" in response["hits"] else len(response["hits"]["hits"])
        }

    @ensure_arguments
    def get_by_space(
        self,
        space_id: str,
        size: int | None = 9999,
        search_after: list | None = None
    ):
        """Retrieve all examples from the example store.

        Args:
            space_id (str): The ID of the space to filter examples by.
            size (int): The maximum number of examples to retrieve in a call. Default is 9999.
            search_after (list): The value to start the search after. Default is None.

        Returns:
            dict: A dictionary containing the status and retrieved examples.
        """
        search_params: dict[str, Any] = {}
        if size:
            search_params["size"] = size
        if search_after:
            search_params["search_after"] = search_after
        response = handle_elasticsearch_error(
            self._example_store.es.search,
            index=self._example_store.index_name,
            body={
                "query": {
                    "term": {
                        "space_id.keyword": space_id
                    }
                },
                "sort": [{"last_updated_time": {"order": "desc"}}],
                "_source": {
                    "excludes": [f"*{DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX}"]
                }
            },
            **search_params,
        )

        logger.debug(response)

        return {
            "status": "success",
            "examples": [self._compose_return_doc(example) for example in response["hits"]["hits"]],
            "search_after": response["hits"]["hits"][-1]["sort"] if response["hits"]["hits"] else None
        }

    @ensure_arguments
    @check_example_fields
    def create(
        self,
        example: dict[str, Any],
        input_fields: list[str],
        space_id: str | None = None
    ) -> dict[str, Any]:
        """Create a new example."""
        elastic_index = self._example_store.index_name  # Use example_store's method
        embeddings_doc = self._generate_example_details(
            example,
            input_fields,
            space_id
        )

        response = handle_elasticsearch_error(
            self._example_store.es.index,
            index=elastic_index,
            body=embeddings_doc
        )

        logger.debug("Create example response: %s", response)
        return {
            "status": "success",
            "id": response["_id"]
        }

    @ensure_arguments
    @check_example_fields
    def create_many(
        self,
        examples: list[dict[str, Any]],
        input_fields: list[str],
        space_id: str | None = None,
        batch_size: int = 500
    ) -> dict[str, Any]:
        """Create multiple examples at once.

        Args:
            examples: List of example documents to create
            input_fields: Fields to generate embeddings for
            space_id: Optional space id to associate with the examples if the index is space-aware
            batch_size: Number of examples per batch for bulk indexing (default: 500)

        Returns:
            Dictionary with status, created IDs, and total count
        """
        if not examples:
            return {"status": "success", "ids": [], "total": 0}

        elastic_index = self._example_store.index_name
        created_ids = []

        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            action_bulk = []
            for example in batch:
                try:
                    embeddings_doc = self._generate_example_details(
                        example,
                        input_fields,
                        space_id
                    )
                    action = {"index": {"_index": elastic_index}}
                    action_bulk.append(action)
                    action_bulk.append(embeddings_doc)
                except errors.InvalidArgumentError as e:
                    logger.warning("Skipping example due to error: %s", e)
                    continue

            if not action_bulk:
                continue

            response = handle_elasticsearch_error(
                self._example_store.es.bulk,
                body=action_bulk
            )

            batch_ids = [
                doc["index"]["_id"]
                for doc in response["items"]
                if doc.get("index", {}).get("status") == HTTPStatus.CREATED
            ]
            created_ids.extend(batch_ids)
            logger.debug("Batch %d created %d examples with IDs: %s", i // batch_size + 1, len(batch_ids), batch_ids)

        logger.debug("Total created %d examples with IDs: %s", len(created_ids), created_ids)
        return {
            "status": "success",
            "ids": created_ids,
            "total": len(created_ids)
        }

    @ensure_arguments
    @check_example_fields
    def update(self, doc_id: str, example: dict[str, Any], input_fields: list[str]) -> dict[str, Any]:
        """Update an existing example.

        Args:
            doc_id: ID of the document to update
            example: Updated example document
            input_fields: Fields to generate embeddings for

        Returns:
            Dictionary with status and updated document ID
        """
        elastic_index = self._example_store.index_name
        doc = self._generate_example_details(example, input_fields)
        response = handle_elasticsearch_error(
            self._example_store.es.update,
            index=elastic_index,
            id=doc_id,
            body={"doc": doc}
        )

        logger.debug("Update example response: %s", response)
        return {
            "status": "success",
            "id": response["_id"],
        }

    @ensure_arguments
    @check_example_fields
    def update_many(self, doc_ids: list[str], examples: list[dict[str, Any]], input_fields: list[str]) -> dict[str, Any]:
        """Update multiple examples at once.

        Args:
            doc_ids: List of document IDs to update
            examples: List of updated example documents
            input_fields: Fields to generate embeddings for

        Returns:
            Dictionary with status, updated IDs, and total count
        """
        if not doc_ids or not examples:
            return {"status": "success", "ids": [], "total": 0}

        if len(doc_ids) != len(examples):
            msg = "Number of document IDs must match number of examples"
            raise errors.InvalidArgumentError(msg)

        action_bulk = []
        elastic_index = self._example_store.index_name

        for doc_id, example in zip(doc_ids, examples, strict=False):
            try:
                embeddings_doc = self._generate_example_details(example, input_fields)
                action = {"update": {"_index": elastic_index, "_id": doc_id}}
                action_bulk.append(action)
                action_bulk.append({"doc": embeddings_doc})
            except errors.InvalidArgumentError as e:
                logger.warning("Skipping example %s due to error: %s", doc_id, e)
                continue

        if not action_bulk:
            return {"status": "success", "ids": [], "total": 0}

        response = handle_elasticsearch_error(
            self._example_store.es.bulk,
            body=action_bulk
        )

        updated_ids = [
            doc["update"]["_id"]
            for doc in response["items"]
            if doc.get("update", {}).get("status") == HTTPStatus.OK
        ]

        logger.debug("Updated %d examples with IDs: %s", len(updated_ids), updated_ids)
        return {
            "status": "success",
            "ids": updated_ids,
            "total": len(updated_ids)
        }

    @ensure_arguments
    def delete(self, doc_id: str) -> dict[str, Any]:
        """Delete an example by ID.

        Args:
            doc_id: ID of the document to delete

        Returns:
            Dictionary with status and deleted document ID
        """
        elastic_index = self._example_store.index_name

        response = handle_elasticsearch_error(
            self._example_store.es.delete,
            index=elastic_index,
            id=doc_id
        )

        logger.debug("Delete example response: %s", response)
        return {
            "status": "success",
            "id": response["_id"]
        }

    @ensure_arguments
    def delete_many(self, doc_ids: list[str]) -> dict[str, Any]:
        """Delete multiple examples by their IDs.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            Dictionary with status, deleted IDs, and total count
        """
        if not doc_ids:
            return {"status": "success", "ids": [], "total": 0}

        action_bulk = []
        elastic_index = self._example_store.index_name

        for doc_id in doc_ids:
            action = {"delete": {"_index": elastic_index, "_id": doc_id}}
            action_bulk.append(action)

        response = handle_elasticsearch_error(
            self._example_store.es.bulk,
            body=action_bulk
        )

        deleted_ids = [
            doc["delete"]["_id"]
            for doc in response["items"]
            if doc.get("delete", {}).get("status") == HTTPStatus.OK
        ]

        logger.debug("Deleted %d examples with IDs: %s", len(deleted_ids), deleted_ids)
        return {
            "status": "success",
            "ids": deleted_ids,
            "total": len(deleted_ids)
        }

    @ensure_arguments
    def delete_by_space(self, space_id: str):
        """Delete all examples based on the space id."""
        response = handle_elasticsearch_error(
            self._example_store.es.delete_by_query,
            index=self._example_store.index_name,
            body={
                "query": {
                    "term": {
                        "space_id.keyword": space_id
                    }
                }
            }
        )

        deleted_count = response.get("deleted", 0)

        logger.debug("Deleted %d examples with space ID: %s", deleted_count, space_id)

        return {
            "status": "success",
            "total": deleted_count
        }

