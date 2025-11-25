from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._example_store_env import DynamicFewShotExampleStoreEnv
    from .example_store import DynamicFewShotExampleStore

import logging
from http import HTTPStatus

from nte_aisdk.exception import check_example_fields
from nte_aisdk.utils import ensure_arguments, handle_elasticsearch_error

from ..._constants import DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX
from ._example_store_examples import DynamicFewShotExampleStoreExamples

logger = logging.getLogger(__name__)

class DynamicFewShotExampleStoreExamplesStaging(DynamicFewShotExampleStoreExamples):
    """DynamicFewShotExampleStoreExamplesStaging is for performing crud examples on staging index"""

    def __init__(
            self,
            env: DynamicFewShotExampleStoreEnv,
            example_store: DynamicFewShotExampleStore
        ):
        super().__init__(env, example_store)

    def _generate_example_details(self, example = None, input_fields = None):
        formatted_datetime = datetime.now(UTC).isoformat()
        example_doc = {}
        for input_field in input_fields:
            embedding_field_name = f"{input_field}{DYNAMIC_FEW_SHOT_EMBEDDING_FIELD_SUFFIX}"
            embedding_response = self._env.embedding_model.do_embed(example[input_field])
            example_doc[embedding_field_name] = embedding_response.embedding
        example_doc.update(example)
        example_doc.update(
            {"last_updated_time": formatted_datetime}
        )
        return (example_doc)

    @ensure_arguments
    @check_example_fields
    def create(self, example: dict[str, Any], input_fields : list[str]):
        elastic_index = self._env.get_index_name()
        embeddings_doc = self._generate_example_details(example, input_fields)
        response = handle_elasticsearch_error(
            self._example_store.es.index,
            index=elastic_index,
            body = embeddings_doc
        )
        logger.debug(response)
        return {
            "status": "success",
            "id": response["_id"]
        }

    @ensure_arguments
    @check_example_fields
    def create_many(self, examples: list[dict[str, Any]], input_fields: list[str], batch_size: int = 500):
        elastic_index = self._env.get_index_name()
        created_ids = []

        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            action_bulk = []
            for example in batch:
                embeddings_doc = self._generate_example_details(example, input_fields)
                action = {"index": {"_index": elastic_index}}
                action_bulk.append(action)
                action_bulk.append(embeddings_doc)

            response = handle_elasticsearch_error(
                self._example_store.es.bulk,
                body=action_bulk)
            batch_ids = [doc["index"]["_id"] for doc in response["items"] if doc["index"]["status"] == HTTPStatus.CREATED]
            created_ids.extend(batch_ids)
            logger.debug(f"Batch {i//batch_size + 1} created IDs: {batch_ids}")

        return {
            "status": "success",
            "ids": created_ids,
            "total": len(created_ids)
        }

    @ensure_arguments
    @check_example_fields
    def update(self, doc_id: str, example: dict[str, Any], input_fields:list[str]):
        elastic_index = self._env.get_index_name()
        doc = self._generate_example_details(example, input_fields)
        response = handle_elasticsearch_error(
            self._example_store.es.update,
            index=elastic_index,
            id=doc_id,
            body={"doc": doc}
        )
        logger.debug(response)
        return {
            "status": "success",
            "id": response["_id"],
        }

    @ensure_arguments
    @check_example_fields
    def update_many(self, doc_ids: list[str], examples: list[dict[str, Any]], input_fields : list[str]):
        action_bulk = []
        elastic_index = self._env.get_index_name()
        for doc_id, example in zip(doc_ids, examples, strict=False):
            embeddings_doc = self._generate_example_details(example, input_fields)
            action = {"update": {"_index": elastic_index, "_id": doc_id}}
            action_bulk.append(action)
            action_bulk.append( { "doc" : embeddings_doc } )
        response = handle_elasticsearch_error(
            self._example_store.es.bulk,
            body=action_bulk)
        updated_ids = [doc["update"]["_id"] for doc in response["items"] if doc["update"]["status"] == HTTPStatus.OK]
        logger.debug(updated_ids)
        return {
            "status": "success",
            "ids": updated_ids,
            "total": len(updated_ids)
        }

    @ensure_arguments
    def delete(self, doc_id: str):
        elastic_index = self._env.get_index_name()
        response = handle_elasticsearch_error(
            self._example_store.es.delete,
            index=elastic_index,
            id=doc_id)
        logger.debug(response)
        return {
            "status": "success",
            "id": response["_id"]
        }

    @ensure_arguments
    def delete_many(self, doc_ids: list[str]):
        action_bulk = []
        elastic_index = self._env.get_index_name()
        for doc_id in doc_ids:
            action = {"delete": {"_index": elastic_index, "_id": doc_id}}
            action_bulk.append(action)
        response = handle_elasticsearch_error(
            self._example_store.es.bulk,
            body=action_bulk)
        deleted_ids = [doc["delete"]["_id"] for doc in response["items"] if doc["delete"]["status"] == HTTPStatus.OK]
        logger.debug(deleted_ids)
        return {
            "status": "success",
            "ids": deleted_ids,
            "total": len(deleted_ids)
        }