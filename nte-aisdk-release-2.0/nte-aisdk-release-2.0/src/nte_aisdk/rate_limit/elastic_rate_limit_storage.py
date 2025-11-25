import logging

from elasticsearch import Elasticsearch

from nte_aisdk import errors
from nte_aisdk.utils import handle_elasticsearch_error

from .rate_limit_storage import RateLimitStorage

logger = logging.getLogger(__name__)


class ElasticRateLimitStorage(RateLimitStorage):
    def __init__(
            self,
            host: str,
            basic_auth: str | tuple[str, str],
            index_name: str
        ):
        self.es = Elasticsearch(
            host,
            basic_auth=basic_auth
        )
        self.index_name = index_name

    def get(self, key: str, user_id: str):
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"key.keyword": key}},
                        {"term": {"id.keyword": user_id}}
                    ]
                }
            }
        }

        response = handle_elasticsearch_error(
            self.es.search,
            index=self.index_name,
            body=query
        )
        hits = response["hits"]["hits"]
        if len(hits) > 1:
            msg = "More than One Rate limiting Records have been found"
            raise errors.SDKError(msg)
        if len(hits) == 0:
            return None

        # Return result in flat format
        hit = hits[0]
        return {
            "_id": hit["_id"],
            "key": hit["_source"]["key"],
            "id": hit["_source"]["id"],
            "timestamp": hit["_source"]["timestamp"],
            "tokens": hit["_source"]["tokens"],
        }

    def put(
        self,
        key: str,
        user_id: str,
        tokens: int,
        timestamp: float,
    ):
        embeddings_doc = {
            "key": key,
            "id": user_id,
            "timestamp": timestamp,
            "tokens": tokens
        }
        response = handle_elasticsearch_error(
            self.es.index,
            index=self.index_name,
            body=embeddings_doc
        )
        logger.debug(response)
        return {
            "status": "success",
            "id": response["_id"]
        }

    def update(
        self,
        doc_id: str,
        tokens: int,
        timestamp: float | None = None
    ):
        update_body: dict[str, dict[str, int | float]] = {
            "doc": {
                "tokens": tokens
            }
        }
        if timestamp is not None:
            update_body["doc"]["timestamp"] = timestamp

        response = handle_elasticsearch_error(
            self.es.update,
            index=self.index_name,
            id=doc_id,
            body=update_body
        )
        logger.debug(response)
        return {
            "status": "success",
            "id": response["_id"]
        }