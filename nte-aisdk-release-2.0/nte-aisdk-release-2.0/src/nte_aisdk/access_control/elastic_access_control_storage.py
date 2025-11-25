import logging

from elasticsearch import Elasticsearch

from .access_control_storage import AccessControlStorage

logger = logging.getLogger(__name__)


class ElasticAccessControlStorage(AccessControlStorage):
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

    def list_collections(self) -> list[dict]:
        """List all the collections with properties in the Sharepoint folder states index in elastic search.

        Returns:
            A list containing the unique collection names and and itemIds.

        """
        response = self.es.search(
            index=self.index_name,
            body={
                "size": 0,
                "aggs": {
                    "unique_values": {
                        "terms": {
                            "field": "parentId.keyword",
                            "size": 1000
                        },
                        "aggs": {
                            "hits": {
                                "top_hits": {
                                    "_source": ["collection"],
                                    "size": 1
                                }
                            }
                        }
                    }
                }
            }
        )
        buckets = response["aggregations"]["unique_values"]["buckets"]
        return [
            {
                "name": bucket["hits"]["hits"]["hits"][0]["_source"]["collection"],
                "driveItemId": bucket["key"]
            }
            for bucket in buckets
        ]