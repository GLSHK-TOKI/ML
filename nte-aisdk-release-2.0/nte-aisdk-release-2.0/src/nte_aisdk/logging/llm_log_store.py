from ..exception import ensure_arguments, handle_elasticsearch_operation
from ..vector_store import VectorStore


class LLMLogStore(VectorStore):
    """LLMLogStore is a class that extends VectorStore for performing actions on log index.

    Args:
        index_name: The name of the index to store the logs.
    """

    index_name: str

    @ensure_arguments
    def __init__(
            self,
            host: str,
            basic_auth: str | tuple[str, str],
            index_name: str,
        ):
        super().__init__(host, basic_auth)
        self.index_name = index_name

    @ensure_arguments
    def prune_by_date(self, datetime: str):
        """Prune logs by a specific datetime. Logs older than the specified datetime will be deleted.

        Args:
            datetime: The date to prune logs by.
        """
        response = handle_elasticsearch_operation(
            self.es.delete_by_query,
            index=self.index_name,
            body={
                "query": {
                    "range": {
                        "@timestamp": {
                            "lt": datetime,
                            "format": "strict_date_optional_time"
                        }
                    }
                }
            }
        )
        return {
            "deleted": response["deleted"],
            "failures": response["failures"]
        }