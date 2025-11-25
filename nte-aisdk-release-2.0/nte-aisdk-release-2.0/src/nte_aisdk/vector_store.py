from elasticsearch import Elasticsearch


class VectorStore:
    """VectorStore is a base class to extend the functionalitiy for ElasticSearch client.

    Args:
        host: The host of the ElasticSearch instance.
        basic_auth: The tuple of username and password of the ElasticSearch user.

    """

    def __init__(
            self,
            host: str,
            basic_auth: str | tuple[str, str],
        ):
        self.es = Elasticsearch(
            host,
            basic_auth=basic_auth
        )