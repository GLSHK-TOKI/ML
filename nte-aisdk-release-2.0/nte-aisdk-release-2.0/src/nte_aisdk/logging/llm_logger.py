import sys
from logging import INFO, Logger, LogRecord, StreamHandler

from ..exception import ensure_arguments
from ._elasticsearch_handler import ElasticsearchHandler
from ._probability_filter import ProbabilityFilter


class LLMLogger(Logger):
    """LLMLogger is a logger for logging persistent LLM logs to a index on Elasticsearch.

    Args:
        name: The name of the logger.
        host: The host of the Elasticsearch instance.
        basic_auth: The basic auth credentials for the Elasticsearch instance.
        index_name: The name of the index to store the logs.
        sampling_probability: The probability for performing probability sampling on logging. Default is 1.
    """

    @ensure_arguments
    def __init__(
            self,
            name: str,
            host: str,
            basic_auth: str | tuple[str, str],
            index_name: str,
            sampling_probability: float = 1
        ) -> None:
        super().__init__(name)
        self.setLevel(INFO)
        self.addFilter(ProbabilityFilter(sampling_probability))
        self.addHandler(
            StreamHandler(stream=sys.stdout)
        )
        self.addHandler(
            ElasticsearchHandler(
                host=host,
                basic_auth=basic_auth,
                index_name=index_name,
            )
        )

class LLMLogRecord(LogRecord):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data = kwargs.get("data", {})