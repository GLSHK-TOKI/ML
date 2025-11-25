# Fork from https://github.com/Mulanir/python-elasticsearch-logging/blob/main/src/elasticsearch_logging_handler/_handlers.py
from logging import NOTSET, Handler, LogRecord
from logging.handlers import QueueListener
from queue import Queue

from elasticsearch import Elasticsearch

from ..exception import ensure_arguments
from ._queue_handler import ObjectQueueHandler
from ._sending_handler import ElasticSendingHandler


class ElasticsearchHandler(Handler):
    @ensure_arguments
    def __init__(
        self,
        host: str,
        basic_auth: str | tuple[str, str],
        index_name: str,
        level=NOTSET,
        flush_period: float = 1,
        batch_size: int = 1000,
        timezone: str | None = None,
    ) -> None:
        super().__init__(level)

        es_client = Elasticsearch(host, basic_auth=basic_auth)
        if es_client is None:
            # Disable emiting LogRecord to queue
            self.emit = lambda *a, **kw: None  # noqa: ARG005

            return
        self._es_client = es_client

        _queue: Queue[dict] = Queue(maxsize=100000)

        # Object for writing logs to the queue.
        self._queue_handler = ObjectQueueHandler(_queue)

        # Object for reading logs from the queue.
        _elastic_listener = ElasticSendingHandler(
            level,
            es_client,
            index_name,
            flush_period=flush_period,
            batch_size=batch_size,
            timezone=timezone
        )
        self._queue_listener = QueueListener(_queue, _elastic_listener)
        self._queue_listener.start()

    def emit(self, record: LogRecord) -> None:
        """Write logs to the queue."""
        self._queue_handler.emit(record)

    def close(self) -> None:
        if hasattr(self, "_queue_listener"):
            self._queue_listener.stop()

        if hasattr(self, "_es_client"):
            self._es_client.close()

        return super().close()