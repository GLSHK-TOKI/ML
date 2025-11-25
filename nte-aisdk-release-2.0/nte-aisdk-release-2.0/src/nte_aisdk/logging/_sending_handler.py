# Fork from https://github.com/Mulanir/python-elasticsearch-logging/blob/main/src/elasticsearch_logging_handler/_queue_handler.py

from __future__ import annotations

import logging
import threading
from datetime import datetime
from logging import Handler
from typing import TYPE_CHECKING

import elasticsearch.helpers as es_helpers
import pytz
from elasticsearch import AuthenticationException, AuthorizationException, ConnectionError, Elasticsearch

if TYPE_CHECKING:
    from .llm_logger import LLMLogRecord

logger = logging.getLogger(__name__)

class ElasticSendingHandler(Handler):
    def __init__(
            self,
            level,
            es_client: Elasticsearch,
            index: str,
            flush_period: float = 1,
            batch_size: int = 1000,
            timezone: str | None = None
        ) -> None:
        super().__init__(level=level)

        self._es_client = es_client
        self._index = index

        self._flush_period = flush_period
        self._batch_size = batch_size
        self._timezone = timezone

        self.__message_buffer = []
        self.__buffer_lock = threading.Lock()

        self.__timer: threading.Timer = None
        self.__schedule_flush()

    def __schedule_flush(self):
        """Start timer that one-time flushes message buffer."""
        if self.__timer is None:
            self.__timer = threading.Timer(self._flush_period, self.flush)
            self.__timer.setDaemon(True)
            self.__timer.start()

    def flush(self):
        """Send all messages from buffer to Elasticsearch."""
        if self.__timer is not None and self.__timer.is_alive():
            self.__timer.cancel()

        self.__timer = None

        if self.__message_buffer:
            try:
                with self.__buffer_lock:
                    actions, self.__message_buffer = self.__message_buffer, []
                es_helpers.bulk(self._es_client, actions, stats_only=True)
            except ConnectionError:
                logging.exception("Unable to connect to Elasticsearch LLM log store")
            except AuthenticationException:
                logging.exception("Authentication failed while sending log to Elasticsearch LLM log store")
            except AuthorizationException:
                logging.exception("Authorization failed while sending log to Elasticsearch LLM log store")
            except Exception:
                logging.exception("Unexpected error while sending log to Elasticsearch LLM log store")

    def emit(self, record: LLMLogRecord): # type: ignore[override]
        """Add log message to the buffer.
        If the buffer is filled up, immedeately flush it.
        """
        action = self.__prepare_action(record)

        with self.__buffer_lock:
            self.__message_buffer.append(action)

        if len(self.__message_buffer) >= self._batch_size:
            self.flush()
        else:
            self.__schedule_flush()

    def __prepare_action(self, record: LLMLogRecord):
        timestamp_dt: datetime = datetime.fromtimestamp(record.created, tz=pytz.utc)

        if self._timezone:
            tz_info = pytz.timezone(self._timezone)
            timestamp_dt = timestamp_dt.astimezone(tz_info)

        timestamp_iso = timestamp_dt.isoformat()

        return {
            "_index": self._index,
            "_op_type": "index",
            "@timestamp": timestamp_iso,
            "level": record.levelname,
            **record.data
        }


    def close(self):
        self.flush()

        return super().close()