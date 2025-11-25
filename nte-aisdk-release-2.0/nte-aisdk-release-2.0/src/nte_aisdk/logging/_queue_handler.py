# Fork from https://github.com/Mulanir/python-elasticsearch-logging/blob/main/src/elasticsearch_logging_handler/_queue_handler.py
from logging import LogRecord
from logging.handlers import QueueHandler


class ObjectQueueHandler(QueueHandler):
    """QueueHandler that preserves message as an object in the separate field - msg_object."""

    def prepare(self, record: LogRecord) -> LogRecord:
        """Create msg_object as raw message before it will be formatted as str."""
        record_dict = record.__dict__
        props_data = record_dict.get("props", {})
        data = props_data.get("data")

        if data is not None:
            record.__setattr__("data", data)

        return super().prepare(record)