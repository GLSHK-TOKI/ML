import logging
import os

logger = logging.getLogger("nte_aisdk")

def setup_logger() -> None:
    logging.basicConfig()

    # Set log level based on environment variable
    log_level = os.getenv("NTE_AISDK_LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)