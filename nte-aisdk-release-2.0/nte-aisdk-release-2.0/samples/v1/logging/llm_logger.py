from nte_aisdk.logging import LLMLogger

logger = LLMLogger(
    "ai.chats",
    host="<your-host>",
    basic_auth=("<your-username>", "<your-password>"),
    index_name="<your-index-for-storing-logs>",
)

logger.info({"key": "value"})