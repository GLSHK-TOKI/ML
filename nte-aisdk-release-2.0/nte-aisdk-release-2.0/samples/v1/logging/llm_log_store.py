import datetime

from nte_aisdk.logging import LLMLogStore

log_store = LLMLogStore(
    host="<your-host>",
    basic_auth=("<your-username>", "<your-password>"),
    index_name="<your-index-for-storing-logs>",
)

yesterday = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(days=1)
response = log_store.prune_by_date(yesterday.isoformat())
print(response)
