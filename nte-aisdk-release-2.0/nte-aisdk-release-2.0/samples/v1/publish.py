from nte_aisdk.azure_openai import AzureOpenAIInstanceConfig
from nte_aisdk.dynamic_few_shot import DynamicFewShotExampleStore

INSTANCE_CONFIGS = [
    AzureOpenAIInstanceConfig("<your_azure_endpoint_1>", "<your_api_key_1"),
    AzureOpenAIInstanceConfig("<your_azure_endpoint_2>", "<your_api_key_2"),
    AzureOpenAIInstanceConfig("<your_azure_endpoint_3>", "<your_api_key_3")
]
example_store = DynamicFewShotExampleStore(
    host="<your_elastic_host>",
    basic_auth=("<your_username>", "<your_password>"),
    instance_configs=INSTANCE_CONFIGS,
    index_prefix="<idx--custom--pvt--app--env--indexprefix>",
)

response = example_store.staging.publish(size=3)
while True:
    if (response["search_after"]):
        response = example_store.staging.publish(search_after=response["search_after"],size=3)
    else:
        break
print(response)