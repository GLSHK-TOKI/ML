from nte_aisdk.azure_openai import AzureOpenAIInstanceConfig
from nte_aisdk.dynamic_few_shot import DynamicFewShotExampleStore

INSTANCES_CONFIGS = [
    AzureOpenAIInstanceConfig("<your_azure_endpoint_1>", "<your_api_key_1>"),
]
example_store = DynamicFewShotExampleStore(
    host="<your_elastic_host>",
    basic_auth=("<your_username>", "<your_password>"),
    instance_configs=INSTANCES_CONFIGS,
    index_prefix="examplestore",
)
examples = example_store.staging.examples.get_all()
print(examples)