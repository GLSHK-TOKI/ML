from nte_aisdk.azure_openai import AzureOpenAIInstanceConfig
from nte_aisdk.knowledge_base import KnowledgeBaseStore
from nte_aisdk.azure_openai import AzureOpenAIModelConfig

INSTANCES_CONFIGS = [
    AzureOpenAIInstanceConfig("<your_azure_endpoint_1>", "<your_api_key_1>"),
]
example_store = KnowledgeBaseStore(
    host="<your_elastic_host>",
    basic_auth=("<your_username>", "<your_password>"),
    model_config=AzureOpenAIModelConfig(
        azure_deployment="<your_azure_deployment>",
        api_version="<your_api_version>",
    ),
    instance_configs=INSTANCES_CONFIGS,
    index_prefix="ai-sdk-node-testingstates"
)

examples = example_store.list_collections()
print(examples)