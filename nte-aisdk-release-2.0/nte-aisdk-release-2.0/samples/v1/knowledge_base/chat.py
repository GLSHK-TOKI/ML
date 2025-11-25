from nte_aisdk.azure_openai import AzureOpenAIInstanceConfig, AzureOpenAIModelConfig
from nte_aisdk.knowledge_base import KnowledgeBaseChatModel, KnowledgeBaseStore

EMBEDDINGS_INSTANCES_CONFIGS = [
    AzureOpenAIInstanceConfig("<your_azure_endpoint_1>", "<your_api_key_1>"),
]
LLM_INSTANCES_CONFIGS = [
    AzureOpenAIInstanceConfig("<your_azure_endpoint_1>", "<your_api_key_1>"),
]
EMBEDDINGS_MODEL_CONFIG = AzureOpenAIModelConfig("<your_azure_deployment_1>", "<your_api_version_1>")
LLM_MODEL_CONFIG = AzureOpenAIModelConfig("<your_azure_deployment_1>", "<your_api_version_1>")

store = KnowledgeBaseStore(
    host="<your_host>",
    basic_auth=("<your_user_name>", "<your_password>"),
    instance_configs=EMBEDDINGS_INSTANCES_CONFIGS,
    model_config=EMBEDDINGS_MODEL_CONFIG,
    index_prefix="<your_index_prefix>",
)
model = KnowledgeBaseChatModel(
    instance_configs=LLM_INSTANCES_CONFIGS,
    model_config=LLM_MODEL_CONFIG,
    store=store,
)
answer = model.chat(
    question="<your_question>",
    collection_id="<your_collection_id>",
    history=[],
)
print(answer)