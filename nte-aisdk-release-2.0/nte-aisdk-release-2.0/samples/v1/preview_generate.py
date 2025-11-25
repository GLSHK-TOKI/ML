from nte_aisdk.azure_openai import AzureOpenAIInstanceConfig, AzureOpenAIModelConfig
from nte_aisdk.dynamic_few_shot import (
    DynamicFewShotDefaultSearchStrategy,
    DynamicFewShotExampleFieldMapping,
    DynamicFewShotExampleStore,
    DynamicFewShotModel,
    DynamicFewShotPromptTemplate,
)

EMBEDDINGS_INSTANCES_CONFIGS = [
    AzureOpenAIInstanceConfig("<your_azure_endpoint_1>/", "<your_api_key_1>")
]
LLM_INSTANCES_CONFIGS = [
    AzureOpenAIInstanceConfig("<your_azure_endpoint_1>/", "<your_api_key_1>")
]

example_store = DynamicFewShotExampleStore(
    host="<your_elastic_host>",
    basic_auth=("<your_username>", "<your_password>"),
    instance_configs=EMBEDDINGS_INSTANCES_CONFIGS,
    index_prefix="examplestore",
)

prompt_template = DynamicFewShotPromptTemplate(
    instruction_text="Instruction: You are an AI categorizer, your task is to tag the comment with the correct label.",
    field_mapping=DynamicFewShotExampleFieldMapping(
        input="text",
        output="result",
    ),
    num_examples=1
)
model = DynamicFewShotModel(
    instance_configs=LLM_INSTANCES_CONFIGS,
    model_config=AzureOpenAIModelConfig(
        azure_deployment="<your_azure_deployment>",
        api_version="<your_apiversion>",
    ),
    prompt_template=prompt_template,
    example_store=example_store,
    search_strategy=DynamicFewShotDefaultSearchStrategy(),
)

answer = model.preview_generate(
    environment="live",
    query="very MCC and even spills drinks on kid, how she pass the assessments?",
    added_examples=[{"text": "i hate this ism, she so bossy and orders people around", "result": "Toxic"}],
    removed_ids=["LOTckZEB78zUTDO_Bd82", "LuTckZEB78zUTDO_Bd82", "GOTckZEB78zUTDO_Bd82"]
)

print(answer)