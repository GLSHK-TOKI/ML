from nte_aisdk.azure_openai import AzureOpenAIInstanceConfig
from nte_aisdk.dynamic_few_shot import DynamicFewShotExampleStore

INSTANCES_CONFIGS = [
    AzureOpenAIInstanceConfig("<your_azure_endpoint_1>", "<your_api_key_1"),
    AzureOpenAIInstanceConfig("<your_azure_endpoint_2>", "<your_api_key_2"),
    AzureOpenAIInstanceConfig("<your_azure_endpoint_3>", "<your_api_key_3")
]
example_store = DynamicFewShotExampleStore(
    host="<your_elastic_host>",
    basic_auth=("<your_username>", "<your_password>"),
    instance_configs=INSTANCES_CONFIGS,
    index_prefix="examplestore",
)

# #--------------------------- Single Example ---------------------------#


#Getting Example by field
example_store.staging.examples.get_by_field("question","What is the capital of New Zealand?")

#Creating Single Example  (example: dict[str, Any], input_fields(embedding_fields) : list[str])
example_store.staging.examples.create(
    {
    "question": "What is the largest planet in our solar system?",
    "answer": "Jupiter"
    }
    ,
    ["question"]
)

#Updating Single Example  (document_id:str, example: dict[str, Any], input_fields(embedding_fields) : list[str])
example_store.staging.examples.update(
    "4c091c223e16ba31726b97b0b4a5011d56748237dd9652a3cce4cba5152f7a94",
    {
    "question": "Who wrote Romeo and Juliet?",
    "answer": "William Shakespeare"
    }
    ,
    ["question"]
)

# Deleting Single Example  (document_id:str)
example_store.staging.examples.delete(
    "dsuNG5EB78zUTDO__ZwX"
)

# #--------------------------- More Examples ---------------------------#

#Creating Many Examples  (examples: list[dict[str, Any]], input_fields(embedding_fields) : list[str])
example_store.staging.examples.create_many(
    [
      {
        "question": "What is the capital of New Zealand?",
        "answer": "Wellington",
      },
      {
        "question": "What is the capital of Australia?",
        "answer": "Canberra",
      }
    ],
    ["question"]
)


#Updating Many Examples  (doc_ids: list[str], examples: list[dict[str, Any]], input_fields(embedding_fields) : list[str])
example_store.staging.examples.update_many(
    ["9c4298ccfad5acaae66fe6e929a6c949f4270cd8edc4c3b505a63727df47f26a","91ec1e8e2bc981c8b2c81e6c6ad075a24a182d91d0f33cd5fab00dc6102aaee9"],
    [
      {
        "question": "What is the capital of Japan?",
        "answer": "Tokyo"
      },
      {
        "question": "What is the speed of light in a vacuum?",
        "answer": "299,792,458 meters per second"
      }
    ],
    ["question"]
)


#Deleting Many Examples  (doc_ids: list[str])
example_store.staging.examples.delete_many(
    ["1cuQG5EB78zUTDO_veTm","bad9ea434b83cd45ad33ad73060a69c6cb4e6add9c9dc75e69258779d0ed2eab"]
)