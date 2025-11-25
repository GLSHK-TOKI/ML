import os

from dotenv import load_dotenv

from nte_aisdk.knowledge_base import KnowledgeBaseChatModel, KnowledgeBaseStore
from nte_aisdk.providers.azure import AzureProvider

load_dotenv()

azure_provider = AzureProvider(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

store = KnowledgeBaseStore(
    host=os.getenv("ELASTICSEARCH_HOST"),
    basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD")),
    embedding_model=azure_provider.create_embedding_model(
        azure_deployment="text-embedding-3-small",
        model_name="text-embedding-3-small",
    ),
    index_prefix=os.getenv("KNOWLEDGE_BASE_STORE_INDEX_PREFIX")
)

chat_model = KnowledgeBaseChatModel(
    language_model=azure_provider.create_language_model(
        azure_deployment="gpt-4.1-mini",
        model_name="gpt-4.1-mini",
    ),
    store=store,
)
result = chat_model.chat(
    messages=[
        {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "有关台风山陀儿之特别票务安排?"
                }
            ]
        }
    ],
    collection_id="01BT6QKHXHAZHP4UWPIZDLW4FLR5TGPEUD"
)
print(result)

azure_deepseek_provider = AzureProvider(
    azure_endpoint=os.getenv("AZURE_ENDPOINT_DEEPSEEK"),
    api_key=os.getenv("AZURE_API_KEY_DEEPSEEK"),
    api_version=os.getenv("AZURE_API_VERSION_DEEPSEEK"),
)

chat_model = KnowledgeBaseChatModel(
    language_model=azure_deepseek_provider.create_language_model(
        azure_deployment="DeepSeek-R1-2",
        model_name="DeepSeek-R1-0528",
    ),
    store=store,
)
result = chat_model.chat(
    messages=[
        {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "有关台风山陀儿之特别票务安排?"
                }
            ]
        }
    ],
    collection_id="01BT6QKHXHAZHP4UWPIZDLW4FLR5TGPEUD"
)
print(result)

from nte_aisdk.providers.vertex import VertexProvider

vertex_provider = VertexProvider(
    location=os.environ["VERTEX_LOCATION"],
    project=os.environ["VERTEX_PROJECT"],
    credentials_base64=os.environ["VERTEX_CREDENTIALS_BASE64"],
)

chat_model = KnowledgeBaseChatModel(
    language_model=vertex_provider.create_language_model(
        model_name="gemini-2.5-flash",
    ),
    store=store,
)
result = chat_model.chat(
    messages=[
        {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "有关台风山陀儿之特别票务安排?"
                }
            ]
        }
    ],
    collection_id="01BT6QKHXHAZHP4UWPIZDLW4FLR5TGPEUD"
)
print(result)