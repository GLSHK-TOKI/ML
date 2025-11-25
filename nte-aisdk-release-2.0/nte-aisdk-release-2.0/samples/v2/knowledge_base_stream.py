import os

from dotenv import load_dotenv

from nte_aisdk.knowledge_base import KnowledgeBaseChatModel, KnowledgeBaseStore
from nte_aisdk.providers.azure import AzureProvider
from nte_aisdk.providers.vertex import VertexProvider

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
for chunk in chat_model.chat_stream(
    messages=[
        {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Who is eligible for the FOC tickets?"
                }
            ]
        }
    ],
    collection_id="01BT6QKHWHW3BW2Q7ESFH3Z6RDYC3UKFFT"
):
    print(chunk)

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

for chunk in chat_model.chat_stream(
    messages=[
        {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Who is eligible for the FOC tickets?",
                },
            ],
        }
    ],
    collection_id="01BT6QKHWHW3BW2Q7ESFH3Z6RDYC3UKFFT",
):
    print(chunk)
