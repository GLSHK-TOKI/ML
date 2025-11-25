import os

from dotenv import load_dotenv

from nte_aisdk.general_chatbot import GeneralChatModel
from nte_aisdk.providers.azure import AzureProvider
from nte_aisdk.providers.vertex import VertexProvider

load_dotenv()

azure_provider = AzureProvider(
    azure_endpoint=os.getenv("AZURE_ENDPOINT_DEEPSEEK"),
    api_key=os.getenv("AZURE_API_KEY_DEEPSEEK"),
    api_version=os.getenv("AZURE_API_VERSION_DEEPSEEK"),
)
chat_model = GeneralChatModel(
    language_model=azure_provider.create_language_model(
        azure_deployment="DeepSeek-R1-2",
        model_name="DeepSeek-R1-0528",
    ),
)
for chunk in chat_model.chat_stream(
    messages=[
        {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "What is the capital of France?"
                }
            ]
        }
    ]
):
    print(chunk)


azure_provider = AzureProvider(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
)
chat_model = GeneralChatModel(
    language_model=azure_provider.create_language_model(
        azure_deployment="gpt-4.1-mini",
        model_name="gpt-4.1-mini",
    ),
)
for chunk in chat_model.chat_stream(
    messages=[
        {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "What is the capital of France?"
                }
            ]
        }
    ]
):
    print(chunk)

vertex_provider = VertexProvider(
    location=os.environ["VERTEX_LOCATION"],
    project=os.environ["VERTEX_PROJECT"],
    credentials_base64=os.environ["VERTEX_CREDENTIALS_BASE64"],
)
chat_model = GeneralChatModel(
    language_model=vertex_provider.create_language_model(
        model_name="gemini-2.5-flash",
    ),
)
for chunk in chat_model.chat_stream(
    messages=[
        {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Tell me a very long story?",
                },
            ],
        }
    ]
):
    print(chunk)
