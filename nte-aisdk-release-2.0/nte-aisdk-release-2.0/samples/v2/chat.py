import os

from dotenv import load_dotenv

from nte_aisdk.general_chatbot import GeneralChatModel
from nte_aisdk.providers.azure import AzureProvider

load_dotenv()

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
result = chat_model.chat(
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
)
print(result)

azure_deepseek_provider = AzureProvider(
    azure_endpoint=os.getenv("AZURE_ENDPOINT_DEEPSEEK"),
    api_key=os.getenv("AZURE_API_KEY_DEEPSEEK"),
    api_version=os.getenv("AZURE_API_VERSION_DEEPSEEK"),
)
chat_model = GeneralChatModel(
    language_model=azure_deepseek_provider.create_language_model(
        azure_deployment="DeepSeek-R1-2",
        model_name="DeepSeek-R1-0528",
    ),
)
result = chat_model.chat(
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
)
print(result)

from nte_aisdk.providers.vertex import VertexProvider

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
result = chat_model.chat(
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
)
print(result)