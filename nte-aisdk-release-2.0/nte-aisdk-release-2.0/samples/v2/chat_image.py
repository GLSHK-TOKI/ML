import base64
import os
from pathlib import Path

from dotenv import load_dotenv

from nte_aisdk import types
from nte_aisdk.errors import InvalidArgumentError
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
        model_name="gpt-4.1-mini"
    ),
)

file_path = "cathay.jpeg"

# Read and encode file content
try:
    with Path(file_path).open("rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")
except Exception as e:
    msg = f"Failed to read file '{file_path}': {e}"
    raise InvalidArgumentError(msg) from e

result = chat_model.chat(
    messages=[{
        "role": "user",
        "parts": [
            types.TextPart(
                text="Analyze this file:"
            ),
            types.FilePart(
                file=types.FileWithBytes(
                    name=Path(file_path).name,
                    mime_type="image/jpeg",
                    bytes=content
                )
            )
        ]
    }]
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
    messages=[{
        "role": "user",
        "parts": [
            types.TextPart(
                text="Analyze this file:"
            ),
            types.FilePart(
                file=types.FileWithBytes(
                    name=Path(file_path).name,
                    mime_type="image/jpeg",
                    bytes=content
                )
            )
        ]
    }]
)
print(result)