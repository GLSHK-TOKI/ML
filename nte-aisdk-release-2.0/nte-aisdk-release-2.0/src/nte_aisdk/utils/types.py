from typing import Any

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from nte_aisdk import types


def normalize_message(
    message: types.MessageOrDict,
) -> types.Message:
    """Normalize a single message to Pydantic types.Message object."""
    if isinstance(message, dict):
        return types.Message.model_validate(message)
    return message

def normalize_messages(messages: list[types.MessageOrDict] | list[types.Message] | list[types.MessageDict]) -> list[types.Message]:
    """Normalize messages to Pydantic types.Message objects."""
    return [
        normalize_message(message)
        for message in messages
    ]

def convert_to_langchain_messages(messages: list[types.MessageOrDict] | list[types.Message] | list[types.MessageDict]) -> list[BaseMessage]:
    """Converts a list of Message to LangChain BaseMessage format.

    For example, Message format with TextPart is:
    {
        "role": "user",
        "parts": [
            {
                "kind": "text",
                "text": "What is the capital of France?"
            }
        ]
    }

    Message format with FilePart is:
    {
        "role": "user",
        "parts": [
            {
                "kind": "file",
                "file": {
                    "name": "output.png",
                    "mime_type": "image/png",
                    "bytes": "ASEDGhw0KGgoAAAANSUhEUgAA..."
                }
            }
        ]
    }

    and the LangChain BaseMessage format is:
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is the capital of France?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,ASEDGhw0KGgoAAAANSUhEUgAA..."
                }
            }
        ]
    }

    Args:
        messages (list[types.MessageOrDict]): The input messages to convert.

    Returns:
        list[BaseMessage]: A list of LangChain BaseMessage objects.
    """
    p_messages = normalize_messages(messages)

    langchain_messages: list[BaseMessage] = []
    for message in p_messages:
        role = message.role
        parts = message.parts

        content_parts: list[str | dict[Any, Any]] = []
        for part in parts:
            kind = part.kind
            if isinstance(part, types.TextPart):
                text = part.text
                content_parts.append({"type": "text", "text": text})
            elif isinstance(part, types.FilePart):
                file_info = part.file
                # Check if this is an image file for Azure OpenAI vision support
                if file_info.mime_type and file_info.mime_type.startswith("image/"):
                    # Create data URI format for images (Azure OpenAI format)
                    data_uri = f"data:{file_info.mime_type};base64,{file_info.bytes}"
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    })
                else:
                    # For non-image files, use the original file format
                    content_parts.append({
                        "type": "file",
                        "file": {
                            "name": file_info.name,
                            "mimeType": file_info.mime_type,
                            "bytes": file_info.bytes
                        }
                    })
            else:
                msg = f"Part kind '{kind}' is not supported yet."
                raise NotImplementedError(msg)

        if role == "user":
            langchain_messages.append(HumanMessage(content=content_parts))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content_parts))
        elif role == "system":
            # System messages in LangChain typically have string content.
            # We'll concatenate all text parts.
            # Concatenate all text parts for system messages.
            full_text = "\n".join(
                part["text"] for part in content_parts if isinstance(part, dict) and part.get("type") == "text"
            )
            langchain_messages.append(SystemMessage(content=full_text))
        else:
            msg = f"Unknown role: {role} in message."
            raise ValueError(msg)

    return langchain_messages