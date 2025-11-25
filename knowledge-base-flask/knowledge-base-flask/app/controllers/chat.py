from app.clients.ai import (
    ai,
)


async def get_granted_collections():
    return await ai.access_control.get_granted_collections()

def chat(messages, collection_id):
    return ai.model.chat(
        messages=messages,
        collection_id=collection_id,
    )
