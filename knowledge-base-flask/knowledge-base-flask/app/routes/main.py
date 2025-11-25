from flask import jsonify, request

from app.clients.ai import ai
from app.controllers.chat import chat, get_granted_collections

from . import bp


@bp.route("/_health", methods=["GET"])
def health():
    return "OK"

@bp.route("/collections", methods=["GET"])
@ai.access_control.token_required
async def get_collections():
    return await get_granted_collections()

@bp.route("/chat", methods=["POST"])
async def post_chat():
    request_json = request.json
    messages = request_json.get("messages")
    collection_id = request_json.get("collection_id")

    return jsonify(chat(messages, collection_id).model_dump()), 200

@bp.route("/protected/chat", methods=["POST"])
@ai.access_control.token_required
@ai.access_control.enforce_access("collection_id")
async def post_protected_chat():
    request_json = request.json
    messages = request_json.get("messages")
    collection_id = request_json.get("collection_id")

    return jsonify(chat(messages, collection_id).model_dump()), 200

@bp.route("/protected/custom/chat", methods=["POST"])
@ai.access_control.token_required
async def post_protected_custom_chat():
    request_json = request.json
    messages = request_json.get("messages")
    collection_id = request_json.get("collection_id")

    if await ai.access_control.has_access(collection_id):
        return jsonify(chat(messages, collection_id).model_dump()), 200
    return jsonify({"error": "Access denied to collection"}), 403

@bp.route("/protected/rate-limiting/chat", methods=["POST"])
@ai.access_control.token_required
@ai.access_control.enforce_access("collection_id")
@ai.rate_limit.limit(key="knowledge-base-chat")
async def post_rate_limiting_chat():
    request_json = request.json
    messages = request_json.get("messages")
    collection_id = request_json.get("collection_id")

    return jsonify(chat(messages, collection_id).model_dump()), 200

@bp.route("/tokenBalance", methods=["GET"])
@ai.access_control.token_required
async def token_remaining():
    key = request.args.get("key")
    if not key:
        return jsonify({"error": "Key is required"}), 400
    return jsonify(ai.rate_limit.get_token_balance(key)), 200
