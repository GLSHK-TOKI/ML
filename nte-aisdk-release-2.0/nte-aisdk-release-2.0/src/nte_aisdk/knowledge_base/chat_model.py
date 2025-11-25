from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from nte_aisdk import errors, types
from nte_aisdk.feature import Feature
from nte_aisdk.utils import (
    convert_to_langchain_messages,
    ensure_arguments,
    normalize_message,
    normalize_messages,
    num_tokens_from_messages,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from concurrent.futures import Future

    from google.genai import types as genai_types

    from nte_aisdk.language_model import LanguageModel

    from ..pii_detection.pii_detector import PIIDetector
    from .store import KnowledgeBaseStore

logger = logging.getLogger(__name__)

DEFAULT_CANNOT_ANSWER_MESSAGE = "Sorry, CX Knowledge base don't have such information."


class KnowledgeBaseChatModel(Feature):
    """KnowledgeBaseChatModel is a class that can be used to chat with Retrieval-Augmented-Generation (RAG) technique
    with contexts from knowledge base store.
    """

    _BASE_RESPONSE_SCHEMA: ClassVar[dict[str, Any]] = {
        "title": "rag_response",
        "description": "The response schema for the RAG chat model.",
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "State you thinking process, quote the exact message from context, and give reason why from this message you can deduce the answer",
            },
            "answer": {
                "type": "string",
                "description": "Your answer here, split content to smaller paragraph and bullet point and add br tag to present line breaking, dont change/add content except line break",
            },
            "challenge_verification": {
                "type": "boolean",
                "description": "Does the given knowledge base provide information for part of the question? Return true or false",
            },
        },
        "required": ["thought", "answer", "challenge_verification"],
        "additionalProperties": False,
    }

    _SOURCE_ITEM_SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Human-readable title with file extension of reference source, for example, 'example.pdf' or '例子.pdf'",
            },
            "url": {
                "type": "string",
                "description": "Url of reference sources that you using and can be found in context in plain text format",
            },
            "start_page": {
                "type": ["integer", "null"],
                "description": "The start page number which must be same as value of field 'startPage' of the context, otherwise null",
            },
            "end_page": {
                "type": ["integer", "null"],
                "description": "The end page number which must be same as value of field 'endPage' of the context, otherwise you must provide null",
            },
            "source_type": {
                "type": "string",
                "enum": ["text", "image"],
                "description": "The type of source: 'text' for text-based sources, 'image' for image-based sources",
            },
        },
        "additionalProperties": False,
        "required": ["title", "url", "start_page", "end_page", "source_type"],
    }

    _SOURCES_SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "array",
        "items": _SOURCE_ITEM_SCHEMA,
    }

    _language_model: LanguageModel
    _store: KnowledgeBaseStore
    _logger: logging.Logger
    max_tokens_history: int
    max_tokens_history_summary: int
    max_tokens_context: int
    max_words_answer: int
    _pii_detector: PIIDetector | None = None
    _pii_detection_mode: Literal["FLAG", "FLAG_MASK"] | None = None

    _background_tasks: set[Future] | None  # This creates a strong reference for the async background tasks
    _background_loop: asyncio.AbstractEventLoop | None
    _background_thread: (
        threading.Thread | None
    )  # A separate thread to run the background loop, to handle async tasks, while the main thread can continue to process requests.

    @ensure_arguments
    def __init__(
        self,
        language_model: LanguageModel,
        store: KnowledgeBaseStore,
        logger: logging.Logger = logger,
        *,
        max_tokens_history: int = 4096,
        max_tokens_history_summary: int = 256,
        max_tokens_context: int = 20000,
        max_words_answer: int = 400,
        # New multimodal parameters
        retriever_size_image: int = 20,
        retriever_size_text: int = 30,
        image_retriever_threshold: float = 0.8,
        text_retriever_threshold: float = 1.2,
    ):
        super().__init__()
        self._language_model = language_model
        self._store = store
        self._logger = logger
        self.max_tokens_history = max_tokens_history
        self.max_tokens_history_summary = max_tokens_history_summary
        self.max_tokens_context = max_tokens_context
        self.max_words_answer = max_words_answer
        # New multimodal parameters
        self.retriever_size_image = retriever_size_image
        self.retriever_size_text = retriever_size_text
        self.image_retriever_threshold = image_retriever_threshold
        self.text_retriever_threshold = text_retriever_threshold
        self._response_schema = self._create_response_schema()

    def _create_response_schema(self) -> dict[str, Any]:
        schema = copy.deepcopy(self._BASE_RESPONSE_SCHEMA)
        schema["properties"] = copy.deepcopy(schema["properties"])
        schema["properties"]["sources"] = copy.deepcopy(self._SOURCES_SCHEMA)
        schema["required"] = [*schema["required"], "sources"]
        return schema

    @ensure_arguments
    def chat(
        self,
        messages: list[types.MessageOrDict],
        collection_id: str,
        *,
        # TODO: Convert into a customizable parameters type for all generate content config
        safety_settings: list[genai_types.SafetySetting] | None = None,
        thinking_config: genai_types.ThinkingConfig | None = None,
        # TODO: Convert into a customizable search parameters type
        retriever_size_text: int | None = None,
        max_tokens_context: int | None = None,
        # New multimodal parameters
        query_image: str | None = None,
        retriever_size_image: int | None = None,
        enable_multimodal_search: bool = False,
        image_retriever_threshold: float | None = None,
        text_retriever_threshold: float | None = None,
    ) -> types.GenerateResponse:
        if not messages or len(messages) < 1:
            msg = "At least one message is required for chat."
            raise errors.InvalidArgumentError(msg)

        if retriever_size_text is None:
            retriever_size_text = self.retriever_size_text
        if max_tokens_context is None:
            max_tokens_context = self.max_tokens_context
        if retriever_size_image is None:
            retriever_size_image = self.retriever_size_image
        if image_retriever_threshold is None:
            image_retriever_threshold = self.image_retriever_threshold
        if text_retriever_threshold is None:
            text_retriever_threshold = self.text_retriever_threshold

        token_counter = self.SingleFunctionTokenCounter()

        p_history_messages = normalize_messages(messages[:-1])  # Exclude the last message which is the question
        p_question_message = normalize_message(messages[-1])  # The last message is the question

        # 2. Reformulate the query from question based on chat history
        query = self._prepare_retrieval_query(p_history_messages, p_question_message, token_counter=token_counter)

        # 3. Retrieve for the relevant content from question (multimodal or text-only)
        cleaned_query_image = (
            self._clean_base64_image(query_image) if enable_multimodal_search else None
        )

        combined_results = self._perform_search(
            query,
            collection_id,
            retriever_size_text,
            text_retriever_threshold,
            cleaned_query_image,
            retriever_size_image,
            image_retriever_threshold,
            enable_multimodal_search
        )
        if len(combined_results) == 0:
            self._pii_and_log_in_background(
                {
                    "model_name": self._language_model.model_id,
                    "question": p_question_message.text,
                    "answer": DEFAULT_CANNOT_ANSWER_MESSAGE,
                    "history": [
                        message.model_dump(mode="json") for message in p_history_messages
                    ],
                    "sources": [],
                    "collection_id": collection_id,
                    "context_list": [],
                    "token_usage": token_counter.to_usage().model_dump(mode="json"),
                }
            )
            return types.GenerateResponse(
                message=types.Message(
                    role=types.Role.ASSISTANT,
                    parts=[
                        types.TextPart(text=DEFAULT_CANNOT_ANSWER_MESSAGE)
                    ],
                    message_id=None,
                ),
                metadata=types.GenerateResponseMetadata(
                    model_id=self._language_model.model_id,
                    usage=types.LanguageModelResponseUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                    ),
                    confidence=None,
                ),
            )

        context, context_list = self._compose_context(combined_results, max_tokens_context, cleaned_query_image)

        # 4 Summarise the chat history when exceed the maximum token
        history_messages_for_prompt = self._summarise_history(p_history_messages, p_question_message, token_counter=token_counter)

        # 5. Generate the chat response with context and images
        system_instruction = self._create_system_instruction(p_question_message, context)

        # Create enhanced question message with image parts if we have images
        enhanced_question_message = self._create_enhanced_message_with_images(
            p_question_message,
            cleaned_query_image,
            combined_results
        )

        result = self._language_model.do_generate(
            messages=[*history_messages_for_prompt, enhanced_question_message],
            instructions=system_instruction,
            response_type="json",
            response_schema=self._response_schema,
            **({"safety_settings": safety_settings} if safety_settings else {}),
            **({"thinking_config": thinking_config} if thinking_config else {})
        )

        self._add_prompt_tokens_to_context(result.metadata.usage.prompt_tokens)
        self._add_completion_tokens_to_context(result.metadata.usage.completion_tokens)
        token_counter.add_prompt_tokens(result.metadata.usage.prompt_tokens)
        token_counter.add_completion_tokens(result.metadata.usage.completion_tokens)

        # There will be a json parsing error later
        if result.message.data is None:
            msg = "JSON parsing error from LLM response, the response does not contain 'data' key."
            raise errors.SDKError(msg)

        answer = getattr(result.message, "data", {}).get("answer", "")
        reasoning = result.message.reasoning
        sources = getattr(result.message, "data", {}).get("sources", [])

        self._pii_and_log_in_background(
            {
                "model_name": result.metadata.model_id,
                "question": p_question_message.text,
                "answer": answer,
                "reasoning": reasoning,
                "history": [
                    message.model_dump(mode="json") for message in history_messages_for_prompt
                ],
                "sources": sources,
                "collection_id": collection_id,
                "context_list": context_list,
                "token_usage": token_counter.to_usage().model_dump(mode="json"),
            }
        )

        result.metadata.usage = token_counter.to_usage()
        return result

    @ensure_arguments
    def chat_stream(
            self,
            messages: list[types.MessageOrDict],
            collection_id: str,
            *,
            max_tokens_context: int | None = None,
            # New multimodal parameters
            query_image: str | None = None,
            retriever_size_image: int | None = None,
            retriever_size_text: int | None = None,
            enable_multimodal_search: bool = False,
            image_retriever_threshold: float | None = None,
            text_retriever_threshold: float | None = None,
        ) -> Iterator[types.StreamResponse]:
        if not messages or len(messages) < 1:
            msg = "At least one message is required for chat."
            raise errors.InvalidArgumentError(msg)

        if max_tokens_context is None:
            max_tokens_context = self.max_tokens_context
        if retriever_size_image is None:
            retriever_size_image = self.retriever_size_image
        if retriever_size_text is None:
            retriever_size_text = self.retriever_size_text
        if image_retriever_threshold is None:
            image_retriever_threshold = self.image_retriever_threshold
        if text_retriever_threshold is None:
            text_retriever_threshold = self.text_retriever_threshold

        token_counter = self.SingleFunctionTokenCounter()

        p_history_messages = normalize_messages(messages[:-1])  # Exclude the last message which is the question
        p_question_message = normalize_message(messages[-1])  # The last message is the question

        # 2. Reformulate the query from question based on chat history
        query = self._prepare_retrieval_query(p_history_messages, p_question_message, token_counter=token_counter)

        cleaned_query_image = (
            self._clean_base64_image(query_image) if enable_multimodal_search else None
        )

        # 3. Retrieve for the relevant content from question (multimodal or text-only)
        combined_results = self._perform_search(
            query,
            collection_id,
            retriever_size_text,
            text_retriever_threshold,
            cleaned_query_image,
            retriever_size_image,
            image_retriever_threshold,
            enable_multimodal_search
        )
        if len(combined_results) == 0:
            self._pii_and_log_in_background(
                {
                    "model_name": self._language_model.model_id,
                    "question": p_question_message.text,
                    "answer": DEFAULT_CANNOT_ANSWER_MESSAGE,
                    "history": [
                        message.model_dump(mode="json") for message in p_history_messages
                    ],
                    "sources": [],
                    "collection_id": collection_id,
                    "context_list": [],
                    "token_usage": token_counter.to_usage().model_dump(mode="json"),
                }
            )
            yield types.StreamResponse(
                message=types.Message(
                    role=types.Role.ASSISTANT,
                    parts=[
                        types.TextPart(text=DEFAULT_CANNOT_ANSWER_MESSAGE)
                    ],
                    message_id=None,
                ),
                last_chunk=True,
                metadata=types.GenerateResponseMetadata(
                    model_id=self._language_model.model_id,
                    usage=types.LanguageModelResponseUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                    ),
                    confidence=None,
                ),
            )
            return

        context, context_list = self._compose_context(combined_results, max_tokens_context, cleaned_query_image)

        # 4 Summarise the chat history when exceed the maximum token
        history_messages_for_prompt = self._summarise_history(p_history_messages, p_question_message, token_counter=token_counter)

        # 5. Generate the chat response with context and images using streaming
        system_instruction = self._create_system_instruction(p_question_message, context)

        # Create enhanced question message with image parts if we have images
        enhanced_question_message = self._create_enhanced_message_with_images(
            p_question_message,
            cleaned_query_image,
            combined_results
        )

        # Process streaming chunks and yield them to the client
        accumulated_json_text = ""
        accumulated_reasoning = ""

        for chunk in self._language_model.do_stream(
            messages=[*history_messages_for_prompt, enhanced_question_message],
            instructions=system_instruction,
            response_type=types.ResponseType.JSON,
            response_schema=self._response_schema,
        ):
            if chunk.last_chunk and chunk.metadata is not None:
                # Add tokens count for rate limiting
                self._add_prompt_tokens_to_context(chunk.metadata.usage.prompt_tokens)
                self._add_completion_tokens_to_context(chunk.metadata.usage.completion_tokens)
                token_counter.add_prompt_tokens(chunk.metadata.usage.prompt_tokens)
                token_counter.add_completion_tokens(chunk.metadata.usage.completion_tokens)

                chunk.metadata.usage = token_counter.to_usage()
                yield chunk

                # Log after streaming completes (success or failure)
                try:
                    parsed_content = json.loads(accumulated_json_text)
                    sources = parsed_content.get("sources", [])
                    answer = parsed_content.get("answer", accumulated_json_text)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse accumulated content as JSON")
                    sources = []
                    answer = accumulated_json_text

                self._pii_and_log_in_background({
                    "question": p_question_message.text,
                    "answer": answer,
                    "reasoning": accumulated_reasoning,
                    "history": [message.model_dump(mode="json") for message in p_history_messages],
                    "sources": sources,
                    "collection_id": collection_id,
                    "context_list": context_list,
                    "token_usage": token_counter.to_usage().model_dump(mode="json"),
                })
            else:
                # Accumulate content for final logging
                if chunk.message.text:
                    accumulated_json_text += chunk.message.text
                if chunk.message.reasoning:
                    accumulated_reasoning += chunk.message.reasoning

                # Yield intermediate chunks as-is
                yield chunk

    def add_pii_detector_on_log(self, pii_detector: PIIDetector, mode: Literal["FLAG", "FLAG_MASK"] ):
        """Add a PII detector to be used during logging.

        Args:
            pii_detector (PIIDetector): The PII detector to use
            mode (str): The detection mode - FLAG
        """
        self._pii_detector = pii_detector
        self._pii_detection_mode = mode

        # Start the background thread and loop if PII detection is enabled
        def _run_background_loop():
            if self._background_loop is not None:
                asyncio.set_event_loop(self._background_loop)
                self._background_loop.run_forever()

        self._background_tasks = set()
        self._background_loop = asyncio.new_event_loop()
        self._background_thread = threading.Thread(target=_run_background_loop, daemon=True)
        self._background_thread.start()

    def _perform_search(
        self,
        query: str,
        collection_id: str,
        retriever_size_text: int,
        text_retriever_threshold: float,
        query_image: str | None,
        retriever_size_image: int | None,
        image_retriever_threshold: float,
        enable_multimodal_search: bool,
    ) -> list[dict[str, Any]]:
        """Perform search based on configuration (multimodal or text-only).

        Uses Reciprocal Rank Fusion (RRF) to combine text and image search results
        when both are available, providing a more balanced hybrid ranking.

        Returns:
            Combined list of text and image search results, ranked by RRF score or _score
        """
        # Fetch text results using regular search
        text_results = self._store.search(
            query,
            collection_id,
            size=retriever_size_text,
            threshold=text_retriever_threshold
        )

        # Fetch image results if multimodal search is enabled
        image_results = []
        if enable_multimodal_search:
            multimodal_results = self._store.search_multimodal(
                query,
                query_image,
                collection_id,
                size_image=retriever_size_image or self.retriever_size_image,
                threshold=image_retriever_threshold
            )
            image_results = multimodal_results["image_results"]

        # Apply Reciprocal Rank Fusion (RRF) if we have both text and image results
        if text_results and image_results:
            combined_results = self._apply_rrf_scoring(text_results, image_results)
        else:
            # If only one type of result, just combine and sort by original score
            combined_results = text_results + image_results
            combined_results.sort(key=lambda x: x.get("_score", 0), reverse=True)

        return combined_results

    def _apply_rrf_scoring(
        self,
        text_results: list[dict[str, Any]],
        image_results: list[dict[str, Any]],
        k: int = 60
    ) -> list[dict[str, Any]]:
        """Apply Reciprocal Rank Fusion (RRF) to combine text and image search results.

        RRF formula: RRF_score = sum(1 / (k + rank_i)) for each retrieval method
        where k is a constant (typically 60) and rank_i is the rank in each result list.

        Args:
            text_results: Ranked list of text search results
            image_results: Ranked list of image search results
            k: RRF constant (default: 60, as recommended in literature)

        Returns:
            Combined and re-ranked list of results sorted by RRF score
        """
        # Create a dictionary to store RRF scores by document ID
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        # Process text results (rank starts at 0)
        for rank, doc in enumerate(text_results):
            doc_id = doc.get("_id", "")
            if doc_id:
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank + 1))
                doc_map[doc_id] = doc

        # Process image results (rank starts at 0)
        for rank, doc in enumerate(image_results):
            doc_id = doc.get("_id", "")
            if doc_id:
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank + 1))
                if doc_id not in doc_map:  # Avoid overwriting if doc appears in both
                    doc_map[doc_id] = doc

        # Add RRF score to each document and create combined list
        combined_results = []
        for doc_id, rrf_score in rrf_scores.items():
            doc = doc_map[doc_id].copy()
            doc["_rrf_score"] = rrf_score
            combined_results.append(doc)

        # Sort by RRF score (highest first)
        combined_results.sort(key=lambda x: x.get("_rrf_score", 0), reverse=True)

        logger.info(
            "Applied RRF scoring",
            extra={
                "props": {
                    "text_results_count": len(text_results),
                    "image_results_count": len(image_results),
                    "combined_results_count": len(combined_results),
                    "rrf_k": k,
                }
            },
        )

        return combined_results

    def _clean_base64_image(self, image: str | None) -> str | None:
        """Normalize image input by removing data URL headers and fixing padding."""
        if not image:
            return None

        base64_part = image
        if image.startswith("data:image/"):
            comma_index = image.find(",")
            if comma_index == -1:
                return None
            base64_part = image[comma_index + 1:]

        missing_padding = len(base64_part) % 4
        if missing_padding:
            base64_part += "=" * (4 - missing_padding)

        return base64_part

    def _prepare_retrieval_query(
            self,
            p_history_messages: list[types.Message],
            p_question_message: types.Message,
            token_counter: Feature.SingleFunctionTokenCounter
        ) -> str:
        """Reformulate the query from question based on chat history"""
        if p_history_messages:
            reformulate_result = self._reformulate_question_with_history(p_question_message, p_history_messages)

            self._add_prompt_tokens_to_context(reformulate_result.metadata.usage.prompt_tokens)
            self._add_completion_tokens_to_context(reformulate_result.metadata.usage.completion_tokens)
            token_counter.add_prompt_tokens(reformulate_result.metadata.usage.prompt_tokens)
            token_counter.add_completion_tokens(reformulate_result.metadata.usage.completion_tokens)

            query = reformulate_result.message.text
            logger.info(
                "Reformulated question with chat history",
                extra={
                    "props": {
                        "question": p_question_message.text,
                        "history": [
                            message.model_dump(mode="json") for message in p_history_messages
                        ],
                        "reformulated_question": query,
                    }
                },
            )
        else:
            query = p_question_message.text
        return query

    def _summarise_history(
        self,
        p_history_messages: list[types.Message],
        p_question_message: types.Message,
        token_counter: Feature.SingleFunctionTokenCounter
    ):
        num_tokens_history = num_tokens_from_messages(convert_to_langchain_messages(p_history_messages))
        num_tokens_question = num_tokens_from_messages(
            convert_to_langchain_messages([p_question_message])
        )  # The last message is the question
        max_tokens_for_history_with_summary = self.max_tokens_history - num_tokens_question
        if num_tokens_history > max_tokens_for_history_with_summary:
            return self._get_history_with_generated_summary(
                p_history_messages,
                max_tokens_for_history_with_summary,
                token_counter=token_counter
            )
        return p_history_messages

    def _get_history_with_generated_summary(
        self,
        history: list[types.Message],
        max_tokens_for_history_with_summary: int,
        token_counter: Feature.SingleFunctionTokenCounter
    ) -> list[types.Message]:
        """Try to keep as much history as possible under the maximum token limit.
        If the history exceeds the maximum token, then keep the latest messages
        and summarize the rest of the history.
        """
        history_to_keep = history.copy()
        history_to_summarize = []
        while (
            history_to_keep
            and num_tokens_from_messages(convert_to_langchain_messages(history_to_keep)) > (
                max_tokens_for_history_with_summary - self.max_tokens_history_summary
            )
        ):
            history_to_summarize.append(history_to_keep.pop(0))

        system_instruction = f"""
You are a summarizer for a conversation between a user and a bot who is an expert
on AI CX Knowledge Base Assistant.

You will be given a conversation history in reverse chronological order. You should
summarize the conversation history to keep the most important information. You should keep
the summary under {self.max_tokens_history_summary} tokens.

When summarizing the conversation history, give higher value to the latest question
and user's question and give lower value to the bot's response.
"""
        human_message = types.Message(
            role=types.Role.USER,
            parts=[
                types.TextPart(
                    text=json.dumps([p_history.model_dump() for p_history in history_to_summarize])
                )
            ],
        )
        result = self._language_model.do_generate(
            messages=[human_message],
            instructions=system_instruction,
        )
        self._add_prompt_tokens_to_context(result.metadata.usage.prompt_tokens)
        self._add_completion_tokens_to_context(result.metadata.usage.completion_tokens)
        token_counter.add_prompt_tokens(result.metadata.usage.prompt_tokens)
        token_counter.add_completion_tokens(result.metadata.usage.completion_tokens)

        return [*history_to_keep, result.message]

    def _reformulate_question_with_history(
        self,
        question_message: types.Message,
        history_messages: list[types.Message]
    ):
        # 1. Construct the system message for RAG qna
        system_instruction = """
You are a conversational interpreter for a conversation between a user and
a bot who is an expert on a knowledge base.

The user will give you a follow up question without context. You will reformulate the question
to take into account the context of the conversation as a standalone question. You should assume the question
is related to the knowledge base. You should also consult with the Chat History
below when reformulating the question. For example,
you will substitute pronouns for mostly likely noun in the conversation
history.

When reformulating the question give higher value to the latest question and response
in the Chat History. The chat history is in reverse chronological order, so the most
recent exchange is at the top.

Only respond with the reformulated question. You should respond with the reformulated question directly and
avoid re-asking user for further clarification. If there is no chat history, then respond
only with the question unchanged.
"""

        human_message = types.Message(
            role=types.Role.USER,
            parts=[
                types.TextPart(
                    text=f"""
        Chat History:
        =============
        {json.dumps([p_history.model_dump() for p_history in history_messages])}
        Follow up input: {question_message.text}
        Standalone Question:
        """
                )
            ],
        )

        # 3. Get the reformulated question from chat model
        return self._language_model.do_generate(
            messages=[human_message],
            instructions=system_instruction,
        )

    def _get_max_token_context(self, search_results, max_tokens_context):
        context = []
        token_count = 0
        for search_result in search_results:
            try:
                content = search_result.get("content", "")
                n_token = search_result.get("n_token", 0)
                meta = search_result.get("meta", {})
                title = meta.get("title", "Unknown")
                url = meta.get("webUrl", "")

                if content and content.startswith("data:image/") and "base64," in content:
                    content = f"[KNOWLEDGE BASE IMAGE: {title}] - This is an image from the knowledge base that has been attached to this message. Please analyze this image as reference material for answering the question."
                if (token_count + n_token) < max_tokens_context:
                    token_count += n_token
                    context.append(
                        {
                            "title": title,
                            "content": content,
                            "url": url,
                            "startPage": meta.get("startPage", None),
                            "endPage": meta.get("endPage", None),
                        }
                    )
                else:
                    break
            except (KeyError, TypeError) as e:
                logger.warning(f"Error processing search result: {e}, result: {search_result}")
                continue

        return context

    def _compose_context(self, search_results, max_tokens_context, query_image=None):
        context_list = self._get_max_token_context(search_results, max_tokens_context)
        context_parts = []
        if query_image:
            context_parts.append(
                "USER PROVIDED IMAGE: The user has provided an image for analysis. "
                "Please analyze this image and reference it in your response when relevant to the question. "
                "The image is attached to this message."
            )

        for c in context_list:
            context_parts.append(  # noqa: PERF401
                "<ContextChunk>"
                f"<title>{c['title']}</title>"
                f"<url>{c['url']}</url>"
                f"<startPage>{'' if c['startPage'] is None else c['startPage']}</startPage>"
                f"<endPage>{'' if c['endPage'] is None else c['endPage']}</endPage>"
                f"<content>{c['content']}</content>"
                "</ContextChunk>"
            )

        context = "\n\n".join(context_parts)
        return context, context_list

    def _create_enhanced_message_with_images(
        self,
        original_message: types.Message,
        user_image: str | None,
        search_results: list[dict[str, Any]]
    ) -> types.Message:
        """Create an enhanced message with proper image parts for multimodal LLM consumption.

        Args:
            original_message: The original text message from the user
            user_image: Cleaned base64 image string provided by the user (single image)
            search_results: Combined search results that may contain image results

        Returns:
            Enhanced message with FilePart objects for images
        """
        enhanced_parts = list(original_message.parts)
        if user_image:
            mime_type = "image/jpeg"
            image_part = types.FilePart(
                file=types.FileWithBytes(
                    name="user_image.jpg",
                    mime_type=mime_type,
                    bytes=user_image
                )
            )
            enhanced_parts.append(image_part)

        for i, result in enumerate(search_results):
            content = result.get("content", "")
            if content and content.startswith("data:image/") and "base64," in content:
                header_part = content.split(",")[0]
                mime_type = header_part.split(";")[0].replace("data:", "")
                comma_index = content.find(",")
                if comma_index != -1:
                    base64_part = content[comma_index + 1:]
                    missing_padding = len(base64_part) % 4
                    if missing_padding:
                        base64_part += "=" * (4 - missing_padding)
                    meta = result.get("meta", {})
                    title = meta.get("title", f"kb_image_{i}")
                    kb_image_part = types.FilePart(
                        file=types.FileWithBytes(
                            name=title,
                            mime_type=mime_type,
                            bytes=base64_part
                        )
                    )
                    enhanced_parts.append(kb_image_part)
        return types.Message(
            role=original_message.role,
            parts=enhanced_parts,
            message_id=original_message.message_id,
            metadata=original_message.metadata
        )

    def _create_system_instruction(
        self,
        question_message: types.Message,
        context: str
    ) -> str:
        # 1. Construct the system message for RAG qna
        schema_properties = json.dumps(self._response_schema["properties"])
        return f"""You are AI CX Knowledge Base Assistant, answer question from user, max characters in answer must under {self.max_words_answer} words and keeping the article formatting.
1.Answering the question or providing useful information in detail based on the information from the given knowledge base below.
2.Quote the source link from knowledge base.
3.Double Check your answer, only give factual information that given in knowledge base.
4.Think about how to answering the question step by step.
5.If knowledge base don't have related information, tell user cx knowledge base don't have such information. Do not exceed the token size.

Knowledge base:
<Knowledge Base Context>
{context}
</Knowledge Base Context>

In the "thought" field:
<A paragraph of your thinking process, quote the exact message from context, and give reason why from this message you can deduce the answer>

In the "answer" field:
<A paragraph of your answer here, split content to smaller paragraph and bullet point, dont change/add content>

In the "sources" field:
<Give a list of ALL sources that you referenced in your answer. IMPORTANT: You must include EVERY source that you used to answer the question.
1. In each source, in the "title" field, provide a human-readable title with file extension of reference source, for example, 'example.pdf' or '例子.pdf'. Please decode any URL-encoded characters, especially Chinese characters, to their original form.
2. In each source, in the "url" field, include the exact url of reference sources that you using and can be found in context.
3. In the "start_page" field, only include the startPage number which must be same as value of field 'startPage' of the context, otherwise you must provide null
4. In the "end_page" field, only include the endPage number which must be same as value of field 'endPage' of the context, otherwise you must provide null
5. In the "source_type" field, determine the type:
   - Use "image" if the Content field starts with "data:image/" OR contains "[KNOWLEDGE BASE IMAGE: title] - This is an image from the knowledge base that has been attached to this message. Please analyze this image as reference material for answering the question."
   - Use "text" for all other sources (normal text content, PDFs, documents, etc.)
   
REMEMBER: Include ALL sources you referenced, both text and images. Do not skip any sources you used in your answer.
>

In the "challenge_verification" field:
<Does the given knowledge base provide information for part of the question '{question_message.text}'? Return true or false>

Follow JSON schema:
<JSONSchema>
{schema_properties}
</JSONSchema>

User Question:
<Question>
{question_message.text}
</Question>

A valid JSON must be constructed with double-quotes.Double quotes within strings must be escaped with backslash, single quotes within strings will not be escaped.
The value must be standard unicode characters without non-standard unicode character, without invalid tag, without markup like \\underline.
Output a response strictly in a valid JSON format that aligns with the schema above."""

    def _pii_and_log_in_background(self, data: dict):
        """This method is a placeholder for the background logging functionality.
        It is not implemented in this example, but can be used to log data in a separate thread or process.
        """
        # If PII detection is not enabled, log the data directly
        if not self._pii_detector or not self._pii_detection_mode:
            self._log(data)
            return

        # Create a new asyncio task to run the PII detection and logging in the background
        if self._background_loop is None or self._background_tasks is None:
            msg = "Background loop or background tasks is not initialized. Please ensure PII detection is enabled."
            raise errors.SDKError(msg)

        future = asyncio.run_coroutine_threadsafe(self._async_pii_and_log(data), self._background_loop)

        # To prevent keeping references to finished tasks forever,
        # make each task remove its own reference from the set after
        self._background_tasks.add(future)
        future.add_done_callback(
            lambda t: self._background_tasks.discard(t) if self._background_tasks is not None else None
        )

    async def _async_pii_and_log(self, data: dict):
        # Detect PII in the question and answer, and log the data
        # Skip pii detection if question or answer is None
        question = data.get("question")
        answer = data.get("answer")

        if question is not None:
            question_pii_response = self._pii_actions(question)
        if answer is not None:
            answer_pii_response = self._pii_actions(answer)

        # Skip logging PII response or masked text if both pii response is none
        if question_pii_response and answer_pii_response:
            pii_detected = bool(
                (question_pii_response.get("detected", False) if question_pii_response else False)
                or (answer_pii_response.get("detected", False) if answer_pii_response else False)
            )

            data["question"] = question_pii_response.get("text_masked", question) if question_pii_response else question
            data["answer"] = answer_pii_response.get("text_masked", answer) if answer_pii_response else answer

            pii_response: dict[str, Any] = {}
            pii_response["detected"] = pii_detected
            pii_response["question"] = (
                {
                    "detected": question_pii_response.get("detected"),
                    "result": question_pii_response.get("result", []) if self._pii_detection_mode == "FLAG" else None,
                }
                if question_pii_response
                else None
            )
            pii_response["answer"] = (
                {
                    "detected": answer_pii_response.get("detected"),
                    "result": answer_pii_response.get("result", []) if self._pii_detection_mode == "FLAG" else None,
                }
                if answer_pii_response
                else None
            )

            data["pii"] = pii_response

        # Exclude chat history in the log if FLAG_MASK mode is enabled to avoid logging text with PII
        if self._pii_detection_mode == "FLAG_MASK":
            data["history"] = None

        self._log(data)

    def _pii_actions(self, text: str) -> dict | None:
        """Detect and run mask on PII in text and return structured results based on the configurations

        Args:
            text: The text to check for PII

        Returns:
            A structured dictionary containing PII detection results, or None if detection fails or is disabled
        """
        if self._pii_detector and self._pii_detection_mode:
            try:  # Continue with the function execution regardless of PII detection errors
                pii_result = self._pii_detector.detect(text)
                pii_result_items = pii_result.get("data", [])

                if self._pii_detection_mode == "FLAG":
                    return {"detected": len(pii_result_items) > 0, "result": pii_result_items}

                if self._pii_detection_mode == "FLAG_MASK":
                    text_pii_masked = None
                    if len(pii_result_items) > 0:
                        pii_masked_result = self._pii_detector.mask(text, pii_result_items)
                        text_pii_masked = pii_masked_result.get("text", text)

                    return {
                        "detected": len(pii_result_items) > 0,
                        "result": pii_result_items,
                        **({"text_masked": text_pii_masked} if text_pii_masked is not None else {}),
                    }
            except errors.SDKError:
                logger.exception("SDK error occurred during PII detection")
                return None
            except Exception:
                logger.exception("Unexpected error occurred during PII detection")
                return None
            else:
                return None
        return None

    def _log(
        self,
        data: dict,
    ):
        user = self._get_user()

        self._logger.info(
            "New message on knowledge base chat",
            extra={
                "props": {
                    "data": {
                        "user": user,
                        "input": {
                            "question": data["question"],
                        },
                        "output": {
                            "answer": data["answer"],
                            "reasoning": data.get("reasoning"),
                            "history": data["history"],
                            "sources": data["sources"],
                        },
                        "model": {
                            "name": data.get("model_name"),
                        },
                        "metadata": {
                            "category": "knowledge-base",
                            "collection_id": data["collection_id"],
                        },
                        "context": {"documents": data["context_list"]},
                        "usage": {
                            "prompt_tokens": data["token_usage"]["prompt_tokens"],
                            "completion_tokens": data["token_usage"]["completion_tokens"],
                            "total_tokens": data["token_usage"]["prompt_tokens"]
                            + data["token_usage"]["completion_tokens"],
                        },
                        "correlation_id": _get_correlation_id(),
                        "pii": data.get("pii"),
                    }
                }
            },
        )


def _get_correlation_id():
    """Helper function to get correlation ID, with fallback if json_logging is not available."""
    try:
        import json_logging # type: ignore  # noqa: I001, PGH003, PLC0415

        return json_logging.get_correlation_id()
    except ImportError:
        json_logging = None
        logger.warning("json_logging is not installed, logging will not include extra properties.")

    return None
