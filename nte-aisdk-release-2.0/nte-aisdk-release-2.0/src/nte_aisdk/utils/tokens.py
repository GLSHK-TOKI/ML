
import logging

import tiktoken
from langchain.schema import BaseMessage

logger = logging.getLogger(__name__)


def num_tokens_from_messages(messages: list[BaseMessage], model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages.
    ref: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
    elif "gpt-3.5-turbo" in model:
        logger.debug("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        logger.debug("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        msg = f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        raise NotImplementedError(
            msg
        )
    num_tokens = 0
    for message in messages:
        # Simplify token calculation for langchain messages
        num_tokens += tokens_per_message
        num_tokens += len(encoding.encode(str(message.content)))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens