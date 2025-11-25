import { get_encoding, encoding_for_model, Tiktoken, TiktokenModel } from 'tiktoken';
import logger from "../logger/logger";
import { BaseMessage } from '@langchain/core/messages';

export function num_tokens_from_messages(messages: BaseMessage[], model: string = "gpt-3.5-turbo-0613"): number {
    let encoding: Tiktoken;
    try {
        encoding = encoding_for_model(model as TiktokenModel);
    } catch (error) {
        logger.debug(`Warning: model "${model}" not found. Using cl100k_base encoding.`);
        encoding = get_encoding("cl100k_base");
    }

    let tokens_per_message: number;

    if (model === "gpt-3.5-turbo-0301") {
        tokens_per_message = 4;
    } else if (
        model === "gpt-4-0613" ||
        model === "gpt-3.5-turbo-0613" ||
        model === "gpt-3.5-turbo-16k-0613"
    ) {
        tokens_per_message = 3;
    } else if (model.includes("gpt-3.5-turbo")) {
        logger.debug(`Warning: gpt-3.5-turbo model "${model}" may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.`);
        const result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613");
        encoding.free();
        return result;
    } else if (model.includes("gpt-4")) {
        logger.debug(`Warning: gpt-4 model "${model}" may update over time. Returning num tokens assuming gpt-4-0613.`);
        const result = num_tokens_from_messages(messages, "gpt-4-0613");
        encoding.free();
        return result;
    } else {
        encoding.free();
        const msg = `num_tokens_from_messages() is not implemented for model "${model}". See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.`;
        throw new Error(msg);
    }

    let num_tokens = 0;
    for (const message of messages) {
        num_tokens += tokens_per_message;
        if (typeof message.content === 'string') {
            num_tokens += encoding.encode(message.content).length;
        } else {
            logger.debug(`Warning: message content is not a string for a message. Skipping content tokenization for it.`);
        }
    }
    num_tokens += 3;

    encoding.free();
    return num_tokens;
}