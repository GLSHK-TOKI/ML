import { SDKException } from './exception-util.js';
import logger from "../logger/logger.js";
import { APIError,  } from 'openai';

const _log = logger.child({ module: 'ai-sdk-node.exception.exception-model' });

// Base class for Azure OpenAI and VertexAI model errors
export class LlmError extends SDKException {
    /** Base exception for AzureOpenAI-related errors. */
}

// Azure OpenAI Model Exceptions
export class LlmAPIError extends LlmError {
    /** Raised when the response validation failed. */
}


// Utility function to handle operations for both Azure OpenAI LLM
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function handleLlmOperation<F extends (...args: any[]) => PromiseLike<Awaited<ReturnType<F>>>>(operation: F) {
    return async (...args: Parameters<F>) => {
        try{
            return await operation(...args)
        } catch(e: unknown) {
            if (e instanceof APIError) {
                const msg = 'An error occurred during an Azure OpenAI LLM operation';
                _log.error({ msg: msg, status_code: e.status ?? 500 , err: e })
                throw new LlmAPIError(e.status ?? 500, msg)
            } else {
                const msg = 'An unexpected LLM Model error occurred';
                if (e instanceof Error) {
                    _log.error({ msg: msg, status_code: 500, err: e })
                } else {
                    _log.error({ msg: msg, status_code: 500 })
                }
                throw new SDKException(500, msg);
            }
        }
    }
}