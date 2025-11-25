import { SDKException } from './exception-util.js';
import logger from "../logger/logger.js";

const _log = logger.child({ module: 'ai-sdk-node.exception.exception-msgraph' });

// Base class for Azure OpenAI and VertexAI model errors
export class MSGraphError extends SDKException {
    /** Base exception for AzureOpenAI-related errors. */
}

// Utility function to handle operations for both Azure and Google Vertex AI models
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function handMSGraphOperation(operation: any, ...args: any[]) {
    try {
        return await operation(...args);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (e: any) {
        if (e.message && e.responseStatusCode) {
            _log.error({ msg: e.message, status_code: e.responseStatusCode })
            throw new MSGraphError(e.responseStatusCode, e.mesage);
        }
        const msg = 'An unexpected MS Graph error occurred';
        _log.error({msg: msg, status_code: 500, err: e})
        throw new MSGraphError(500, msg);
    }
}