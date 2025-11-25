import { SDKException } from './exception-util.js';
import { errors, TransportRequestOptionsWithOutMeta } from '@elastic/transport';
import logger from "../logger/logger.js";
import type { estypes } from '@elastic/elasticsearch';

const _log = logger.child({ module: 'ai-sdk-node.exception.exception-es' });

export class ElasticsearchError extends SDKException {
    constructor(statusCode: number, message: string) {
        super(statusCode, message);
    }
}

export class ElasticsearchTransportError extends ElasticsearchError {
    constructor(statusCode: number, message: string) {
        super(statusCode, message);
    }
}

export class ElasticsearchApiError extends ElasticsearchError {
    constructor(statusCode: number, message: string) {
        super(statusCode, message);
    }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function handleElasticsearchOperation<F extends (...args: any[]) => PromiseLike<Awaited<ReturnType<F>>>>(operation: F) {
    return async (...args: Parameters<F>) => {
        try{
            return await operation(...args)
        } catch(e: unknown) {
            if (e instanceof errors.ConnectionError || e instanceof errors.TimeoutError) {
                const msg = 'An error occurred during an Elasticsearch operation';
                _log.error({ msg: msg, status_code: 400, err: e })
                throw new ElasticsearchTransportError(400, msg);
            } else if (e instanceof errors.ResponseError) {
                const msg = 'API error in Elasticsearch';
                _log.error({ msg: msg, status_code: e.statusCode || 500, err: e })
                throw new ElasticsearchApiError(e.statusCode || 500, msg);
            } else {
                const msg = 'An unexpected error occurred';
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

export function handleElasticsearchBulkOperation(operation: (params: estypes.BulkRequest, options?: TransportRequestOptionsWithOutMeta) => Promise<estypes.BulkResponse>) {
    return async (...args: Parameters<typeof operation>) => {
            const result = await handleElasticsearchOperation(operation)(...args);
            if (result.errors) {
                const errors = result.items
                    .filter(
                        (item) =>
                        item.index?.error ||
                        item.update?.error ||
                        item.delete?.error ||
                        item.create?.error
                    )
                    .map(
                        (item) => ({
                            status: item.index?.status || item.update?.status || item.delete?.status || item.create?.status,
                            ...item.index?.error || item.update?.error || item.delete?.error || item.create?.error
                        })
                    );
                const msg = 'Error(s) occurred during a bulk Elasticsearch operation';
                _log.error({ msg: msg, status_code: 400, err: errors })
                throw new ElasticsearchApiError(400, msg);
            }
            return result;
    }
}