import {AzureOpenAIModelConfig} from "./azure-openai/index.js";
import {AzureLangchainJsonParsingWrapper, AzureUnderlyingRawResponse, LlmBaseModel, ModelInvokeReturn} from "./llm-base-model.js";
import {SDKException} from "./exception/index.js";

type ResponseType = "text" | "json";

interface ParsedResult {
    data?: any;
    text?: string;
    confidence?: number | null;
    prompt_tokens: number;
    completion_tokens: number;
}

const logger = {
    error: (message: string, ...args: any[]) => console.error(message, ...args),
};

class LlmBaseModelParser {
    private model: LlmBaseModel;

    constructor(model: LlmBaseModel) {
        this.model = model;
    }

    public parse(response: ModelInvokeReturn, responseType: ResponseType = "text"): ParsedResult {
        const parserFunction = this._getParser();
        return parserFunction(response, responseType);
    }

    private _getParser(): (response: ModelInvokeReturn, responseType: ResponseType) => ParsedResult {
        if (this.model.modelConfig instanceof AzureOpenAIModelConfig) {
            return this._parseModelAzureAIResult;
        }
        const errorMessage = "Unsupported model type. Please provide a valid AzureOpenAIModelConfig.";
        logger.error(errorMessage);
        throw new SDKException(400,errorMessage);
    }

    private _parseModelAzureAIResult(response: ModelInvokeReturn, responseType: ResponseType): ParsedResult {
        let underlyingRawResponse: AzureUnderlyingRawResponse;
        let parsedJsonFromWrapper: any | null = null;
        let hadJsonParsingError: boolean = false;

        if (responseType === "json") {
            const jsonWrapper = response.raw as AzureLangchainJsonParsingWrapper;
            if (typeof jsonWrapper.raw === 'undefined') {
                 logger.error("Invalid response structure for JSON parsing. Expected Langchain wrapper with 'raw' and 'parsing_error' fields.");
                 underlyingRawResponse = response.raw as AzureUnderlyingRawResponse;
                 hadJsonParsingError = jsonWrapper.parsing_error;;
            } else {
                underlyingRawResponse = jsonWrapper.raw;
                parsedJsonFromWrapper = jsonWrapper.parsed;
            }
        } else {
            underlyingRawResponse = response.raw as AzureUnderlyingRawResponse;
        }

        const metadata = underlyingRawResponse?.response_metadata;
        let confidence: number | null = null;

        if (metadata?.logprobs?.content && Array.isArray(metadata.logprobs.content)) {
            let totalLogProb = 0.0;
            for (const contentItem of metadata.logprobs.content) {
                if (typeof contentItem.logprob === 'number') {
                    totalLogProb += contentItem.logprob;
                }
            }
            if (metadata.logprobs.content.length > 0) {
                confidence = Math.exp(totalLogProb);
            }
        }

        if (responseType === "json") {
            if (!hadJsonParsingError) {
                return {
                    data: parsedJsonFromWrapper,
                    confidence: confidence,
                    prompt_tokens: response.promptTokens,
                    completion_tokens: response.completionTokens,
                };
            } else {
                return {
                    text: String(parsedJsonFromWrapper),
                    confidence: confidence,
                    prompt_tokens: response.promptTokens,
                    completion_tokens: response.completionTokens,
                };
            }
        }

        const textContent = underlyingRawResponse?.content?.trim() || "";
        return {
            text: textContent,
            confidence: confidence,
            prompt_tokens: response.promptTokens,
            completion_tokens: response.completionTokens,
        };
    }
}

export { LlmBaseModelParser, ParsedResult, ResponseType };