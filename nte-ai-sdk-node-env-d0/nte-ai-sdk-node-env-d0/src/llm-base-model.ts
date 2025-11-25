import { BaseMessage } from '@langchain/core/messages';
import { AzureChatOpenAI } from '@langchain/openai';
import {
    AzureChatOpenAILoadBalancer,
    AzureOpenAIInstanceConfig,
    AzureOpenAIModelConfig,
    AzureOpenAIReasoningModelConfig,
} from './azure-openai/index.js';
import { LlmError, handleLlmOperation } from './exception/index.js';
import { BaseInstanceConfig } from './instance-config.js';
import { BaseModelConfig } from './model-config.js';
import { num_tokens_from_messages } from './utils/index.js';
import { LlmBaseModelParser } from './_llm-base-model-parser.js';

interface LlmBaseModelOptions {
    instanceConfigs: BaseInstanceConfig[];
    modelConfig: BaseModelConfig;
}

interface AzureLogProbsContent {
    logprob: number;
}

interface AzureResponseMetadata {
    logprobs?: {
        content: AzureLogProbsContent[];
    };
}

export interface AzureUnderlyingRawResponse {
    content: string;
    response_metadata: AzureResponseMetadata;
}

export interface AzureLangchainJsonParsingWrapper {
    raw: AzureUnderlyingRawResponse;
    parsed: any;
    parsing_error: boolean;
}

export interface ModelInvokeReturn {
    raw: AzureLangchainJsonParsingWrapper | AzureUnderlyingRawResponse;
    promptTokens: number;
    completionTokens: number;
}

export interface ModelInvokeOptions {
    responseType?: 'text' | 'json';
    responseSchema?: object | null;
    skipTokenCount?: boolean;
}

export class LlmBaseModel {
    instanceConfigs: BaseInstanceConfig[];
    modelConfig: BaseModelConfig;
    parser: LlmBaseModelParser;
    lb: AzureChatOpenAILoadBalancer;

    constructor(options: LlmBaseModelOptions) {
        const { instanceConfigs, modelConfig } = options;

        this.instanceConfigs = instanceConfigs;
        this.modelConfig = modelConfig;
        this.parser = new LlmBaseModelParser(this);
        if (this.modelConfig instanceof AzureOpenAIModelConfig && 
            this.isAzureInstanceList(this.instanceConfigs)) {
            this.lb = this.initAzureModel(
                this.instanceConfigs as AzureOpenAIInstanceConfig[], 
                this.modelConfig
            );
        } else {
            throw new Error("Invalid configuration: Expected AzureOpenAI configuration types");
        }
    }

    public async modelInvoke(
        messages: BaseMessage[],
        options?: ModelInvokeOptions
    ): Promise<ModelInvokeReturn | null> {
        // Set defaults
        const responseType = options?.responseType ?? 'text';
        const responseSchema = options?.responseSchema ?? null;
        const skipTokenCount = options?.skipTokenCount ?? false;

        if (this.modelConfig instanceof AzureOpenAIModelConfig) {
            const promptTokens = skipTokenCount ? 0 : num_tokens_from_messages(messages);
            const llm = this.lb.getInstance();
            let aiMessage: any;
            let finalResponsePayload: any;

            if (responseType === 'json') {
                if (!responseSchema) {
                    throw new LlmError(400, "responseSchema is required for json responseType.");
                }
                
                const structuredLlm = llm.withStructuredOutput(responseSchema as Record<string, any>, {
                    name: 'json_schema',
                    method: "functionCalling",
                    includeRaw: true
                });
                
                const structuredInvocationResult = await handleLlmOperation(
                    (messages: any) => structuredLlm.invoke(messages)
                )(messages);
                
                aiMessage = structuredInvocationResult.raw;
                finalResponsePayload = structuredInvocationResult;
            } else {
                const directInvocationResult = await handleLlmOperation(
                    (messages: any) => llm.invoke(messages)
                )(messages);
                
                aiMessage = directInvocationResult;
                finalResponsePayload = directInvocationResult;
            }

            const completionTokens = skipTokenCount ? 0 : num_tokens_from_messages([aiMessage]);
            return {
                raw: finalResponsePayload,
                promptTokens: promptTokens,
                completionTokens: completionTokens
            };
        }
        return null;
    }

    private initAzureModel(instanceConfigs: AzureOpenAIInstanceConfig[], modelConfig: AzureOpenAIModelConfig) {
        const models = instanceConfigs.map(instanceConfig => {
            if (modelConfig instanceof AzureOpenAIReasoningModelConfig) {
                return new AzureChatOpenAI({
                    azureOpenAIApiInstanceName: instanceConfig.azureEndpoint,
                    azureOpenAIApiDeploymentName: modelConfig.azureDeployment,
                    azureOpenAIApiKey: instanceConfig.apiKey,
                    azureOpenAIApiVersion: modelConfig.apiVersion
                });
            } else {
                return new AzureChatOpenAI({
                    azureOpenAIApiInstanceName: instanceConfig.azureEndpoint,
                    azureOpenAIApiDeploymentName: modelConfig.azureDeployment,
                    azureOpenAIApiKey: instanceConfig.apiKey,
                    azureOpenAIApiVersion: modelConfig.apiVersion,
                    temperature: 0,
                    logprobs: true
                });
            }
        });

        return new AzureChatOpenAILoadBalancer(models);
    }

    private isAzureInstanceList(lst: any[]): lst is AzureOpenAIInstanceConfig[] {
        return lst.every(x => x instanceof AzureOpenAIInstanceConfig);
    }
}