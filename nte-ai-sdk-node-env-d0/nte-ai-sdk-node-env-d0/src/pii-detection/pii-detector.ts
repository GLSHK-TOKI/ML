import {
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
} from "@langchain/core/prompts";
import { BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { AzureUnderlyingRawResponse, LlmBaseModel, ModelInvokeReturn } from "../llm-base-model.js";
import { AzureOpenAIInstanceConfig, AzureOpenAIModelConfig } from '../azure-openai/index.js'
import { PIIDetectionConfig, PIICategory } from "./pii-detection-config.js";

export interface PIIDetectorOptions {
    instanceConfigs: AzureOpenAIInstanceConfig[],
    modelConfig: AzureOpenAIModelConfig,
    config: PIIDetectionConfig
}

interface PIIDetectorDetectOptions {
    enableLocationMark?: boolean;
}

export interface PIIDetectionResult {
  entity: string;
  text: string;
  location?: number;
}

export interface PIIDetectionResponse {
    data: Array<{ entity: string; text: string; location?: number }>;
    usage: {
        promptTokens: number;
        completionTokens: number;
    };
    confidence?: number | null;
}

export class PIIDetector extends LlmBaseModel {
    piiCategories: PIICategory[];
    /**
     * A class for detecting and handling PII in text.
     * @param instanceConfigs List of instance configurations of llm that used to generate the result.
     * @param modelConfig Model configuration of llm that used to generate the result.
     * @param config PII detection config. Configuration for PII detection, including categories of PII and description to detect.
     */
    constructor(options: PIIDetectorOptions) {
        const { instanceConfigs, modelConfig, config } = options;

        super({ instanceConfigs, modelConfig });
        this.piiCategories = config.categories;
    }

    /**
     * Detect PII in the given text.
     * @param text Text to scan for PII.
     * @param options Options for PII detection.
     * * `enableLocationMark` - Whether to include the location of detected PII in the result. Default is true.
     * @returns A result object with the detected PII items, and additional information.
     */
    public async detect(text: string, options?: PIIDetectorDetectOptions): Promise<PIIDetectionResponse> {
        const enableLocationMark = options?.enableLocationMark ?? true;

        const tokenUsage = {
            promptTokens: 0,
            completionTokens: 0,
        };
        const promptMessages = await this._preparePiiDetectionPrompt(text);
        
        const markdownResponse = await this.modelInvoke(
            promptMessages,
            {
                responseType: 'text',
                skipTokenCount: false,
            }
        ) as ModelInvokeReturn & {
            raw: AzureUnderlyingRawResponse
        };
        const markdownResult = this.parser.parse(markdownResponse);
        tokenUsage.promptTokens += markdownResponse.promptTokens || 0;
        tokenUsage.completionTokens += markdownResponse.completionTokens || 0;

        const markdownText = markdownResult.text;
        if (!markdownText) {
            return {
                data: [],
                usage: tokenUsage,
            }
        }

        const jsonMessage = this._preparePiiMarkdownToJsonPrompt(markdownText);

        const jsonResponse = await this.modelInvoke(
            jsonMessage,
            {
                responseType: 'json',
                responseSchema: {
                    title: "pii_response",
                    description: "The response schema for the pii detection.",
                    type: "object",
                    properties: {
                        pii: {
                            type: "array",
                            items: {
                                type: "object",
                                properties: {
                                    entity: {
                                        type: "string",
                                        description: "The PII category from the markdown table"
                                    },
                                    text: {
                                        type: "string",
                                        description: "The exact raw text from the markdown table"
                                    },
                                },
                                additionalProperties: false,
                                required: ["entity", "text"]
                            },
                        },
                    },
                    required: ["pii"],
                    additionalProperties: false
                },
                skipTokenCount: false
            }
        );
        tokenUsage.promptTokens += markdownResponse.promptTokens || 0;
        tokenUsage.completionTokens += markdownResponse.completionTokens || 0;

        if (!jsonResponse) {
            return {
                data: [],
                usage: tokenUsage,
            }
        }

        const result = this.parser.parse(jsonResponse, "json");
        const piiResultList: Array<{ entity: string; text: string }> = result.data?.pii || [];

        const piiFinalResult: PIIDetectionResult[] = [];
        for (const item of piiResultList) {
            const trimmedText = item.text.trim();
            let regex;

            if (trimmedText.length > 0) {
                // A regex that matches the text, allowing for whitespace variations
                const textParts = trimmedText
                    .split(/\s+/) 
                    .map(part => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')); 

                const patternString = textParts.join('\\s+'); 
                regex = new RegExp(patternString, 'g');
            } else {
                regex = new RegExp('(?!)', 'g'); // A fallback regex that never matches for empty or all-whitespace item.text
            }
            let match;
            if ((match = regex.exec(text)) !== null) {
                piiFinalResult.push({
                    entity: item.entity,
                    text: item.text,
                    location: match.index
                });
            } else {
                piiFinalResult.push({ 
                    entity: item.entity, 
                    text: item.text 
                });
            }
        }

        return {
            data: enableLocationMark ? piiFinalResult : piiResultList,
            usage: tokenUsage,
            confidence: result.confidence
        }
    }

    private async _preparePiiDetectionPrompt(text: string): Promise<BaseMessage[]> {
        const piiCategoriesNameFormatted: string[] = [];
        const piiCategoriesDefinitionFormatted: string[] = [];

        for (const category of this.piiCategories) {
            piiCategoriesNameFormatted.push(`- ${category.name} \n`);
            if (category.definition) {
                const tagName = category.name.toUpperCase().replace(/ /g, '-');
                piiCategoriesDefinitionFormatted.push(`<${tagName}-PATTERN>\n${category.definition}\n</${tagName}-PATTERN>`);
            }
        }

        const namesSection = piiCategoriesNameFormatted.join("");
        const definitionsSection = piiCategoriesDefinitionFormatted.join("\n\n");
        const formattedPiiCat = `${namesSection}\n${definitionsSection}`;

        const piiDetectionSysPromptTemplate = `You are given the <USER-INPUT> that might contain PII or sensitive data listed in the <PII-SENSITIVE-CAT> session. You are asked to detect the PII or sensitive data from it.
<PII-SENSITIVE-CAT>
${'{pii_sensitive_cat}'}
</PII-SENSITIVE-CAT>

<INSTRUCTIONS>
1. Scan the entire <USER-INPUT>, including multiline text, to detect **all** items of PII listed in <PII-SENSITIVE-CAT>.
2. For each potential match, verify that its category is in <PII-SENSITIVE-CAT> before including it in the output.
3. Ignore any data that does not match a category in <PII-SENSITIVE-CAT>, even if it appears to be sensitive.
4. Only report PII that can be directly identified as matching a category in <PII-SENSITIVE-CAT> without requiring contextual inference from surrounding text (e.g., do not combine data with context above or below to infer PII).
5. Preserve the exact text of each detected item as it appears in <USER-INPUT>.
6. Output only the markdown table, with no additional explanations or comments.
</INSTRUCTIONS>

<OUTPUT-FORMAT>
Generate the pairs of the PII and the raw text you found in markdown table
</OUTPUT-FORMAT>`;

        const piiDetectionUserPromptTemplate = `<USER-INPUT>\n${'{user_input}'}\n</USER-INPUT>. Think it step by step`;

        const systemPrompt = SystemMessagePromptTemplate.fromTemplate(piiDetectionSysPromptTemplate);
        const systemMessage = (await systemPrompt.formatMessages({ pii_sensitive_cat: formattedPiiCat }))[0];

        const humanPrompt = HumanMessagePromptTemplate.fromTemplate(piiDetectionUserPromptTemplate);
        const humanMessage = (await humanPrompt.formatMessages({ user_input: text }))[0];
        
        if (!systemMessage || !humanMessage) {
            throw new Error("Failed to format PII detection prompt messages.");
        }

        return [systemMessage, humanMessage];
    }

    private _preparePiiMarkdownToJsonPrompt(markdownResult: string): BaseMessage[] {
        const markdownToJsonPrompt = `You are given the PII and senstive data detection result in Markdown table. Your task is to convert the entire table and process **all** rows into a JSON format. Preserve the exact text as it appears in the Markdown table.
JSON schema below:
pii:[\\{"entity":string, "text": string\\}]`;
        const systemMessage = new SystemMessage(markdownToJsonPrompt);
        const humanMessage = new HumanMessage(markdownResult);
        return [systemMessage, humanMessage];
    }
}