import {BaseModelConfig} from "./../model-config.js";
export class AzureOpenAIModelConfig extends BaseModelConfig {
    azureDeployment: string;
    apiVersion: string

    constructor(options: { azureDeployment: string, apiVersion: string }) {
        const { azureDeployment, apiVersion } = options;
        super();
        this.azureDeployment = azureDeployment;
        this.apiVersion = apiVersion;
    }
}

export class AzureOpenAIReasoningModelConfig extends AzureOpenAIModelConfig {
    
    constructor(options: { azureDeployment: string, apiVersion: string }) {
        super(options);
    }
}