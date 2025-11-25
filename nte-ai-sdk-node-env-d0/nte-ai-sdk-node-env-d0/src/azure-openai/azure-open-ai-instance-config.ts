import  {BaseInstanceConfig} from '../instance-config.js';

export class AzureOpenAIInstanceConfig extends BaseInstanceConfig {
    azureEndpoint: string;
    apiKey: string;

    constructor(options: { azureEndpoint: string; apiKey: string }) {
        const { azureEndpoint, apiKey } = options;
        super();
        this.azureEndpoint = azureEndpoint;
        this.apiKey = apiKey;
    }
}