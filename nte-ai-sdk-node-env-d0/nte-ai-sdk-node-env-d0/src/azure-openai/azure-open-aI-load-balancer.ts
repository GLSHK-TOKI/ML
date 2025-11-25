import { AzureOpenAI } from "openai";
import { AzureChatOpenAI } from "@langchain/openai";

export class AzureOpenAIEmbeddingLoadBalancer {
    private instances: AzureOpenAI[];

    constructor(instances: AzureOpenAI[]) {
        this.instances = instances;
    }

    public getInstance(): AzureOpenAI {
        if (this.instances.length === 0) {
            throw new Error("No instances available to choose from.");
        }
        const randomIndex = Math.floor(Math.random() * this.instances.length);
        return this.instances[randomIndex];
    }
}

export class AzureChatOpenAILoadBalancer {
    private instances: AzureChatOpenAI[];

    constructor(instances: AzureChatOpenAI[]) {
        this.instances = instances;
    }

    public getInstance(): AzureChatOpenAI {
        if (this.instances.length === 0) {
            throw new Error("No instances available to choose from.");
        }
        const randomIndex = Math.floor(Math.random() * this.instances.length);
        return this.instances[randomIndex];
    }
}