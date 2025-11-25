import { PIIDetector, AzureOpenAIInstanceConfig, AzureOpenAIModelConfig } from "@com.cathaypacific.teams.nte/ai-sdk-node";
import dotenv from "dotenv";
import path from "path";

dotenv.config({ path: path.resolve(import.meta.dirname, '.env') });

const piiDetector = new PIIDetector({
  instanceConfigs: [new AzureOpenAIInstanceConfig({
    apiKey: process.env.LLM_INSTANCE_API_KEY!,
    azureEndpoint: process.env.LLM_INSTANCE_AZURE_ENDPOINT!
  })],
  modelConfig: new AzureOpenAIModelConfig({
    azureDeployment: process.env.LLM_MODEL_AZURE_DEPLOYMENT!,
    apiVersion: process.env.LLM_MODEL_API_VERSION!
  }),
  config: {
    categories: [
      { name: "email" },
      { name: "name" },
    ]
  }
});

const result = await piiDetector.detect("John's email is john@gmail.com");
console.log(result);