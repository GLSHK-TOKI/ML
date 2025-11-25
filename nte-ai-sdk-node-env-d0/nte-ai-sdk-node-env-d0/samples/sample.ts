import {  PIIDetector, KnowledgeBaseStore, SharePointConnector, AzureOpenAIInstanceConfig, AzureOpenAIModelConfig } from "@com.cathaypacific.teams.nte/ai-sdk-node";
import dotenv from "dotenv";
import path from "path";

dotenv.config({ path: path.resolve(import.meta.dirname, '.env') });

const store = new KnowledgeBaseStore(
  {
    host: process.env.ELASTICSEARCH_HOST!,
    basicAuth: {
      username: process.env.ELASTICSEARCH_USERNAME!,
      password: process.env.ELASTICSEARCH_PASSWORD!
    },
    indexPrefix: process.env.ELASTICSEARCH_INDEX_PREFIX!,
    instanceConfigs: [{
      apiKey: process.env.EMBEDDING_INSTANCE_API_KEY!,
      azureEndpoint: process.env.EMBEDDING_INSTANCE_AZURE_ENDPOINT!
    }],
    modelConfig: {
      azureDeployment: process.env.EMBEDDING_MODEL_AZURE_DEPLOYMENT!,
      apiVersion: process.env.EMBEDDING_MODEL_API_VERSION!
    },
    chunkSize: 3000,
    chunkOverlap: 600
  }
);
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

const connector = new SharePointConnector(
  {
    azure: {
      tenantId: process.env.AZURE_SHAREPOINT_TENANTID!,
      clientId: process.env.AZURE_SHAREPOINT_CLIENTID!,
      clientSecret: process.env.AZURE_SHAREPOINT_CLIENTSECRET!,
    },
    sharepoint: {
      driveId: process.env.AZURE_SHAREPOINT_DRIVEID!,
      folderId: process.env.AZURE_SHAREPOINT_FOLDERID!,
    },
    webLinkExpireInterval: 432000,
  },
  store
);
connector.add_pii_detector(piiDetector)
connector.add_pii_detected_callback((piiResults, document, chunkContent) => {
  console.log({
    piiResults,
    document,
    chunkContent
  })
})
const result = await connector.run();
console.log(result);