import { KnowledgeBaseStore, SharePointConnector } from '@com.cathaypacific.teams.nte/ai-sdk-node';

export const store = new KnowledgeBaseStore({
  host: process.env.ELASTICSEARCH_HOST || '',
  basicAuth: {
    username: process.env.ELASTICSEARCH_USERNAME || '',
    password: process.env.ELASTICSEARCH_PASSWORD || '',
  },
  indexPrefix: process.env.ELASTICSEARCH_INDEX_PREFIX || '',
  instanceConfigs: [
    {
      apiKey: process.env.EMBEDDING_INSTANCE_API_KEY || '',
      azureEndpoint: process.env.EMBEDDING_INSTANCE_AZURE_ENDPOINT || '',
    },
  ],
  modelConfig: {
    azureDeployment: process.env.EMBEDDING_MODEL_AZURE_DEPLOYMENT || '',
    apiVersion: process.env.EMBEDDING_MODEL_API_VERSION || '',
  },
  chunkSize: 3000,
  chunkOverlap: 600,
});

export const connector = new SharePointConnector(
  {
    azure: {
      tenantId: process.env.AZURE_SHAREPOINT_TENANTID || '',
      clientId: process.env.AZURE_SHAREPOINT_CLIENTID || '',
      clientSecret: process.env.AZURE_SHAREPOINT_CLIENTSECRET || '',
    },
    sharepoint: {
      driveId: process.env.AZURE_SHAREPOINT_DRIVEID || '',
      folderId: process.env.AZURE_SHAREPOINT_FOLDERID || '',
    },
  },
  store
);
