import { client } from './mock-elasticsearch';
import { KnowledgeBaseStore } from '#ai-sdk-node';

export function createMockKnowledgeBaseStore() {
  const store = new KnowledgeBaseStore({
    host: 'https://my-deployment-restore.es.us-east-2.aws.elastic-cloud.com:443',
    basicAuth: {
      username: 'ai-sdk-node-test',
      password: 'dHBSRJYYUbs4k3d',
    },
    indexPrefix: 'ai-sdk-node--andrew',
    instanceConfigs: [
      {
        apiKey: 'dummy',
        azureEndpoint: 'dummy',
      },
    ],
    modelConfig: {
      azureDeployment: 'dummy',
      apiVersion: 'dummy',
    },
    chunkSize: 100,
    chunkOverlap: 20,
  });
  store.es = client;
  return store;
}

export function createDummyKnowledgeBaseStore() {
  const store = new KnowledgeBaseStore({
    host: process.env.DUMMY_ELASTICSEARCH_HOST || '',
    basicAuth: {
      username: process.env.DUMMY_ELASTICSEARCH_USERNAME || '',
      password: process.env.DUMMY_ELASTICSEARCH_PASSWORD || '',
    },
    indexPrefix: process.env.DUMMY_ELASTICSEARCH_INDEX_PREFIX || '',
    instanceConfigs: [
      {
        apiKey: 'dummy',
        azureEndpoint: 'dummy',
      },
    ],
    modelConfig: {
      azureDeployment: 'dummy',
      apiVersion: 'dummy',
    },
    chunkSize: 100, // Small chunk size for testing
    chunkOverlap: 20, // Small chunk overlap for testing
  })
  return store;
}