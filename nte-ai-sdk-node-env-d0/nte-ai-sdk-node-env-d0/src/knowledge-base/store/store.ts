import { VectorStore, VectorStoreOptions } from '../../vector-store.js';
import { KnowledgeBaseStoreDocuments } from './documents.js';
import { KnowledgeBaseStoreStates } from './states.js';
import { AzureOpenAIModelConfig } from '../../azure-openai/azure-open-ai-model-config.js'
import { AzureOpenAIInstanceConfig } from '../../azure-openai/azure-open-ai-instance-config.js'
import { AzureOpenAIEmbeddings } from "@langchain/openai";
import { estypes, Client } from '@elastic/elasticsearch';
import { PIIDetector } from '../../pii-detection/index.js';
import { PIIDetectedCallback } from '../../sharepoint-connector.js';
import { VertexAIInstanceConfig, VertexAIMultimodalEmbeddingModel, VertexAIMultimodalEmbeddingConfig } from '../../vertex_ai/index.js';
import { SDKException } from '#ai-sdk-node/exception/index.js';

export interface KnowledgeBaseStoreOptions extends VectorStoreOptions {
  /**
   * The name of the index that stores the documents for the knowledge base.
   */
  indexPrefix: string;

  /**
   * The size of the chunk.
   */
  chunkSize: number;
  /**
   * The size of the chunk overlap.
   */
  chunkOverlap: number;
  /**
   * List of instance configurations of the embedding model to be used for indexing and retrieving documents.
   * This is the new field replacing the deprecated `instanceConfigs`.
   */
  textInstanceConfigs?: AzureOpenAIInstanceConfig[];
  /**
   * Model configuration for the embedding model to be used for indexing and retrieving documents.
   * This is the new field replacing the deprecated `modelConfig`.
   */
  textModelConfig?: AzureOpenAIModelConfig;
  /**
   * (Deprecated) Backward-compatible alias for `textInstanceConfigs`. Use `textInstanceConfigs` instead.
   */
  instanceConfigs?: AzureOpenAIInstanceConfig[];
  /**
   * (Deprecated) Backward-compatible alias for `textModelConfig`. Use `textModelConfig` instead.
   */
  modelConfig?: AzureOpenAIModelConfig;
  /**
   * Multimodal instance configuration for Vertex AI (optional).
   */
  multimodalInstanceConfig?: VertexAIInstanceConfig;
  /**
   * Multimodal model configuration for Vertex AI (optional).
   */
  multimodalModelConfig?: VertexAIMultimodalEmbeddingConfig;
}

export class KnowledgeBaseStore extends VectorStore {
  indexPrefix: string;
  chunkSize: number;
  chunkOverlap: number;
  textInstanceConfigs: AzureOpenAIInstanceConfig[];
  textModelConfig: AzureOpenAIModelConfig;
  textEmbeddingModels: AzureOpenAIEmbeddings[];
  multimodalEmbeddingModel?: VertexAIMultimodalEmbeddingModel;
  piiDetector?: PIIDetector;
  piiDetectedCallback?: PIIDetectedCallback;

  /**
   * Interacting with the knowledge base in Elasticsearch.
   * 
   * @param options.host The host of the Elasticsearch instance.
   * @param options.basicAuth The basic auth credentials for the Elasticsearch instance.
   * @param options.indexPrefix The name of the index that stores the documents for the knowledge base.
   * @param options.textInstanceConfigs List of instance configurations of the embedding model to be used for indexing and retrieving documents.
   * @param options.textModelConfig Model configuration for the embedding model to be used for indexing and retrieving documents. 
   * @param options.multimodalInstanceConfig Multimodal instance configuration for Vertex AI (optional).
   * @param options.multimodalModelConfig Multimodal model configuration for Vertex AI (optional).
   */
  constructor(options: KnowledgeBaseStoreOptions) {
    super(options);

    this.indexPrefix = options.indexPrefix;   
    this.chunkSize = options.chunkSize || 1000;
    this.chunkOverlap = options.chunkOverlap || 200;

    const { textInstanceConfigs, textModelConfig } = this.normalizeTextEmbeddingModelOptions(options);
    this.textInstanceConfigs = textInstanceConfigs;
    this.textModelConfig = textModelConfig;
  
    this.textEmbeddingModels = this.initTextEmbeddingModelInstances()
    
    // Optional: Initialize multimodal embedding model from instance and model configs
    if (options.multimodalInstanceConfig && options.multimodalModelConfig) {
      this.multimodalEmbeddingModel = this.initMultimodalEmbeddingModel(
        options.multimodalInstanceConfig,
        options.multimodalModelConfig
      );
    }
  }

  documents = new KnowledgeBaseStoreDocuments(this);
  states = new KnowledgeBaseStoreStates(this);

  /**
   * Get the Elasticsearch client instance
   * @returns The Elasticsearch client
   */
  public getClient(): Client {
    return this.es;
  }

  getInstance(embeddingModels: AzureOpenAIEmbeddings[]) {
    const randomIndex = Math.floor(Math.random() * embeddingModels.length);
    const selectedInstance = embeddingModels[randomIndex];
    return selectedInstance
  }

  private initTextEmbeddingModelInstances() {
    const models = [];
    for (const instanceConfig of this.textInstanceConfigs) {
      models.push(new AzureOpenAIEmbeddings({
          azureOpenAIApiKey: instanceConfig.apiKey,
          azureOpenAIApiInstanceName: instanceConfig.azureEndpoint,
          azureOpenAIApiEmbeddingsDeploymentName: this.textModelConfig.azureDeployment,
          azureOpenAIApiVersion: this.textModelConfig.apiVersion
      }));
    }
    return models;
  }

  /**
   * Initialize multimodal embedding model from instance and model configs
   */
  private initMultimodalEmbeddingModel(
    instanceConfig: VertexAIInstanceConfig,
    modelConfig: VertexAIMultimodalEmbeddingConfig
  ): VertexAIMultimodalEmbeddingModel {
    if (!instanceConfig.project) {
      throw new SDKException(400, "Missing required instanceConfig.project for multimodal embedding model");
    }
    if (!instanceConfig.location) {
      throw new SDKException(400, "Missing required instanceConfig.location for multimodal embedding model");
    }
    if (!instanceConfig.credentials_base64) {
      throw new SDKException(400, "Missing required instanceConfig.credentials_base64 for multimodal embedding model");
    }
    if (!modelConfig.modelName) {
      throw new SDKException(400, "Missing required modelConfig.modelName for multimodal embedding model");
    }
    return new VertexAIMultimodalEmbeddingModel(instanceConfig, modelConfig);
  }

  private normalizeTextEmbeddingModelOptions(options: KnowledgeBaseStoreOptions) {
    if (options.instanceConfigs && !options.textInstanceConfigs) {
      console.warn("Warning: 'instanceConfigs' is deprecated. Please use 'textInstanceConfigs' for text embedding model instead.");
      options.textInstanceConfigs = options.instanceConfigs;
    }
    if (options.modelConfig && !options.textModelConfig) {
      console.warn("Warning: 'modelConfig' is deprecated. Please use 'textModelConfig' for text embedding model instead.");
      options.textModelConfig = options.modelConfig;
    }
    if (!options.textInstanceConfigs || options.textInstanceConfigs.length === 0) {
      throw new SDKException(400, "At least one text embedding model instance configuration must be provided.");
    }
    if (!options.textModelConfig) {
      throw new SDKException(400, "Text embedding model configuration must be provided.");
    }
    return {
      textInstanceConfigs: options.textInstanceConfigs,
      textModelConfig: options.textModelConfig
    }
  }
}

export interface SearchHitWithSource<T> extends estypes.SearchHit<T> {
  _source: T;
}