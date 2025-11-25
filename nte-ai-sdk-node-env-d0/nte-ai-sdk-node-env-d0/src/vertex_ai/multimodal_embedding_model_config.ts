import { BaseModelConfig } from '../model-config.js';

/**
 * VertexAI Multimodal Embedding Model Configuration
 * Specialized configuration for multimodal embedding models
 */
export class VertexAIMultimodalEmbeddingConfig extends BaseModelConfig {
  modelName: string;
  publisher: string;

  constructor(options: { modelName: string}) {
    super();
    this.modelName = options.modelName;
    this.publisher = 'google';
  }
}