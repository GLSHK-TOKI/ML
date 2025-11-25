import { VertexAIInstanceConfig } from './instance_config.js';
import { SDKException } from '../exception/index.js';
import { VertexAIMultimodalEmbeddingConfig } from './multimodal_embedding_model_config.js';
import { GoogleAuth } from 'google-auth-library';
import * as aiplatform from '@google-cloud/aiplatform';
/**
 * Vertex AI Embedding Model
 * Unified interface for multimodal embedding using PredictionServiceClient
 */
export class VertexAIMultimodalEmbeddingModel {
  private instanceConfig: VertexAIInstanceConfig;
  private modelConfig: VertexAIMultimodalEmbeddingConfig;
  private predictionServiceClient: aiplatform.v1.PredictionServiceClient | null = null; // Lazy initialization for the client
  private helpers: typeof aiplatform.helpers | null = null; // Lazy initialization for helpers
  private endpoint: string = '';
  private isInitialized: boolean = false;

  constructor(
    instanceConfig: VertexAIInstanceConfig,
    modelConfig: VertexAIMultimodalEmbeddingConfig
  ) {
    this.instanceConfig = instanceConfig;
    this.modelConfig = modelConfig;
  }

  /**
   * Generate embeddings for an image using Vertex AI multimodal embedding
   * @param params Parameters containing image data
   * @returns Promise<number[]> Image embeddings
   */
  async getEmbeddings(params: { image: { bytesBase64Encoded: string } }): Promise<number[]> {
    await this.initialize();

    if (!this.predictionServiceClient || !this.helpers) {
      throw new SDKException(500, "Failed to initialize Vertex AI client or helpers");
    }

    // Validate input parameters
    if (!params.image?.bytesBase64Encoded) {
      throw new SDKException(400, "Missing required parameter: image.bytesBase64Encoded");
    }

    // Validate base64 image data
    this.validateBase64Image(params.image.bytesBase64Encoded);

    // Clean the base64 data (remove data URL prefix if present)
    const cleanBase64 = params.image.bytesBase64Encoded.replace(/^data:image\/[a-zA-Z]*;base64,/, '');

    // Prepare the instance for multimodal embedding - correct format for multimodalembedding@001
    const instance = {
      image: {
        bytesBase64Encoded: cleanBase64
      }
    };

    // Convert instance to protobuf Value using the helpers
    const instanceValue = this.helpers.toValue(instance);

    // Prepare the request with correct structure for PredictionServiceClient
    const request = {
      endpoint: this.endpoint,
      instances: [instanceValue] as aiplatform.protos.google.protobuf.IValue[],
      parameters: this.helpers.toValue({}) as aiplatform.protos.google.protobuf.IValue // Empty parameters for multimodal embedding
    };

    // Make prediction using the PredictionServiceClient
    const [response] = await this.predictionServiceClient.predict(request);

    // Extract embeddings from response - multimodal embedding returns different structure
    if (!response.predictions || response.predictions.length === 0) {
      throw new SDKException(400, "No predictions returned from Vertex AI multimodal embedding");
    }

    const rawPrediction = response.predictions[0];
    // Convert protobuf Value back to JavaScript object
    const prediction = this.helpers.fromValue(rawPrediction as protobuf.common.IValue) as {
      // This return type is not well-documented. prediction.imageEmbedding is observed in practice.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      [key: string]: any
    }; // Note: this cannot be a list of predictions

    // For multimodalembedding@001, the embeddings are in embedding.float_array field
    let embeddings: number[];
    if (prediction.embedding?.float_array) {
      embeddings = prediction.embedding.float_array;
    } else if (prediction.imageEmbedding) {
      embeddings = prediction.imageEmbedding;
    } else if (prediction.embeddings) {
      embeddings = prediction.embeddings;
    } else if (prediction.embedding) {
      embeddings = prediction.embedding;
    } else {
      const predictionKeys = Object.keys(prediction);
      throw new SDKException(400, `No embedding data found in prediction response. Available fields: ${predictionKeys.join(', ')}`);
    }

    if (!embeddings || !Array.isArray(embeddings)) {
      throw new SDKException(400, `Invalid embedding format: expected array, got ${typeof embeddings}`);
    }

    if (embeddings.length === 0) {
      throw new SDKException(400, "Empty embedding array returned");
    }

    return embeddings;
  }


  /**
   * Initialize the multimodal model (lazy initialization)
   */
  private async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }
    // Create prediction service client
    this.predictionServiceClient = await this.createPredictionServiceClient();
    
    // Get helpers for protobuf conversion
    this.helpers = this.getHelpers();
    
    // Get model endpoint
    this.endpoint = this.getModelResourceName(
      {
        project: this.instanceConfig.project,
        location: this.instanceConfig.location,
        modelName: this.modelConfig.modelName,
        publisher: this.modelConfig.publisher
      }
    );

    this.isInitialized = true;
  }

  /**
   * Validate base64 image data
   */
  private validateBase64Image(base64: string): void {
    if (!base64 || typeof base64 !== 'string') {
      throw new SDKException(400, "Image data must be a non-empty string");
    }

    // Remove data URL prefix if present
    const cleanBase64 = base64.replace(/^data:image\/[a-zA-Z]*;base64,/, '');
    
    // Basic base64 validation
    const base64Regex = /^[A-Za-z0-9+/]*={0,2}$/;
    if (!base64Regex.test(cleanBase64)) {
      throw new SDKException(400, "Invalid base64 image format");
    }

    // Check minimum size (should be at least a few KB for a real image)
    if (cleanBase64.length < 100) {
      throw new SDKException(400, "Image data too small - possible corruption");
    }
  }

  /**
   * Initialize authentication credentials using Application Default Credentials
   * TypeScript equivalent of Python's default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
   */
  private async initAuth(): Promise<GoogleAuth> {
    // Use service account credentials from base64-encoded JSON
    const decodedCredentials = Buffer.from(this.instanceConfig.credentials_base64, 'base64').toString('utf-8');
    const auth = new GoogleAuth({
      scopes: ['https://www.googleapis.com/auth/cloud-platform'],
      projectId: this.instanceConfig.project,
      credentials: JSON.parse(decodedCredentials)
    });

    // Get the client and refresh credentials (equivalent to credentials.refresh())
    const authClient = await auth.getClient();
    await authClient.getAccessToken(); // This automatically refreshes if needed

    return auth;
  }

  /**
   * Create PredictionServiceClient with Application Default Credentials
   */
  private async createPredictionServiceClient(): Promise<aiplatform.v1.PredictionServiceClient> {
    const auth = await this.initAuth();

    const clientOptions = {
      apiEndpoint: `${this.instanceConfig.location}-aiplatform.googleapis.com`,
      auth: auth
    };

    return new aiplatform.v1.PredictionServiceClient(clientOptions);
  }

  /**
   * Get the helpers module for protobuf Value conversion
   */
  private getHelpers() {
    return aiplatform.helpers;
  }

  /**
   * Get the full model resource name for Vertex AI
   */
  private getModelResourceName({ project, location, modelName, publisher }: { project: string, location: string, modelName: string, publisher: string }): string {
    return `projects/${project}/locations/${location}/publishers/${publisher}/models/${modelName}`;
  }
}