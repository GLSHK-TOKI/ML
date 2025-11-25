import { estypes } from '@elastic/elasticsearch';
import { KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX } from '../_constants.js';
import { KnowledgeBaseDocument } from '../document/index.js';
import { KnowledgeBasePDFDocument } from "../document/pdf-document.js";
import { KnowledgeBaseImageDocument } from "../document/image-document.js";
import type { KnowledgeBaseStore } from './store.js';
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { get_encoding } from "tiktoken";
import * as Jimp from 'jimp';
import { handleElasticsearchOperation, handleElasticsearchBulkOperation, handleLlmOperation, ElasticsearchApiError, SDKException } from '../../exception/index.js'
import logger from "../../logger/logger.js";
import { SearchHitWithSource } from './store.js';
import { KnowledgeBaseDocumentMetadata } from '../document/document.js';
import _ from 'lodash';
import fs from 'fs';
import { DriveItem } from "@microsoft/msgraph-sdk/models/index.js";
const _log = logger.child({ module: 'ai-sdk-node.knowledge-base.store.documents' });

export class KnowledgeBaseStoreDocuments {
  protected store: KnowledgeBaseStore;

  constructor(store: KnowledgeBaseStore) {
    this.store = store;
  }

  /**
   * Index or update chunks of a document in the knowledge base.
   * 
   * @param document The document to be indexed
   * @returns 
   */
  async upsert(document: KnowledgeBaseDocument) {
    // 1. Chunking the document content
    // 2. Retrieve all chunks from documents index with same id
    // 3. Diff the chunks with the new chunks by content
    // 4. Create embedding and upsert the new chunks
    // 5. Delete the chunks that are not in the new chunks
    // 6. If renamed or collection renamed, update the fields of the unchanged chunks
    // 7. Execute bulk operation
    // 8. Log debug messages for the bulk operations

    _log.debug({
      msg: `Starting upsert operation. Folder [ ${document.getCollection()} ] - File [ ${(await document.getMetadata()).title} ]`,
      status_code: 200
    })

    // 1. Chunking with using Langchain Chunk package
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize : this.store.chunkSize,
      chunkOverlap: this.store.chunkOverlap,
      separators :[
          "\n\n",
          "\n",
          " ",
          ".",
          ",",
          "\u200b",  // Zero-width space
          "\uff0c",  // Fullwidth comma
          "\u3001",  // Ideographic comma
          "\uff0e",  // Fullwidth full stop
          "\u3002",  // Ideographic full stop
          "",
      ],
      keepSeparator: false
    });
    const chunksContent = await splitter.splitText(await document.getContent());
  
    // 2. Retrieve all indexed chunks from documents index with same id
    const indexedChunksResponse: estypes.SearchResponse = await handleElasticsearchOperation((params: estypes.SearchRequest) => this.store.es.search<KnowledgeBaseStoreDocument>(params))({
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
      query: {
        match: {
          id: document.getDocId()
        }
      },
      size: 9999
    });

    // Edge case: If the number of indexed chunks is more than 9999, delete all the chunks and re-index the document
    if (typeof indexedChunksResponse.hits.total === 'number' && indexedChunksResponse.hits.total > 9999) {
      await this.delete(document.getDocId());
      indexedChunksResponse.hits.hits = [];
    }
    const indexedChunks = indexedChunksResponse.hits.hits.filter((hit) => hit._source) as SearchHitWithSource<KnowledgeBaseStoreDocument>[];

    // 3. Diff the chunks with the new chunks by content
    const chunksContentSet = new Set(chunksContent);
    const indexedChunksContentSet = new Set(indexedChunks.map((indexedChunk) => indexedChunk._source?.content));

    const newChunksContent = chunksContent.filter((chunk) => {
      return !indexedChunksContentSet.has(chunk);
    });
    const removedChunks = indexedChunks.filter((indexedChunk) => {
      return !chunksContentSet.has(indexedChunk._source.content);
    });
    const unchangedChunks = indexedChunks.filter((indexedChunk) => {
      return chunksContentSet.has(indexedChunk._source.content);
    });
    const operations = {
      index: [] as (estypes.BulkOperationContainer | estypes.BulkUpdateAction | KnowledgeBaseStoreDocument)[],
      update: [] as (estypes.BulkOperationContainer | estypes.BulkUpdateAction | KnowledgeBaseStoreDocument)[],
      delete: [] as (estypes.BulkOperationContainer | estypes.BulkUpdateAction)[],
    };

    // 4. Create embedding for the new chunks
    for (const chunkContent of newChunksContent) {
      // PII Detection - if detector is configured
      if (this.store.piiDetector) {
        try {
          const piiResult = await this.store.piiDetector.detect(chunkContent, { enableLocationMark: true });
          const piiResultItems = piiResult.data || [];
          
          if (piiResultItems.length > 0) {
            _log.warn({
              msg: `PII detected in chunk. Document: ${document.getDocId()}, Results count: ${piiResultItems.length}`,
              piiResults: piiResultItems.map(entity => ({ type: entity.entity, text: entity.text }))
            });

            // Execute user-defined callback if provided
            if (this.store.piiDetectedCallback) {
              await this.store.piiDetectedCallback(piiResultItems, document, chunkContent);
            }
          }
        } catch (error) {
          _log.error({
            msg: `PII detection failed for chunk in document ${document.getDocId()}`,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }

      // 1. Get number of tokens
      const nToken = this.getNoOfToken(chunkContent);
      // 2. Create embedding with OpenAI embedding model
      const lbEmbeddingModel = this.store.getInstance(this.store.textEmbeddingModels)
      const singleVector = await handleLlmOperation((text: string) => lbEmbeddingModel.embedQuery(text))(chunkContent);
      const currentTime = new Date();

      // 3. Insert the embedding and chunk to elastic search
      operations.index.push({
        index: {
          _index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
        }
      })
      let metadata: KnowledgeBaseDocumentMetadata
      if (document instanceof KnowledgeBasePDFDocument) {
        metadata = await document.getMetadata(chunkContent)
      } else {
        metadata = await document.getMetadata()
      }
      operations.index.push({
        embeddings: singleVector,
        chunk_type: "text",
        id: document.getDocId(),
        content: chunkContent,
        collection: document.getCollection(),
        parentId: document.getParentId(),
        n_token: nToken,
        last_updated_time: new Date(currentTime.getTime()).toISOString(),
        meta: metadata,
      })
    }

    // 5. Delete the chunks that are not in the new chunks
    for (const removedChunk of removedChunks) {
      if (!removedChunk._id) continue;

      operations.delete.push({
        delete: {
          _index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
          _id: removedChunk._id
        }
      })
    }

    // 6. If renamed or collection renamed, update the fields of the unchanged chunks
    const hasCollectionChanged = document.getCollection() !== unchangedChunks[0]?._source.collection;
    const hasParentIdChanged = document.getParentId() !== unchangedChunks[0]?._source.parentId;
    const hasMetadataChanged = _.isEqual(await document.getMetadata(), unchangedChunks[0]?._source.meta);
    const hasChanged = hasCollectionChanged || hasParentIdChanged || !hasMetadataChanged;
    
    if (hasChanged) {
      for (const unchangedChunk of unchangedChunks) {
        if (!unchangedChunk._id) continue;

        operations.update.push({
          update: {
            _index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
            _id: unchangedChunk._id
          }
        })
        operations.update.push({
          doc: {
            collection: document.getCollection(),
            parentId: document.getParentId(),
            meta: await document.getMetadata(),
            last_updated_time: new Date().toISOString(),
          }
        })
      }
    }

    // 7. Execute bulk operations
    if (operations.index.length === 0 && operations.update.length === 0 && operations.delete.length === 0) {
      const msg = `No operations to be executed. Folder [ ${document.getCollection()} ] - File [ ${(await document.getMetadata()).title} ]`;
      _log.debug({ msg: msg, status_code: 200 })
      return null;
    }

    const bulkResponseItems = [] as Partial<Record<estypes.BulkOperationType, estypes.BulkResponseItem>>[];
    for (const key in operations) {
      const operationsChunks = _.chunk(operations[key as keyof typeof operations], 50); // Further chunk the operations to avoid timeout
      for (const operations of operationsChunks) {
        const response: estypes.BulkResponse = await handleElasticsearchBulkOperation(
          (params: estypes.BulkRequest) => this.store.es.bulk(params)
        )({ operations });
        bulkResponseItems.push(...response.items);
      }
    }

    // 8. Log debug messages for the bulk operations
    const indexCount = bulkResponseItems.filter((item) => item.index && item.index.result === 'created').length;
    if (indexCount > 0) {
      _log.debug({
        msg: `Successfully upserted ${indexCount} document chunk record(s). Folder [ ${document.getCollection()} ] - File [ ${(await document.getMetadata()).title} ]`,
        status_code: 201
      })
    }
    const deleteCount = bulkResponseItems.filter((item) => item.delete).length;
    if (deleteCount > 0) {
      _log.debug({
        msg: `Successfully deleted ${deleteCount} document chunk record(s). Folder [ ${document.getCollection()} ] - File [ ${(await document.getMetadata()).title} ]`,
        status_code: 200
      })
    }
    const updateCount = bulkResponseItems.filter((item) => item.update).length;
    if (updateCount > 0) {
      _log.debug({
        msg: `Successfully updated properties for ${updateCount} document chunk record(s). Folder [ ${document.getCollection()} ] - File [ ${(await document.getMetadata()).title} ]`,
        status_code: 200
      })
    }

    return bulkResponseItems;
  }

  async deleteByCollection(parentId: string): Promise<estypes.DeleteByQueryResponse> {
    /**
     * Delete a record of parent key ID from document index
     **/
    const deleteResponse: estypes.DeleteByQueryResponse = await handleElasticsearchOperation(
      (params: estypes.DeleteByQueryRequest) => this.store.es.deleteByQuery(params)
    )({
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
      query: {
        match: {
          parentId: parentId,
        },
      },
    });

    const deletedCount: number = deleteResponse.deleted ?? 0;

    if (deletedCount > 0) {
      const msg = `Successfully delete the document records. Folder ID [ ${parentId} ] ${deletedCount} chunk records are deleted.`;
      _log.debug({msg: msg, status_code: 200})
    } else {
      _log.debug({
        msg: `No chunk records are deleted. Folder ID [ ${parentId} ]`,
        status_code: 200,
      });
    }

    if (deleteResponse.failures && deleteResponse.failures.length > 0) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      deleteResponse.failures.forEach((failure:any) => {
        const msg = `Failed to delete folder record: ${failure}. Folder ID [ ${parentId}`;
        _log.error({msg: msg, status_code: 400})
      });
      const msg = `Failed to delete folder record. Folder ID [ ${parentId}`;
      throw new ElasticsearchApiError(400, msg);
    }

    return deleteResponse;
  }

  async delete(id: string): Promise<estypes.DeleteByQueryResponse> {
    /**
     * Delete a document (with all its chunks) by id from document index
     * @param id: The id of the document to be delete.
     * @return: The response from the elastic search.
     **/
    const deleteResponse: estypes.DeleteByQueryResponse = await handleElasticsearchOperation(
      (params: estypes.DeleteByQueryRequest) => this.store.es.deleteByQuery(params)
    )({
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
      query: {
        match: {
          id: id,
        },
      },
    });
    const deletedCount: number = deleteResponse.deleted ?? 0;

    if (deletedCount > 0) {
      const msg = `Successfully delete the document record. Document ID [ ${id} ] ${deletedCount} chunk records are deleted.`;
      _log.debug({msg: msg, status_code: 200})
    } else {
      _log.debug({
        msg: `No chunk records are deleted. Document ID [ ${id} ]`,
        status_code: 200
      });
    }
    if (deleteResponse.failures && deleteResponse.failures.length > 0) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      deleteResponse.failures.forEach((failure:any) => {
        const msg = `Failed to delete document record: ${failure}. Document ID [ ${id} ]`;
        _log.error({msg: msg, status_code: 400})
      });
      const msg = `Failed to delete document record. Document ID [ ${id} ]`;
      throw new ElasticsearchApiError(400, msg);
    }

    return deleteResponse;
  }

  async renameCollection(parentId: string, newName: string, newFolderWebUrl: string) {
    /**
     * Rename a collection(folder) name of the documents
     **/
    const updateResponse = await this.store.es.updateByQuery({
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
      body: {
        script: {
          source: `
            ctx._source.collection = params.newName;
            // Get the filename by finding the last part after "/"
            def oldUrl = ctx._source.meta.webUrl;
            def fileName = oldUrl.substring(oldUrl.lastIndexOf('/') + 1);
            // Combine new folder URL with original filename
            ctx._source.meta.webUrl = params.newFolderWebUrl + '/' + fileName
          `,
          params : {
            newName : newName,
            newFolderWebUrl : newFolderWebUrl
          }
        },
        query: {
          match: {
            parentId: parentId
          }
        },
      }
    });

    if (updateResponse.failures && updateResponse.failures.length > 0) {
      const msg = `Failed to rename collection. Folder ID [ ${parentId} ] - New Name [ ${newName} ]`;
      _log.error({ msg: msg, status_code: 500 });
    } else if (updateResponse.updated) {
      const msg = `Successfully to rename collection name for ${updateResponse.updated} record(s). Folder ID [ ${parentId} ] - New Name [ ${newName} ]`;
      _log.debug({ msg: msg, status_code: 200 });
    }

    return updateResponse;
  }

  /**
   * Process document images and upsert multimodal embeddings to Elasticsearch
   * Handles both PDF documents (extracts page images) and raw image files
   *
   * @param document The document to process
   */
  async upsertImage(document: KnowledgeBaseDocument) {
    // Type guard: check if document supports image extraction
    const isImageDocument = (doc: KnowledgeBaseDocument): doc is KnowledgeBasePDFDocument | KnowledgeBaseImageDocument => {
      return 'extractPath' in doc && typeof doc.extractPath === 'function';
    };

    if (!isImageDocument(document)) {
      _log.debug({
        msg: `Document type does not support image extraction. Skipping image processing for ${document.doc.name}`,
        status_code: 200
      });
      return;
    }    // Declare allFilePaths outside try block for cleanup access
    let allFilePaths: string[] = [];

    try {
      // Extract images from the document (handles download and extraction)
      allFilePaths = await document.extractPath();

      if (allFilePaths.length === 0) {
        return;
      }

      // Determine if this is a raw image document
      const isRawImage = document instanceof KnowledgeBaseImageDocument;

      // Upsert image embeddings with multimodal model
      return await this.upsertImageEmbeddings(
        allFilePaths,
        document,
        isRawImage
      );

    } finally {
      // Clean up extracted image files
      for (const filePath of allFilePaths) {
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      }
    }
  }

  /**
   * Upsert image embeddings to the knowledge base
   * 
   * @param imagePaths Array of image file paths to process
   * @param document The PDF document or raw image document these images belong to
   * @param isRawImage Whether this is a raw image file (true = delete all and recreate, false = diff PDF pages)
   * @returns Promise with processing results
   */
  async upsertImageEmbeddings(
    imagePaths: string[],
    document: KnowledgeBasePDFDocument | KnowledgeBaseImageDocument,
    isRawImage: boolean = false
  ){
    if (!this.store.multimodalEmbeddingModel) {
      _log.warn({ 
        msg: `Multimodal embedding model not configured. Skipping image processing for ${document.doc.name}`,
        status_code: 400 
      });
      throw new SDKException(400, "Multimodal embedding model not configured");
    }

    const collection = document.getCollection();
    const parentId = document.getParentId();
    const documentId = document.getDocId();
    const documentWebUrl = document.doc.webUrl || '';
    const multimodalEmbeddingModel = this.store.multimodalEmbeddingModel;

    const operations = {
      index: [] as (estypes.BulkOperationContainer | KnowledgeBaseStoreDocument)[],
      delete: [] as estypes.BulkOperationContainer[],
      update: [] as (estypes.BulkOperationContainer | estypes.BulkUpdateAction<KnowledgeBaseStoreDocument>)[]
    };

    // Get existing image embeddings for this document to handle updates/deletes (similar to text upsert logic)
    const existingImagesResponse = await handleElasticsearchOperation(
      (params: estypes.SearchRequest) => this.store.es.search(params)
    )({
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
      query: {
        bool: {
          must: [
            { term: { chunk_type: "image" } },
            { term: { 'id.keyword': documentId } }
          ]
        }
      },
      size: 9999
    });

    const existingImages = existingImagesResponse.hits.hits as SearchHitWithSource<KnowledgeBaseStoreDocument>[];
    
    _log.debug({
      msg: `Raw image upsert: Found ${existingImages.length} existing embedding(s) for document ID [ ${documentId} ]. IsRawImage: ${isRawImage}`,
      status_code: 200
    });
    
    // Handle deletion logic based on document type
    let imagesToDelete: SearchHitWithSource<KnowledgeBaseStoreDocument>[];
    let newImagePaths_filtered: string[];
    
    if (isRawImage) {
      // For raw images: Only create if not exist (file has single image, no pages)
      // If the file content changed, SharePoint connector already handles deletion
      // So we just need to check if embedding exists and if webUrl/title changed (rename)
      if (existingImages.length > 0) {
        // Check if the image was renamed (webUrl or title changed)
        const existingWebUrl = existingImages[0]._source.meta?.webUrl || '';
        const hasWebUrlChanged = existingWebUrl !== documentWebUrl;
        
        if (hasWebUrlChanged) {
          // Update the metadata for the existing image embedding
          for (const existingImage of existingImages) {
            if (!existingImage._id) continue;
            
            operations.update.push({
              update: {
                _index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
                _id: existingImage._id
              }
            });
            
            operations.update.push({
              doc: {
                meta: {
                  title: documentWebUrl.substring(documentWebUrl.lastIndexOf('/') + 1),
                  webUrl: documentWebUrl,
                  startPage: existingImage._source.meta?.startPage || 1,
                  endPage: existingImage._source.meta?.endPage || 1
                }
              }
            });
          }
        }
        // Image embedding already exists - skip creating new embeddings
        imagesToDelete = [];
        newImagePaths_filtered = [];
      } else {
        // No existing embedding - create new one
        imagesToDelete = [];
        newImagePaths_filtered = imagePaths;
      }
    } else {
      // For PDFs: diff individual pages (only delete/create changed pages)
      const newImagePaths = new Set(imagePaths);
      const existingImagePaths = new Set(
        existingImages
          .map(img => img._source.meta?.webUrl?.split('_page_')[0] + '_page_' + (img._source.meta?.startPage || 0) + '.jpg')
          .filter(path => path !== undefined)
      );

      // Find images to delete (exist in ES but not in new list)
      imagesToDelete = existingImages.filter(img => {
        const imagePath = img._source.meta?.webUrl?.split('_page_')[0] + '_page_' + (img._source.meta?.startPage || 0) + '.jpg';
        return imagePath && !newImagePaths.has(imagePath);
      });

      // Find new images to process (not in existing list)
      newImagePaths_filtered = imagePaths.filter(imgPath => {
        const filename = imgPath.split(/[\\/]/).pop() || '';
        return !existingImagePaths.has(filename);
      });
    }

    // Early return if no operations needed (similar to text upsert logic)
    if (newImagePaths_filtered.length === 0 && imagesToDelete.length === 0 && operations.update.length === 0) {
      _log.debug({
        msg: `No image operations to be executed. Document ID [ ${documentId} ]`,
        status_code: 200
      });
      return [];
    }

    // Process only new images for embeddings (similar to text chunk processing)
    for (let imgIdx = 0; imgIdx < newImagePaths_filtered.length; imgIdx++) {
      const imgPath = newImagePaths_filtered[imgIdx];
      
      // Check if image file exists
      if (!fs.existsSync(imgPath)) {
        _log.warn({ msg: `Image ${imgPath} not found!` });
        continue;
      }

      // Get image embeddings using multimodal model
      let imageEmbedding: number[];
      if (multimodalEmbeddingModel.getEmbeddings) {
        // Use the new VertexAI multimodal model with proper base64 interface
        const imageBuffer = fs.readFileSync(imgPath);
        const base64Image = imageBuffer.toString('base64');
        imageEmbedding = await handleLlmOperation(
          (base64: string) => multimodalEmbeddingModel.getEmbeddings({
            image: {
              bytesBase64Encoded: base64
            }
          })
        )(base64Image);
      } else {
        throw new Error("Multimodal embedding model does not have compatible interface");
      }

      if (!imageEmbedding || imageEmbedding.length === 0) {
        continue;
      }

      // Extract metadata from image path
      const pathParts = imgPath.split(/[\\/]/);
      const filename = pathParts[pathParts.length - 1];

      // Simple page number extraction from filename
      // Expected format: pdf_name_page_XXX.jpg
      const pageMatch = filename.match(/_page_(\d+)\./);
      const pageNum = pageMatch ? parseInt(pageMatch[1]) : 0;

      // Get base64 representation using static method from KnowledgeBasePDFDocument
      let base64Image = '';
      try {
        const imageBuffer = fs.readFileSync(imgPath);
        base64Image = await KnowledgeBasePDFDocument.base64FromBufferStatic(imageBuffer, 'jpg');
      } catch (base64Error) {
        // Try fallback base64 conversion without Jimp processing
        const imageBuffer = fs.readFileSync(imgPath);
        base64Image = `data:image/jpeg;base64,${imageBuffer.toString('base64')}`;
        _log.warn({ msg: `Base64 conversion with Jimp failed for ${imgPath}. Used fallback method. Error: ${base64Error}` });
      }

      const currentTime = new Date();

      // Prepare document for Elasticsearch - using correct bulk operation structure
      operations.index.push({
        index: {
          _index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
        }
      });

      operations.index.push({
        image_embedding: imageEmbedding,
        chunk_type: "image",
        id: documentId,
        collection: collection,
        parentId: parentId,
        content: base64Image,
        n_token: await this.getImageTokenCount(base64Image),
        last_updated_time: currentTime.toISOString(),
        meta: {
          title: documentWebUrl.substring(documentWebUrl.lastIndexOf('/') + 1),
          webUrl: documentWebUrl,
          startPage: pageNum + 1,
          endPage: pageNum + 1
        }
      } as KnowledgeBaseStoreDocument);
    }

    // Delete images that are no longer present
    for (const imageToDelete of imagesToDelete) {
      if (!imageToDelete._id) continue;

      operations.delete.push({
        delete: {
          _index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX,
          _id: imageToDelete._id
        }
      });
    }

    // Execute bulk operations
    const bulkResponseItems = [] as Partial<Record<estypes.BulkOperationType, estypes.BulkResponseItem>>[];

    if (operations.index.length > 0 || operations.delete.length > 0 || operations.update.length > 0) {
      // Combine all operations into a single flat array
      const allOperations = [
        ...operations.index,
        ...operations.delete,
        ...operations.update
      ];

      const operationsChunks = _.chunk(allOperations, 100);
      for (const operationChunk of operationsChunks) {
        if (operationChunk.length === 0) continue;
        
        const response = await handleElasticsearchBulkOperation(
          (params: estypes.BulkRequest) => this.store.es.bulk(params)
        )({ 
          refresh: true,
          operations: operationChunk 
        });
        bulkResponseItems.push(...response.items);
      }
    }

    // Log results
    const indexCount = bulkResponseItems.filter((item) => item.index && item.index.result === 'created').length;
    if (indexCount > 0) {
      _log.debug({
        msg: `Successfully upserted ${indexCount} image embedding record(s). Collection [ ${collection} ] - Document ID [ ${documentId} ]`,
        status_code: 201
      });
    }

    const deleteCount = bulkResponseItems.filter((item) => item.delete).length;
    if (deleteCount > 0) {
      _log.debug({
        msg: `Successfully deleted ${deleteCount} image embedding record(s). Collection [ ${collection} ] - Document ID [ ${documentId} ]`,
        status_code: 200
      });
    }

    const updateCount = bulkResponseItems.filter((item) => item.update).length;
    if (updateCount > 0) {
      _log.debug({
        msg: `Successfully updated ${updateCount} image embedding record(s) metadata. Collection [ ${collection} ] - Document ID [ ${documentId} ]`,
        status_code: 200
      });
    }

    return bulkResponseItems;
  }

  private async getImageTokenCount(base64Image: string): Promise<number> {
      return 1000;
      // Since the image token calculation logic is complex and model-dependent, we are currently returning a fixed estimate.
      // For Gemini multimodal models, a standard image requires approximately 1,806 tokens. However, when multiple images are input, the token count may be lower — for example, two images require around 700 tokens.
      // For OpenAI models, a 1024 × 1024 image requires approximately 1,000 tokens.
      // Because images are often used for search and multiple images are common, we are using 1,000 tokens as the estimate.
  }

  private getNoOfToken(chunk: string) {
      const tokenizerObj = get_encoding("cl100k_base")
      const nToken = tokenizerObj.encode(chunk).length;
      return nToken
  }
}

export interface KnowledgeBaseStoreDocument {
  id: string;
  content: string;
  chunk_type: "text" | "image";
  embeddings?: number[];
  image_embedding?: number[];
  path?: string;
  collection: string;
  parentId: string;
  n_token: number;
  last_updated_time: string;
  meta: KnowledgeBaseDocumentMetadata;
}