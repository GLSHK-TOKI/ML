import { SourceConnector, SourceConnectorResult } from "./source-connector.js";
import { ClientSecretCredential } from '@azure/identity';
import { AzureIdentityAuthenticationProvider } from "@microsoft/kiota-authentication-azure";
import { FetchRequestAdapter } from "@microsoft/kiota-http-fetchlibrary";
import { createGraphServiceClient, type GraphServiceClient } from "@microsoft/msgraph-sdk";
import "@microsoft/msgraph-sdk-drives";
import { type KnowledgeBaseStore } from "./knowledge-base/index.js";
import {
  KnowledgeBasePDFDocument,
  KnowledgeBaseDocument,
  KnowledgeBaseImageDocument,
  KnowledgeBaseWebLink,
  KnowledgeBaseDOCDocument,
  KnowledgeBaseDOCXDocument,
  KnowledgeBaseHTMLDocument,
  KnowledgeBaseMDDocument,
  KnowledgeBaseRTFDocument,
  KnowledgeBaseTXTDocument,
} from './knowledge-base/document/index.js';
import { KnowledgeBaseStoreFileState } from './knowledge-base/store/states.js';
import { type SearchHitWithSource } from "./knowledge-base/store/store.js";
import { DriveItem } from "@microsoft/msgraph-sdk/models/index.js";
import _ from 'lodash';
import { handMSGraphOperation, MSGraphError, SDKException } from "./exception/index.js";
import logger from "./logger/logger.js";
import { OperationStatus } from "./knowledge-base/_constants.js";
import { PIIDetector, PIIDetectionResult } from "./pii-detection/index.js";
export interface SharePointConnectorOptions {
  azure: {
    tenantId: string;
    clientId: string;
    clientSecret: string;
  },
  sharepoint: {
    driveId: string;
    folderId: string;
  },
  webLinkExpireInterval?: number;
  maxRetryCount?: number;
  collectionMultimodalSettings?: Record<string, boolean>;
}

export type PIIDetectedCallback = (
  piiResults: PIIDetectionResult[],
  document: KnowledgeBaseDocument,
  chunkContent: string
) => void | Promise<void>;

const _log = logger.child({ module: 'ai-sdk-node.sharepoint-connector' });

export class SharePointConnector extends SourceConnector {
  protected graph: GraphServiceClient;
  protected store: KnowledgeBaseStore;
  readonly driveId: string;
  readonly folderId: string;
  readonly webLinkExpireInterval: number;
  readonly maxRetryCount: number;
  readonly collectionMultimodalSettings: Record<string, boolean>;

  constructor(options: SharePointConnectorOptions, store: KnowledgeBaseStore) {
    super();
    this.store = store;
    this.driveId = options.sharepoint.driveId;
    this.folderId = options.sharepoint.folderId;
    this.webLinkExpireInterval = options.webLinkExpireInterval || 43200000; // 12 hours
    this.maxRetryCount = options.maxRetryCount || 3; // Default to 3 retries
    this.collectionMultimodalSettings = options.collectionMultimodalSettings || {};

    // Initialize the GraphServiceClient for Microsoft Graph API on SharePoint
    const credential = new ClientSecretCredential(
      options.azure.tenantId,
      options.azure.clientId,
      options.azure.clientSecret
    );
    const authProvider = new AzureIdentityAuthenticationProvider(
      credential,
      ['https://graph.microsoft.com/.default']
    );
    const requestAdapter = new FetchRequestAdapter(authProvider)

    this.graph = createGraphServiceClient(requestAdapter);

    // Throw error if any space has image processing enabled but multimodal embedding model is not configured
    const hasImageProcessingEnabled = Object.values(this.collectionMultimodalSettings).some(enabled => enabled === true);
    if (hasImageProcessingEnabled && !this.store.multimodalEmbeddingModel) {
      const msg = "Image processing is enabled for one or more spaces, but multimodal embedding model is not configured in KnowledgeBaseStore.";
      _log.error({ msg: msg, status_code: 400 });
      throw new SDKException(400, msg);
    }
  }

  /**
   * Execute the SharePoint connector to sync files from SharePoint knowledge base
   * to Elastic Search index documents with embeddings.
   * @returns Changes made in the sync process.
   */
  async run(): Promise<SharePointConnectorResult> {
    const runResult: SharePointConnectorResult = {
      status: 'success',
      noChange: { total: 0, files: [] },
      upserted: { total: 0, files: [] },
      removed: { total: 0, files: [] },
    };

    // 1.1 Get folders from SharePoint, under knowledge_base root folder
    // 1.2 Get folder states from Elastic Search
    const folders = await this.getFoldersFromRootFolder();
    const folderStates = await this.store.states.getFolders();

// 2 Delete documents by collection if entire folder is removed compared to states
    const folderMap = _.keyBy(folders, 'id');
    for (const folderState of folderStates) {
      const folder = folderMap[folderState.id]; // Using folder state to check if folder exists in SharePoint

      // If folder is not found in SharePoint, but found in states
      if (!folder) {
        // Delete all files under the folder
        await this.store.documents.deleteByCollection(folderState.id);
        const statesToBeDeleteByCollection = await this.store.states.getByParentId(folderState.id);
        await this.store.states.deleteByCollection(folderState.id);
        // Also delete the folder state itself
        await this.store.states.deleteFolderState(folderState.id);

        runResult.removed.total += statesToBeDeleteByCollection.length;
        runResult.removed.files = statesToBeDeleteByCollection
          .filter((state) => state._source.parentId === folderState.id)
          .map((state) => ({
            id: state._source.id || '',
            collection: folderState.name || '',
            name: state._source.name || '',
          }));
        continue;
      }

      // Rename all documents and file states collection name if folder name is changed
      if (folder.name && folder.name !== folderState.name) {
        await this.store.documents.renameCollection(folderState.id, folder.name, folder.webUrl || '');
        await this.store.states.renameCollection(folderState.id, folder.name, folder.webUrl || '');
        const existingFolderStateWithId = await this.store.states.getFolderState(folderState.id);
        const folderStatus = await this.checkFolderFilesStatus(folderState.id);
        await this.store.states.putFolder(folder, existingFolderStateWithId?._id, folderStatus);
      }
    };

    const filesWithoutChangesResult: DriveItem[] = [];
    const filesToBeUpsertResult: DriveItem[] = [];
    const statesToBeRemovedResult: SearchHitWithSource<KnowledgeBaseStoreFileState>[] = [];

    // 3 Get files and file states per folder
    // Use folders here, as it's guaranteed to be defined
    for (const folder of folders) {
      if (!folder.id || !folder.name) {
        _log.error({ msg: `Invalid folder found`, status_code: 400 });
        return runResult;
      }
      // Update folder state to PROCESSING at the start of processing
      const existingFolderState = await this.store.states.getFolderState(folder.id);
      let folderStateId = existingFolderState?._id;
      const processingStateResponse = await this.store.states.putFolder(folder, folderStateId, OperationStatus.PROCESSING);
      
      // If this was a new folder state record, capture the ID for future updates
      if (!folderStateId) {
        folderStateId = processingStateResponse._id;
      }
      
      const files = await this.getFilesFromFolder(folder.id);
      const fileStates = await this.store.states.getByParentId(folder.id);

      // Convert fileStates to a map for quick lookup
      const statesMap = _.keyBy(fileStates, (state: SearchHitWithSource<KnowledgeBaseStoreFileState>) => (state._source.id || ''));
      const fileMap = _.keyBy(files, 'id');

      const filesWithoutChanges: DriveItem[] = [];
      const filesToBeUpsert: DriveItem[] = [];
      const statesToBeRemoved: SearchHitWithSource<KnowledgeBaseStoreFileState>[] = [];

      // 3.1 Delete files if found in states but not in SharePoint
      for (const state of fileStates) {
        const file = fileMap[state._source.id]; // Using file state to check if file exists in SharePoint

        if (!file) {
          statesToBeRemoved.push(state);
          if (state._id) {
            // Delete the all documents with the same id (all chunks)
            try {
              await this.store.documents.delete(state._source.id);
              await this.store.states.delete(state._id);
            } catch (error) {
              const msg = `Error processing folder ${folder.name} (ID: ${folder.id})`;
              _log.error({ msg: msg, status_code: 500, err: error });
            }
          }
        }
      }

      // 3.2 Check each file fetched from SharePoint against states
      // Sort files: non-failed files first
      const sortedFiles = [...files].sort((a, b) => {
        const stateA = statesMap[a.id || ''];
        const stateB = statesMap[b.id || ''];

        // Non-failed files come first
        if (
          stateA?._source.status === OperationStatus.FAILED &&
          stateB?._source.status !== OperationStatus.FAILED
        )
          return 1;
        if (
          stateA?._source.status !== OperationStatus.FAILED &&
          stateB?._source.status === OperationStatus.FAILED
        )
          return -1;
        return 0;
      });

      // Process each file with circuit breaker pattern
      for (const file of sortedFiles) {
        const fileId = file.id;
        if (!fileId) {
          const msg = `Invalid file found`;
          _log.error({msg: msg, status_code: 400});
          continue;
        }
        // Find out all File ID can be found in Elastic Search States
        const fileState = statesMap[fileId]; // Using file in SharePoint to check if file exists in Elasticsearch

        const document = this.getDocumentInstance(file);
        //Handling the unsupported document type
        if (!document) {
          if (!fileState) {
            await this.store.states.put(file, undefined, OperationStatus.FAILED, this.maxRetryCount);
          } else {
            const isModified = this.isFileModified(file, fileState);
            let fileStateId = fileState._id;
            if (isModified) {
              await this.store.states.put(file, fileStateId, OperationStatus.FAILED, this.maxRetryCount);
            }
          }
          continue;
        }

        if (this.shouldUpsertFile(file, fileState)) {
          filesToBeUpsert.push(file);
          
          // Get the current retry count and file state ID
          const currentRetryCount = fileState?._source.retryCount || 0;
          let fileStateId = fileState?._id;
          
          try {
            // Update file state to PROCESSING before starting upsert
            const processingStateResponse = await this.store.states.put(file, fileStateId, OperationStatus.PROCESSING, currentRetryCount);
            
            // If this was a new state record, capture the ID for future updates
            if (!fileStateId) {
              fileStateId = processingStateResponse._id;
            }
            
            // Process text chunks for all document types except raw images
            if (!(document instanceof KnowledgeBaseImageDocument)) {
              await this.store.documents.upsert(document);
            }

            // Process image embeddings if enabled for this collection
            if (this.isImageProcessingEnabled(folder.id)) {
              await this.store.documents.upsertImage(document);
            }
            
            // Update the same file state record to SUCCESSFUL after successful upsert (reset retry count)
            await this.store.states.put(file, fileStateId, OperationStatus.SUCCESSFUL, 0);
            
            _log.debug({
              msg: `Successfully processed file ${file.name} (ID: ${file.id})`,
              status_code: 200
            });
            
          } catch (error) {
            const isModified = this.isFileModified(file, fileState);
            let newRetryCount: number;
            if (isModified) {
              newRetryCount = 0;
            } else {
              newRetryCount = currentRetryCount + 1;
            }
            await this.store.states.put(
              file,
              fileState?._id,
              OperationStatus.FAILED,
              newRetryCount
            );
            const msg = `Error upserting document: ${document.getDocId()}, retry ${newRetryCount}/${this.maxRetryCount}`;
            _log.error({msg: msg, status_code: 400, err: error});
            continue;
          }
        } else {
          filesWithoutChanges.push(file);
        }
      }

      filesToBeUpsertResult.push(...filesToBeUpsert);
      filesWithoutChangesResult.push(...filesWithoutChanges);
      statesToBeRemovedResult.push(...statesToBeRemoved);

      // At the end of folder processing
      try {
        // Small delay (~1s) added to allow Elasticsearch sufficient time to index and reflect state updates,
        await new Promise(resolve => setTimeout(resolve, 1000));
        const folderStatus = await this.checkFolderFilesStatus(folder.id);
        await this.store.states.putFolder(folder, folderStateId, folderStatus);
        
        const msg = `Folder processing completed for folder ${folder.name} (ID: ${folder.id}) - Final Status: ${folderStatus}`;
        _log.debug({msg: msg, status_code: 200});
      } catch (error) {
        const msg = `Error updating folder status for folder ${folder.name} (ID: ${folder.id})`;
        _log.error({ msg: msg, status_code: 500, err: error });
      }
    }

    // Combine actions into result
    runResult.noChange.total += filesWithoutChangesResult.length;
    runResult.noChange.files = [
      ...runResult.noChange.files,
      ...filesWithoutChangesResult.map((file) => ({ id: file.id || '', collection: file.parentReference?.name || '', name: file.name || '' }))
    ];
    runResult.upserted.total += filesToBeUpsertResult.length;
    runResult.upserted.files = [
      ...runResult.upserted.files,
      ...filesToBeUpsertResult.map((file) => ({ id: file.id || '', collection: file.parentReference?.name || '', name: file.name || '' }))
    ];
    runResult.removed.total += statesToBeRemovedResult.length;
    runResult.removed.files = [
      ...runResult.removed.files,
      ...statesToBeRemovedResult.map((state) => ({ id: state._source.id || '', collection: state._source.collection || '', name: state._source.name || '' }))
    ]
    return runResult;
  }

  async add_pii_detector(piiDetector: PIIDetector) {
    this.store.piiDetector = piiDetector;
  }

  async add_pii_detected_callback(callback: PIIDetectedCallback) {
    if (!this.store.piiDetector) {
      throw new Error("PII Detector is not initialized. Please add PII Detector first.");
    }
    this.store.piiDetectedCallback = callback;
  }

  /**
   * Check if image processing is enabled for a specific collection/space
   * @param collectionId The collection/space ID (folder ID)
   * @returns true if image processing is enabled for this collection
   */
  private isImageProcessingEnabled(collectionId: string): boolean {
    return this.collectionMultimodalSettings[collectionId] === true;
  }

  private async getFoldersFromRootFolder() {
    const items = await this.paginateDriveItemsFromFolder(this.folderId);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return items.filter((item:any) => item.folder !== null);
  }

  private async getFilesFromFolder(folderId: string) {
    const items = await this.paginateDriveItemsFromFolder(folderId);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return items.filter((item:any) => item.file !== null);
  }

  private async paginateDriveItemsFromFolder(folderId: string): Promise<DriveItem[]> {
    const folders = []
    let response = await handMSGraphOperation(() => this.graph.drives.byDriveId(this.driveId).items.byDriveItemId(folderId).children.get());

    if (!response) {
      const msg = `Error while fetching items from SharePoint folder with id: ${folderId}`;
      _log.error({msg: msg, status_code: 400})
      throw new MSGraphError(400, msg);
    }

    if (response.value) {
      folders.push(...response.value);
    }

    while (response && response.odataNextLink) {
      response = await this.graph.drives.withUrl(response.odataNextLink).get();
      if (response && response.value) {
        folders.push(...response.value);
      }
    }
    // Shuffle the folders/files to avoid the same order of folders/files
    return folders.sort(() => Math.random() - 0.5);
  }

  private getDocumentInstance(file: DriveItem): KnowledgeBaseDocument | null {
    const webUrl = file.webUrl || ''; // Handle undefined case
    if (!webUrl) {
      const msg = `Invalid webUrl : ${file}`;
      _log.error({msg: msg, status_code: 400})
      return null;
    }
    const fileExtension = this.getFileExtension(webUrl);
    
    // Check for image files first (handles multiple extensions: jpg, jpeg, png)
    if (KnowledgeBaseImageDocument.extension.split(',').includes(fileExtension)) {
      return new KnowledgeBaseImageDocument(file);
    }
    
    // Check other document types
    switch(fileExtension) {
      case KnowledgeBasePDFDocument.extension: {
        return new KnowledgeBasePDFDocument(file);
      }
      case KnowledgeBaseWebLink.extension: {
        return new KnowledgeBaseWebLink(file);
      }
      case KnowledgeBaseTXTDocument.extension: {
        return new KnowledgeBaseTXTDocument(file);
      }
      case KnowledgeBaseMDDocument.extension: {
        return new KnowledgeBaseMDDocument(file);
      }
      case KnowledgeBaseDOCDocument.extension: {
        return new KnowledgeBaseDOCDocument(file);
      }
      case KnowledgeBaseDOCXDocument.extension: {
        return new KnowledgeBaseDOCXDocument(file);
      }
      case KnowledgeBaseRTFDocument.extension: {
        return new KnowledgeBaseRTFDocument(file);
      }
      case KnowledgeBaseHTMLDocument.extension: {
        return new KnowledgeBaseHTMLDocument(file);
      }
      default: {
        const msg = `Unsupported file type: ${webUrl}`;
        _log.error({msg: msg, status_code: 400});
        return null;
      }
    }
  }

  private getFileExtension(webUrl: string): string {
    const lastSlashIndex = webUrl.lastIndexOf('/');
    if (lastSlashIndex === -1) return '';
  
    const fileName = webUrl.substring(lastSlashIndex + 1);
    const dotIndex = fileName.lastIndexOf('.');
    if (dotIndex === -1) return '';
  
    if(fileName.includes('&action')) {
      const actionIndex = fileName.lastIndexOf('&action');
      return fileName.substring(dotIndex + 1, actionIndex);
    }    

    return fileName.substring(dotIndex + 1).toLowerCase();
  }
  
  private shouldUpsertFile(file: DriveItem, state: SearchHitWithSource<KnowledgeBaseStoreFileState>): boolean {
    if (!state) {
      // File is not found in states, should be upserted
      return true;
    }

    // All file types should have lastModifiedDateTime
    if (!state._source.lastModifiedDateTime) {
      const msg = `Invalid state lastModifiedDateTime for file: ${state._source.name}`;
      _log.error({msg: msg, status_code: 400});
      return true; // Treat as needs upsert if state is invalid
    }
    const fileStateLastModified = new Date(state._source.lastModifiedDateTime);
    if (!file.lastModifiedDateTime ) {
      const msg = `Invalid file state file.lastModifiedDateTime : ${file.lastModifiedDateTime}`;
      _log.error({msg: msg, status_code: 400})
      throw new Error('Invalid file state file.lastModifiedDateTime');
    }
    const fileLastModified = new Date(file.lastModifiedDateTime);

    // Different upsert logic for different file types
    const webUrl = file.webUrl || ''; // Handle undefined case
    const fileExtension = this.getFileExtension(webUrl);
    
    // Check for image files first (handles multiple extensions: jpg, jpeg, png)
    if (KnowledgeBaseImageDocument.extension.split(',').includes(fileExtension)) {
      // Image files: check if renamed (different webUrl) or modified
      const hasWebUrlChanged = state._source.webUrl !== webUrl;
      if (hasWebUrlChanged || fileStateLastModified.getTime() !== fileLastModified.getTime()) {
        return true;
      }
      if ((state._source.status ?? OperationStatus.SUCCESSFUL) === OperationStatus.SUCCESSFUL) {
        return false;
      }
      if ((state._source.retryCount ?? 0) < this.maxRetryCount) {
        return true;
      } else {
        return false;
      }
    }
    
    // Check other document types
    switch(fileExtension) {
      case KnowledgeBasePDFDocument.extension:
      case KnowledgeBaseTXTDocument.extension: 
      case KnowledgeBaseMDDocument.extension: 
      case KnowledgeBaseDOCDocument.extension: 
      case KnowledgeBaseDOCXDocument.extension: 
      case KnowledgeBaseRTFDocument.extension: 
      case KnowledgeBaseHTMLDocument.extension: {
        if (fileStateLastModified.getTime() !== fileLastModified.getTime()) {
          // if file is modified, should be upserted
          return true;
        }
        if ((state._source.status ?? OperationStatus.SUCCESSFUL) === OperationStatus.SUCCESSFUL) {
          // If file is not modified and status is successful, should not be upserted
          return false;
        }
        if ((state._source.retryCount ?? 0) < this.maxRetryCount) {
          // If file is not modified, status is not successful, and retry count is less than max retry count, should be upserted
          return true;
        } else {
          // If file is not modified, status is not successful, and retry count has reached max retry count, should not be upserted
          const msg = `File ${file.name} (ID: ${file.id}) has reached max retry count`;
          _log.warn({msg: msg});
          return false;
        }
      }
      case KnowledgeBaseWebLink.extension: {
        const isModified = fileStateLastModified.getTime() !== fileLastModified.getTime();
        return isModified || this.isLinkExpired(state);
      }
      default: {
        const msg = `Unsupported file type : ${webUrl}`;
        _log.error({msg: msg, status_code: 400})
        return false;
      }
    }
  }

  private isLinkExpired(state: SearchHitWithSource<KnowledgeBaseStoreFileState>): boolean {
    const now = new Date();
    if (!state._source.lastUpsertedDateTime) {
      return true; // Treat as expired if no timestamp
    }
    const lastUpsertedDateTime = new Date(state._source.lastUpsertedDateTime);
    const linkSourceExpireInterval = this.webLinkExpireInterval || 43200000; // 12 hours
    const diff = now.getTime() - lastUpsertedDateTime.getTime();
    return diff > linkSourceExpireInterval;
  }

  private isFileModified(file: DriveItem, state: SearchHitWithSource<KnowledgeBaseStoreFileState>): boolean {
    if (!state || !file.lastModifiedDateTime) {
      return true; // Treat as modified if no state or no timestamp
    }

    if (!state._source.lastModifiedDateTime) {
      return true; // Treat as modified if state has no timestamp
    }

    const fileStateLastModified = new Date(state._source.lastModifiedDateTime);
    const fileLastModified = new Date(file.lastModifiedDateTime);
    
    return fileStateLastModified.getTime() !== fileLastModified.getTime();
  }

  
  private async checkFolderFilesStatus(folderId: string): Promise<OperationStatus> {
    /**
     * Check the status of all files within a folder and determine folder status
     * Returns SUCCESSFUL only if all files have SUCCESSFUL status
     * Returns FAILED if any file has FAILED status
     * Returns PROCESSING if any file has PROCESSING status
     **/
    try {
      const fileStates = await this.store.states.getByParentId(folderId);
      
      if (fileStates.length === 0) {
        return OperationStatus.SUCCESSFUL;
      }

      let hasFailedFiles = false;

      for (const fileState of fileStates) {
        const status = fileState._source.status || OperationStatus.SUCCESSFUL;
        
        if (status === OperationStatus.FAILED) {
          hasFailedFiles = true;
        }
      }
      
      if (hasFailedFiles) {
        return OperationStatus.FAILED;
      } else {
        return OperationStatus.SUCCESSFUL;
      }

    } catch (error) {
      const msg = `Error checking folder files status for folder ID: ${folderId}`;
      _log.error({ msg: msg, status_code: 500, err: error });
      return OperationStatus.FAILED;
    }
  }
}

export interface SharePointConnectorResult extends SourceConnectorResult {
  noChange: {
    total: number;
    files: {
      id: string;
      collection: string;
      name: string;
    }[],
  },
  upserted: {
    total: number;
    files: {
      id: string;
      collection: string;
      name: string;
    }[],
  },
  removed: {
    total: number;
    files: {
      id: string;
      collection: string;
      name: string;
    }[],
  },
}