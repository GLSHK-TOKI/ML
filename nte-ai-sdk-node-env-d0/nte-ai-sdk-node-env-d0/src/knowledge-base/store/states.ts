import type { KnowledgeBaseStore, SearchHitWithSource } from './store.js';
import { DriveItem } from '@microsoft/msgraph-sdk/models/index.js';
import { KNOWLEDGE_BASE_INDEX_STATES_SUFFIX, OperationStatus } from '../_constants.js';
import { estypes, estypesWithBody } from '@elastic/elasticsearch';
import { handleElasticsearchOperation } from '../../exception/index.js'
import logger from "../../logger/logger.js";

const _log = logger.child({ module: 'ai-sdk-node.knowledge-base.store.states' });
export class KnowledgeBaseStoreStates {
  protected store: KnowledgeBaseStore;

  constructor(store: KnowledgeBaseStore) {
    this.store = store;
  }

  async put(file: DriveItem, fileStateId?: string, status: OperationStatus = OperationStatus.SUCCESSFUL, retryCount: number = 0) {
    /**
     * Create or update a record of document key in states index
     * If a state record is provided, it will be updated
     * Otherwise, a new record will be created
     **/
    const currentTime = new Date();

    const indexResponse = await handleElasticsearchOperation(
      (params: estypes.IndexRequest) => this.store.es.index(params)
    )({
      id: fileStateId ?? undefined,
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_STATES_SUFFIX,
      document: {
        parentId: file.parentReference?.id,
        id: file.id,
        collection: file.parentReference?.name,
        name: file.name,
        webUrl: file.webUrl,
        lastUpsertedDateTime: new Date(currentTime.getTime()).toISOString(),
        lastModifiedDateTime: file.lastModifiedDateTime?.toISOString(),
        status: status,
        retryCount: retryCount
      },
    });
    
    if (indexResponse.result == 'created' || indexResponse.result == 'updated') {
      const msg = `Successfully to put states record. Folder [ ${file.parentReference?.name} ] - File [ ${file.name} ]`;
      _log.debug({msg: msg, status_code: 200})
    } else {
      const msg = `Failed to put state record. Folder [ ${file.parentReference?.name} ] - File [ ${file.name} ]`;
      _log.debug({msg: msg, status_code: 400})
    }

    return indexResponse;
  }

  async deleteByCollection(parentId: string) {
    /**
     * Delete a record of document key from states index
     **/
    const deleteResponse = await handleElasticsearchOperation(
      (params: estypes.DeleteByQueryRequest) => this.store.es.deleteByQuery(params)
    )({
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_STATES_SUFFIX,
      query: {
        match: {
          parentId: parentId,
        },
      },
    });

    const deletedCount: number = deleteResponse.deleted ?? 0;

    if (deletedCount > 0) {
      const msg = `Successfully delete the states record. Folder ID [ ${parentId} ] ${deletedCount} ]`;
      _log.debug({msg: msg, status_code: 200})
    } else {
      _log.debug({
        msg: `No state records are deleted. Folder ID [ ${parentId} ]`,
        status_code: 200
      })
    }

    if (deleteResponse.failures && deleteResponse.failures.length > 0) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      deleteResponse.failures.forEach((failure:any) => {
        const msg = `Failed to delete states record: ${failure}`;
        _log.debug({msg: msg, status_code: 400})
      });
    }

    return deleteResponse;
  }

  /**
   * Delete a record by id from states index
   **/
  async delete(id: string) {
    const deleteResponse = await handleElasticsearchOperation(
      (params: estypes.DeleteRequest) => this.store.es.delete(params)
    )({
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_STATES_SUFFIX,
      id: id,
    });

    if (deleteResponse.result == 'deleted') {
      const msg = `Successfully to delete states record. File ID - [ ${id} ]`;
      _log.debug({msg: msg, status_code: 200})
    } else {
      const msg = `Failed to delete states record. File ID - [ ${id} ]`;
      _log.debug({msg: msg, status_code: 400})
    }

    
    return deleteResponse;
  }

  async getByParentId(parentId: string): Promise<SearchHitWithSource<KnowledgeBaseStoreFileState>[]> {
    /**
     * Retrieve a record of parentId key from states index
     **/
    const searchResponse: estypes.SearchResponse = await handleElasticsearchOperation(
      (params: estypes.SearchRequest) =>
        this.store.es.search<KnowledgeBaseStoreFileState>(params)
    )({
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_STATES_SUFFIX,
      query: {
        bool: {
          must: {
            match: {
              parentId: parentId,
            },
          },
        },
      },
      size: 9999
    });
    return searchResponse['hits']['hits'].filter((hit) => hit._source) as SearchHitWithSource<KnowledgeBaseStoreFileState>[];
  }

  async getFolders(): Promise<KnowledgeBaseStoreFolderState[]> {
    /**
     * Retrieve all folders (collections) from states index
     * @return: A list of folder states with complete details
     */
    const searchResponse: estypes.SearchResponse = await handleElasticsearchOperation(
      (params: estypes.SearchRequest) =>
        this.store.es.search<KnowledgeBaseStoreFolderState>(params)
    )({
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_STATES_SUFFIX,
      query: {
        bool: {
          must_not: [
            { exists: { field: "parentId" } }
          ]
        }
      },
      size: 9999
    });

    const hits = searchResponse.hits.hits.filter((hit) => hit._source) as SearchHitWithSource<KnowledgeBaseStoreFolderState>[];
    
    return hits.map(hit => ({
      parentId: null,
      id: hit._source.id,
      collection: null,
      name: hit._source.name,
      webUrl: hit._source.webUrl,
      lastProcessDateTime: hit._source.lastProcessDateTime || new Date().toISOString(),
      status: hit._source.status || OperationStatus.SUCCESSFUL,
    }));
  }

  async renameCollection(parentId: string, newName: string, newFolderWebUrl: string) {
    /**
     * Update collection(folder) name of file states
     **/
    const updateResponse: estypes.UpdateByQueryResponse = await handleElasticsearchOperation(
      (params: estypes.UpdateByQueryRequest) => 
      this.store.es.updateByQuery(params))(
      {
        index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_STATES_SUFFIX,
        script: {
          source: `
            ctx._source.collection = params.newName;
            // Get the filename by finding the last part after "/"
            def oldUrl = ctx._source.webUrl;
            def fileName = oldUrl.substring(oldUrl.lastIndexOf('/') + 1);
            // Combine new folder URL with original filename
            ctx._source.webUrl = params.newFolderWebUrl + '/' + fileName
          `,
          params : {
            newName : newName,
            newFolderWebUrl : newFolderWebUrl
          }
        },
        query: {
          match: {
            parentId: parentId,
          },
        },
      }
    );

    if (updateResponse.failures && updateResponse.failures.length > 0) {
      const msg = `Failed to rename collection name. Folder ID [ ${parentId} ] - New Name [ ${newName} ]`;
      _log.error({ msg: msg, status_code: 500 });
    } else if (updateResponse.updated) {
      const msg = `Successfully to rename collection name for ${updateResponse.updated} record(s). Folder ID [ ${parentId} ] - New Name [ ${newName} ]`;
      _log.debug({ msg: msg, status_code: 200 });
    }

    return updateResponse;
  }

  async putFolder(folder: DriveItem, folderStateId?: string, status: OperationStatus = OperationStatus.SUCCESSFUL) {
    /**
     * Create or update a folder state record in states index
     * If folderStateId is provided, it will be updated
     * Otherwise, a new record will be created
     **/
    
    if (!folder.id || !folder.name) {
      const msg = `Invalid folder data provided`;
      _log.error({ msg: msg, status_code: 400 });
      throw new Error(msg);
    }

    const currentTime = new Date();

    const folderStateDocument: KnowledgeBaseStoreFolderState = {
      parentId: null,
      id: folder.id,
      collection: null,
      name: folder.name,
      webUrl: folder.webUrl ?? '',
      lastProcessDateTime: new Date(currentTime.getTime()).toISOString(),
      status: status,
    };

    const folderResponse = await handleElasticsearchOperation(
      (params: estypes.IndexRequest) => this.store.es.index(params)
    )({
      id: folderStateId,
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_STATES_SUFFIX,
      document: folderStateDocument,
    });

    if (folderResponse.result === 'created' || folderResponse.result === 'updated') {
      const action = folderResponse.result == 'created' ? 'created' : 'updated';
      const msg = `Successfully ${action} folder state record. Folder [ ${folder.name} ] - Status [ ${status} ]`;
      _log.debug({msg: msg, status_code: 200})
    } else {
      const msg = `Failed to put folder state record. Folder [ ${folder.name} ]`;
      _log.error({msg: msg, status_code: 400})
    }

    return folderResponse;
  }

  async getFolderState(folderId: string): Promise<SearchHitWithSource<KnowledgeBaseStoreFolderState> | null> {
    /**
     * Retrieve a folder state record by folder ID
     * Folder states are identified by having parentId: null
     **/
    const searchResponse: estypes.SearchResponse = await handleElasticsearchOperation(
      (params: estypes.SearchRequest) =>
        this.store.es.search<KnowledgeBaseStoreFolderState>(params) // Change from KnowledgeBaseStoreFileState
    )({
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_STATES_SUFFIX,
      query: {
        bool: {
          must: [
            { term: { "id.keyword": folderId } },
            { bool: { must_not: { exists: { field: "parentId" } } } }
          ]
        }
      },
      size: 1
    });

    const hits = searchResponse.hits.hits.filter((hit) => hit._source) as SearchHitWithSource<KnowledgeBaseStoreFolderState>[];
    return hits.length > 0 ? hits[0] : null;
  }

  async getAllFolderStates(): Promise<SearchHitWithSource<KnowledgeBaseStoreFolderState>[]> {
    /**
     * Retrieve all folder state records
     * Folder states are identified by having parentId: null
     **/
    const searchResponse: estypes.SearchResponse = await handleElasticsearchOperation(
      (params: estypes.SearchRequest) =>
        this.store.es.search<KnowledgeBaseStoreFolderState>(params) // Change from KnowledgeBaseStoreFileState to KnowledgeBaseStoreFolderState
    )({
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_STATES_SUFFIX,
      query: {
        bool: {
          must_not: [
            { exists: { field: "parentId" } }
          ]
        }
      },
      size: 9999
    });

    return searchResponse.hits.hits.filter((hit) => hit._source) as SearchHitWithSource<KnowledgeBaseStoreFolderState>[];
  }

  async deleteFolderState(folderId: string) {
    /**
     * Delete a folder state record by folder ID
     * Folder states are identified by having parentId: null
     **/
    const deleteResponse: estypes.DeleteByQueryResponse = await handleElasticsearchOperation(
      (params: estypes.DeleteByQueryRequest) => this.store.es.deleteByQuery(params)
    )({
      index: this.store.indexPrefix + KNOWLEDGE_BASE_INDEX_STATES_SUFFIX,
      query: {
        bool: {
          must: [
            { term: { "id.keyword": folderId } },
            { bool: { must_not: { exists: { field: "parentId" } } } }
          ]
        }
      },
    });

    const deletedCount: number = deleteResponse.deleted ?? 0;

    if (deletedCount > 0) {
      const msg = `Successfully deleted folder state record. Folder ID [ ${folderId} ]`;
      _log.debug({msg: msg, status_code: 200})
    } else {
      _log.debug({
        msg: `No folder state record deleted. Folder ID [ ${folderId} ]`,
        status_code: 200
      })
    }

    if (deleteResponse.failures && deleteResponse.failures.length > 0) {
      deleteResponse.failures.forEach((failure: estypes.BulkIndexByScrollFailure) => {
        const msg = `Failed to delete folder state record: ${failure}`;
        _log.error({msg: msg, status_code: 400})
      });
    }

    return deleteResponse;
  }
}

interface Aggregations {
  unique_values: estypes.AggregationsTermsAggregateBase<{ key: string, hits: estypesWithBody.SearchResponseBody<KnowledgeBaseStoreFileState> }>
}
export interface KnowledgeBaseStoreFileState {
  parentId: string;
  id: string;
  collection: string;
  name: string;
  webUrl: string;
  lastUpsertedDateTime: string;
  lastModifiedDateTime: string;
  status: OperationStatus;
  retryCount: number;
}

export interface KnowledgeBaseStoreFolderState {
  parentId: null;
  id: string;
  collection: null;
  name: string;
  webUrl: string;
  lastProcessDateTime: string;
  status: OperationStatus;
}