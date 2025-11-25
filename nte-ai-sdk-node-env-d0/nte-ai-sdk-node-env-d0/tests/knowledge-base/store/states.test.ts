import { expect } from 'chai';
import { DriveItem } from '@microsoft/msgraph-sdk/models';
import { createDummyKnowledgeBaseStore } from '../../utils/mock-knowledge-base-store';

import { type KnowledgeBaseStore } from '#ai-sdk-node';
import { KNOWLEDGE_BASE_INDEX_STATES_SUFFIX } from '#ai-sdk-node/knowledge-base/_constants';

describe('Knowledge Base Store States', () => {
  let store: KnowledgeBaseStore;
  const dummyFile: DriveItem = {
    createdDateTime: new Date('2024-07-31T01:54:20Z'),
    id: 'item-id',
    lastModifiedDateTime: new Date('2024-07-31T01:54:20Z'),
    name: 'some-document.pdf',
    parentReference: {
      id: 'parent-id',
      name: 'parent-name',
    },
    webUrl: 'https://dummy-url.pdf',
  };

  before(() => {
    store = createDummyKnowledgeBaseStore();
  });

  beforeEach(async () => {
    await cleanUpStatesIndex(store);
    await refreshStatesIndex(store);
  })

  afterEach(async () => {
    await refreshStatesIndex(store);
  })

  after(async () => {
    await cleanUpStatesIndex(store);
  })

  it('should put a record into states', async () => {
    // Act
    const response = await store.states.put(dummyFile);

    // Assert
    expect(response.result).to.equal('created');
  });

  it('should update a record if it already exists in states', async () => {
    // Arrange
    const prevResponse = await store.states.put(dummyFile);
    await refreshStatesIndex(store);

    // Act
    const searchResponse = await store.states.getByParentId(dummyFile.parentReference!.id!);

    dummyFile.lastModifiedDateTime = new Date('2024-08-31T01:54:21Z');
    const response = await store.states.put(dummyFile, searchResponse[0]?._id);

    // Assert
    expect(prevResponse._id).to.equal(response._id);
    expect(response.result).to.equal('updated');
  });

  it('should get records by parent id from states', async () => {
    // Arrange
    await store.states.put(dummyFile);
    await refreshStatesIndex(store);

    // Act
    const response = await store.states.getByParentId(dummyFile.parentReference!.id!);

    // Assert
    expect(response.length).to.equal(1);
  });

  it('should delete records by collection', async () => {
    // Arrange
    await store.states.put(dummyFile);
    const anotherDummyFile = { ...dummyFile, id: 'another-item-id' };
    await store.states.put(anotherDummyFile);
    await refreshStatesIndex(store);

    // Act
    const response = await store.states.deleteByCollection(dummyFile.parentReference!.id!);

    // Assert
    expect(response.deleted).to.equal(2);
  });

  it('should delete record by id', async () => {
    // Arrange
    await store.states.put(dummyFile);
    await refreshStatesIndex(store);

    // Act
    const searchResponse = await store.states.getByParentId(dummyFile.parentReference!.id!);
    const response = await store.states.delete(searchResponse[0]._id!);

    // Assert
    expect(response.result).to.equal('deleted');
  });

  it('should get all folders collections from states', async () => {
    // Arrange
    await store.states.put(dummyFile);
    await refreshStatesIndex(store);

    // Act
    const response = await store.states.getFolders();

    // Assert
    expect(response.length).to.equal(1);
    expect(response[0].id).to.equal(dummyFile.parentReference!.id);
    expect(response[0].name).to.equal(dummyFile.parentReference!.name);
  });

  it('should rename collection name by query ', async ()=> {
    // Arrange
    await store.states.put(dummyFile);
    await refreshStatesIndex(store);

    // Act
    const newFolderName = 'new-folder-name';
    const response = await store.states.renameCollection(dummyFile.parentReference!.id!, newFolderName);
    await refreshStatesIndex(store);

    // Assert
    const searchResponse = await store.states.getByParentId(dummyFile.parentReference!.id!);
    expect(response.updated).to.equal(1);
    expect(searchResponse[0]._source.collection).to.equal(newFolderName);
  })
});

async function cleanUpStatesIndex(store: KnowledgeBaseStore) {
  /**
    * Clear every records in states index
    */
  return store.es.deleteByQuery({
    index: `${store.indexPrefix}${KNOWLEDGE_BASE_INDEX_STATES_SUFFIX}`,
    body: {
      query: {
        match_all: {},
      },
    },
  })
}

async function refreshStatesIndex(store: KnowledgeBaseStore) {
  /**
   * Refresh states index to avoid version conflict
   * (Required maintenance role for the user)
   */
  return store.es.indices.refresh({
    index: `${store.indexPrefix}${KNOWLEDGE_BASE_INDEX_STATES_SUFFIX}`,
  });
}