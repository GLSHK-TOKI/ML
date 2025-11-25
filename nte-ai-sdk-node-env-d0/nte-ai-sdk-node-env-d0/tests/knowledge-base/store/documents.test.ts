import { expect } from 'chai';
import { DriveItem } from '@microsoft/msgraph-sdk/models';
import { type SearchTotalHits } from '@elastic/elasticsearch/lib/api/types';
import { createDummyKnowledgeBaseStore } from '../../utils/mock-knowledge-base-store';

import { type KnowledgeBaseStore } from '#ai-sdk-node';
import { KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX } from '#ai-sdk-node/knowledge-base/_constants';
import { KnowledgeBaseDocument, KnowledgeBasePDFDocument } from '#ai-sdk-node/knowledge-base';
import sinon from 'sinon';
import { KnowledgeBaseStoreDocument } from '#ai-sdk-node/knowledge-base/store/documents';

describe('Knowledge Base Store Documents', () => {
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
  const dummyPdfDocument: KnowledgeBaseDocument = new KnowledgeBasePDFDocument(dummyFile);

  before(() => {
    store = createDummyKnowledgeBaseStore();
  });

  beforeEach(async () => {
    sinon.stub(store.getInstance(store.embeddingModels), 'embedQuery')
      .resolves([123, 456, 789]);

    await cleanUpDocumentsIndex(store);
    await refreshDocumentsIndex(store);
  })

  afterEach(async () => {
    await refreshDocumentsIndex(store);
    sinon.restore();
  })

  after(async () => {
    await cleanUpDocumentsIndex(store);
  })

  it('should upsert a new pdf document into docs', async () => {
    // Arrange
    sinon.stub(dummyPdfDocument, 'getContent').resolves('dummy-content');

    // Act
    const response = await store.documents.upsert(dummyPdfDocument);
    await refreshDocumentsIndex(store);

    // Assert
    const searchResponse = await search(dummyPdfDocument.getDocId(), store);
    const total = searchResponse.hits.total as SearchTotalHits;
    expect(total.value).to.equal(1);
    expect(response).to.not.equal(null);
    expect(response?.length).to.be.equal(1);
    expect(response?.[0].index?.result).to.be.equal('created');
    expect(response?.[0].index?.status).to.be.equal(201);
  });

  it('should upsert a new pdf document with 3 chunks', async () => {
    // Arrange
    // Content is around 240 characters
    // Chunk size is 100 characters, overlapping is 20 characters
    const content = 'Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.';
    sinon.stub(dummyPdfDocument, 'getContent').resolves(content);

    // Act
    const response = await store.documents.upsert(dummyPdfDocument);
    await refreshDocumentsIndex(store);

    // Assert
    const searchResponse = await search(dummyPdfDocument.getDocId(), store);
    const total = searchResponse.hits.total as SearchTotalHits;
    expect(total.value).to.equal(3);
    expect(response).to.not.equal(null);
    expect(response?.length).to.be.equal(3);
    for (const res of response || []) {
      expect(res.index?.result).to.be.equal('created');
      expect(res.index?.status).to.be.equal(201);
    }
  });

  it('should upsert no chunks if chunks are indexed already', async () => {
    // Arrange
    // Content is around 240 characters
    // Chunk size is 100 characters, overlapping is 20 characters
    const content = 'Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.';
    sinon.stub(dummyPdfDocument, 'getContent').resolves(content);
    await store.documents.upsert(dummyPdfDocument);
    await refreshDocumentsIndex(store);

    // Act
    const response2 = await store.documents.upsert(dummyPdfDocument);
    await refreshDocumentsIndex(store);

    // Assert
    const searchResponse = await search(dummyPdfDocument.getDocId(), store);
    const total = searchResponse.hits.total as SearchTotalHits;
    expect(total.value).to.equal(3);
    expect(response2).to.be.equal(null);
  });

  it('should upsert new chunks, delete not equal chunks if new content is longer', async () => {
    // Arrange
    // Chunk size is 100 characters, overlapping is 20 characters
    // Content is around 150 characters
    const content = 'Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s,';
    const getContentStub = sinon.stub(dummyPdfDocument, 'getContent').resolves(content);
    await store.documents.upsert(dummyPdfDocument);
    await refreshDocumentsIndex(store);
    const searchResponse = await search(dummyPdfDocument.getDocId(), store);
    const total = searchResponse.hits.total as SearchTotalHits;

    // Act
    // Content is around 240 characters
    const content2 = 'Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.';
    getContentStub.resolves(content2);
    const response2 = await store.documents.upsert(dummyPdfDocument);
    await refreshDocumentsIndex(store);

    // Assert
    const searchResponse2 = await search(dummyPdfDocument.getDocId(), store);
    const total2 = searchResponse2.hits.total as SearchTotalHits;
    expect(total.value).to.equal(2);
    expect(total2.value).to.equal(3);
    expect(response2).to.not.equal(null);
    expect(response2?.length).to.be.equal(3);
    expect(response2?.[0].index?.result).to.be.equal('created');
    expect(response2?.[0].index?.status).to.be.equal(201);
    expect(response2?.[1].index?.result).to.be.equal('created');
    expect(response2?.[1].index?.status).to.be.equal(201);
    expect(response2?.[2].delete?.result).to.be.equal('deleted');
    expect(response2?.[2].delete?.status).to.be.equal(200);
  });

  it('should delete chunks, upsert not equal chunks if new content is shorter', async () => {
    // Arrange
    // Content is around 240 characters
    // Chunk size is 100 characters, overlapping is 20 characters
    const content = 'Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.';
    const getContentStub = sinon.stub(dummyPdfDocument, 'getContent').resolves(content);
    await store.documents.upsert(dummyPdfDocument);
    await refreshDocumentsIndex(store);
    const searchResponse = await search(dummyPdfDocument.getDocId(), store);
    const total = searchResponse.hits.total as SearchTotalHits;

    // Act
    // Content is around 150 characters
    const content2 = 'Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s,';
    getContentStub.resolves(content2);
    const response2 = await store.documents.upsert(dummyPdfDocument);
    await refreshDocumentsIndex(store);

    // Assert
    const searchResponse2 = await search(dummyPdfDocument.getDocId(), store);
    const total2 = searchResponse2.hits.total as SearchTotalHits;
    expect(total.value).to.equal(3);
    expect(total2.value).to.equal(2);
    expect(response2).to.not.equal(null);
    expect(response2?.length).to.be.equal(3);
    expect(response2?.[0].index?.result).to.be.equal('created');
    expect(response2?.[0].index?.status).to.be.equal(201);
    expect(response2?.[1].delete?.result).to.be.equal('deleted');
    expect(response2?.[1].delete?.status).to.be.equal(200);
    expect(response2?.[2].delete?.result).to.be.equal('deleted');
    expect(response2?.[2].delete?.status).to.be.equal(200);
  });

  it('should update chunks if file properties are updated', async () => {
    // Arrange
    // Content is around 240 characters
    // Chunk size is 100 characters, overlapping is 20 characters
    const content = 'Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.';
    sinon.stub(dummyPdfDocument, 'getContent').resolves(content);
    await store.documents.upsert(dummyPdfDocument);
    await refreshDocumentsIndex(store);

    // Act

    const updatedDummyFile = structuredClone(dummyFile);
    updatedDummyFile.parentReference!.name = 'new-parent-name';
    updatedDummyFile.parentReference!.id = 'new-parent-id';
    updatedDummyFile.webUrl = 'https://new-dummy-url.pdf';
    updatedDummyFile.name = 'new-some-document.pdf';

    const updatedDummyPdfDocument = new KnowledgeBasePDFDocument(updatedDummyFile);
    sinon.stub(updatedDummyPdfDocument, 'getContent').resolves(content);
    const response2 = await store.documents.upsert(updatedDummyPdfDocument);
    await refreshDocumentsIndex(store);

    // Assert
    const searchResponse = await search(dummyPdfDocument.getDocId(), store);
    const total = searchResponse.hits.total as SearchTotalHits;
    const docs = searchResponse.hits.hits;
    expect(total.value).to.equal(3);
    for (const doc of docs) {
      expect(doc._source?.collection).to.equal(updatedDummyPdfDocument.getCollection());
      expect(doc._source?.parentId).to.equal(updatedDummyPdfDocument.getParentId());
      expect(doc._source?.meta.title).to.equal((await updatedDummyPdfDocument.getMetadata()).title);
      expect(doc._source?.meta.webUrl).to.equal((await updatedDummyPdfDocument.getMetadata()).webUrl);
    }
    expect(response2).to.not.equal(null);
    expect(response2?.length).to.be.equal(3);
    for (const res of response2 || []) {
      expect(res.update?.result).to.be.equal('updated');
      expect(res.update?.status).to.be.equal(200);
    }
  });
});

async function cleanUpDocumentsIndex(store: KnowledgeBaseStore) {
  /**
    * Clear every records in states index
    */
  return store.es.deleteByQuery({
    index: `${store.indexPrefix}${KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX}`,
    body: {
      query: {
        match_all: {},
      },
    },
  })
}

async function refreshDocumentsIndex(store: KnowledgeBaseStore) {
  /**
   * Refresh states index to avoid version conflict
   * (Required maintenance role for the user)
   */
  return store.es.indices.refresh({
    index: `${store.indexPrefix}${KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX}`,
  });
}

/**
 * Helper function for asserting the result on docs index.
 * @param docId The document id.
 * @param store The knowledge base store.
 * @returns 
 */
async function search(docId: string, store: KnowledgeBaseStore) {
  return store.es.search<KnowledgeBaseStoreDocument>({
    index: `${store.indexPrefix}${KNOWLEDGE_BASE_INDEX_DOCS_SUFFIX}`,
    body: {
      query: {
        match: {
          id: docId,
        },
      },
    },
    size: 9999,
  });
}