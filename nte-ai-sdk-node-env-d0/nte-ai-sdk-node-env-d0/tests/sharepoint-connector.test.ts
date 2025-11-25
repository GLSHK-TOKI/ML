/* eslint-disable @typescript-eslint/no-explicit-any */
/* workaround for stubbing private methods */

import sinon from "sinon";
import { expect } from "chai";

import { createMockKnowledgeBaseStore } from "./utils/mock-knowledge-base-store";

import { KnowledgeBaseStore, SharePointConnector } from "#ai-sdk-node";
import { dummyDocumentsDeleteByCollectionResponse, dummyDocumentsDeleteResponse, dummyDocumentsUpsertChunkResponse, dummyStatesDeleteByCollectionResponse, dummyStatesDeleteResponse, dummyStatesPutResponse, dummyUpdateByQueryResponse, fakeFileDriveItem, fakeFileWebLinkDriveItem, fakeFolderDriveItem, fakeFolderStateRecord, fakeStateRecord, fakeStateWebLinkRecord } from "./stubs/sharepoint-connector.stub";
import { KnowledgeBasePDFDocument, KnowledgeBaseWebLink } from "#ai-sdk-node/knowledge-base";


describe('Sharepoint Connector', () => {
  let store: KnowledgeBaseStore;
  let sharepointConnector: SharePointConnector;

  let documentsDeleteByCollectionStub: sinon.SinonStub;
  let statesDeleteByCollectionStub: sinon.SinonStub;
  let documentDeleteStub: sinon.SinonStub;
  let statesDeleteStub: sinon.SinonStub;
  let documentsUpsertStub: sinon.SinonStub;
  let statesPutStub: sinon.SinonStub;
  let documentsRenameCollectionStub: sinon.SinonStub;
  let statesRenameCollectionStub: sinon.SinonStub;

  before(async () => {
    store = createMockKnowledgeBaseStore();
    sharepointConnector = new SharePointConnector(
      {
        azure: {
          tenantId: 'dummy',
          clientId: 'dummy',
          clientSecret: 'dummy',
        },
        sharepoint: {
          driveId: 'dummy',
          folderId: 'dummy',
        },
      },
      store
    );
  })

  beforeEach(() => {
    documentsDeleteByCollectionStub = sinon.stub(store.documents, 'deleteByCollection')
      .resolves(dummyDocumentsDeleteByCollectionResponse);
    statesDeleteByCollectionStub = sinon.stub(store.states, 'deleteByCollection')
      .resolves(dummyStatesDeleteByCollectionResponse);
    documentDeleteStub = sinon.stub(store.documents, 'delete')
      .resolves(dummyDocumentsDeleteResponse);
    statesDeleteStub = sinon.stub(store.states, 'delete')
      .resolves(dummyStatesDeleteResponse);
    documentsUpsertStub = sinon.stub(store.documents, 'upsert')
      .resolves([dummyDocumentsUpsertChunkResponse]);
    statesPutStub = sinon.stub(store.states, 'put')
      .resolves(dummyStatesPutResponse);
    documentsRenameCollectionStub = sinon.stub(store.documents, 'renameCollection')
      .resolves(dummyUpdateByQueryResponse);
    statesRenameCollectionStub = sinon.stub(store.states, 'renameCollection')
      .resolves(dummyUpdateByQueryResponse);
  })

  afterEach(() => {
    sinon.restore();
  })

  it('should no actions if no folder and no state', async () => {
    // Arrange
    const statesGetFoldersStub = sinon.stub(store.states, 'getFolders').resolves([]);
    const statesGetByParentIdStub = sinon.stub(store.states, 'getByParentId').resolves([]);
    const getFoldersFromRootFolderStub = sinon.stub(sharepointConnector, <any>'getFoldersFromRootFolder').resolves([]);
    const getFilesFromFolderStub = sinon.stub(sharepointConnector, <any>'getFilesFromFolder').resolves([]);

    // Act
    const result = await sharepointConnector.run();

    // Assert
    sinon.assert.calledOnce(getFoldersFromRootFolderStub);
    sinon.assert.calledOnce(statesGetFoldersStub);
    sinon.assert.notCalled(documentsDeleteByCollectionStub);
    sinon.assert.notCalled(statesDeleteByCollectionStub);
    sinon.assert.notCalled(getFilesFromFolderStub);
    sinon.assert.notCalled(statesGetByParentIdStub);
    sinon.assert.notCalled(documentDeleteStub);
    sinon.assert.notCalled(statesDeleteStub);
    sinon.assert.notCalled(documentsUpsertStub);
    sinon.assert.notCalled(statesPutStub);
    expect(result).to.deep.equal({
      status: 'success',
      noChange: { total: 0, files: [] },
      upserted: { total: 0, files: [] },
      removed: { total: 0, files: [] },
    });
  });

  it('should upsert file and put state if new file', async () => {
    // Arrange
    const statesGetFoldersStub = sinon.stub(store.states, 'getFolders').resolves([]);
    const statesGetByParentIdStub = sinon.stub(store.states, 'getByParentId').resolves([]);
    const getFoldersFromRootFolderStub = sinon.stub(sharepointConnector, <any>'getFoldersFromRootFolder')
      .resolves([fakeFolderDriveItem]);
    const getFilesFromFolderStub = sinon.stub(sharepointConnector, <any>'getFilesFromFolder')
      .resolves([fakeFileDriveItem]);

    // Act
    const result = await sharepointConnector.run();

    // Assert
    sinon.assert.calledOnce(getFoldersFromRootFolderStub);
    sinon.assert.calledOnce(statesGetFoldersStub);
    sinon.assert.notCalled(documentsDeleteByCollectionStub);
    sinon.assert.notCalled(statesDeleteByCollectionStub);
    sinon.assert.calledOnceWithExactly(getFilesFromFolderStub, fakeFolderDriveItem.id);
    sinon.assert.calledOnceWithExactly(statesGetByParentIdStub, fakeFolderDriveItem.id);
    sinon.assert.notCalled(documentDeleteStub);
    sinon.assert.notCalled(statesDeleteStub);
    sinon.assert.calledOnceWithMatch(documentsUpsertStub, sinon.match.instanceOf(KnowledgeBasePDFDocument));
    sinon.assert.calledOnceWithExactly(statesPutStub, fakeFileDriveItem, undefined);
    expect(result).to.deep.equal({
      status: 'success',
      noChange: { total: 0, files: [] },
      upserted: {
        total: 1,
        files: [
          {
            id: fakeFileDriveItem.id,
            collection: fakeFolderDriveItem.name,
            name: fakeFileDriveItem.name,
          },
        ],
      },
      removed: { total: 0, files: [] },
    });
  })

  it('should no change if file is not modified', async () => {
    // Arrange
    const statesGetFoldersStub = sinon.stub(store.states, 'getFolders').resolves([fakeFolderStateRecord]);
    const statesGetByParentIdStub = sinon.stub(store.states, 'getByParentId').resolves([fakeStateRecord]);
    const getFoldersFromRootFolderStub = sinon.stub(sharepointConnector, <any>'getFoldersFromRootFolder')
      .resolves([fakeFolderDriveItem]);
    const getFilesFromFolderStub = sinon.stub(sharepointConnector, <any>'getFilesFromFolder')
      .resolves([fakeFileDriveItem]);

    // Act
    const result = await sharepointConnector.run();

    // Assert
    sinon.assert.calledOnce(getFoldersFromRootFolderStub);
    sinon.assert.calledOnce(statesGetFoldersStub);
    sinon.assert.notCalled(documentsDeleteByCollectionStub);
    sinon.assert.notCalled(statesDeleteByCollectionStub);
    sinon.assert.calledOnceWithExactly(getFilesFromFolderStub, fakeFolderDriveItem.id);
    sinon.assert.calledOnceWithExactly(statesGetByParentIdStub, fakeFolderDriveItem.id);
    sinon.assert.notCalled(documentDeleteStub);
    sinon.assert.notCalled(statesDeleteStub);
    sinon.assert.notCalled(documentsUpsertStub);
    sinon.assert.notCalled(statesPutStub);
    expect(result).to.deep.equal({
      status: 'success',
      noChange: {
        total: 1,
        files: [
          {
            id: fakeFileDriveItem.id,
            collection: fakeFolderDriveItem.name,
            name: fakeFileDriveItem.name,
          },
        ],
      },
      upserted: { total: 0, files: [] },
      removed: { total: 0, files: [] },
    });
  })

  it('should upsert and update state file if file is modified', async () => {
    // Arrange
    const statesGetFoldersStub = sinon.stub(store.states, 'getFolders').resolves([fakeFolderStateRecord]);
    const statesGetByParentIdStub = sinon.stub(store.states, 'getByParentId').resolves([fakeStateRecord]);
    const getFoldersFromRootFolderStub = sinon.stub(sharepointConnector, <any>'getFoldersFromRootFolder')
      .resolves([fakeFolderDriveItem]);

    const modifiedFakeFileDriveItem = structuredClone(fakeFileDriveItem);
    modifiedFakeFileDriveItem.lastModifiedDateTime.setSeconds(modifiedFakeFileDriveItem.lastModifiedDateTime.getSeconds() + 1);

    const getFilesFromFolderStub = sinon.stub(sharepointConnector, <any>'getFilesFromFolder')
      .resolves([modifiedFakeFileDriveItem]);

    // Act
    const result = await sharepointConnector.run();

    // Assert
    sinon.assert.calledOnce(getFoldersFromRootFolderStub);
    sinon.assert.calledOnce(statesGetFoldersStub);
    sinon.assert.notCalled(documentsDeleteByCollectionStub);
    sinon.assert.notCalled(statesDeleteByCollectionStub);
    sinon.assert.calledOnceWithExactly(getFilesFromFolderStub, fakeFolderDriveItem.id);
    sinon.assert.calledOnceWithExactly(statesGetByParentIdStub, fakeFolderDriveItem.id);
    sinon.assert.notCalled(documentDeleteStub);
    sinon.assert.notCalled(statesDeleteStub);
    sinon.assert.calledOnceWithMatch(documentsUpsertStub, sinon.match.instanceOf(KnowledgeBasePDFDocument));
    sinon.assert.calledOnceWithExactly(statesPutStub, modifiedFakeFileDriveItem, fakeStateRecord._id);
    expect(result).to.deep.equal({
      status: 'success',
      noChange: { total: 0, files: [] },
      upserted: {
        total: 1,
        files: [
          {
            id: fakeFileDriveItem.id,
            collection: fakeFolderDriveItem.name,
            name: fakeFileDriveItem.name,
          },
        ],
      },
      removed: { total: 0, files: [] },
    });
  })

  it('should delete and remove state if file is deleted', async () => {
    // Arrange
    const statesGetFoldersStub = sinon.stub(store.states, 'getFolders').resolves([fakeFolderStateRecord]);
    const statesGetByParentIdStub = sinon.stub(store.states, 'getByParentId').resolves([fakeStateRecord]);
    const getFoldersFromRootFolderStub = sinon.stub(sharepointConnector, <any>'getFoldersFromRootFolder')
      .resolves([fakeFolderDriveItem]);
    const getFilesFromFolderStub = sinon.stub(sharepointConnector, <any>'getFilesFromFolder')
      .resolves([]);

    // Act
    const result = await sharepointConnector.run();

    // Assert
    sinon.assert.calledOnce(getFoldersFromRootFolderStub);
    sinon.assert.calledOnce(statesGetFoldersStub);
    sinon.assert.notCalled(documentsDeleteByCollectionStub);
    sinon.assert.notCalled(statesDeleteByCollectionStub);
    sinon.assert.calledOnceWithExactly(getFilesFromFolderStub, fakeFolderDriveItem.id);
    sinon.assert.calledOnceWithExactly(statesGetByParentIdStub, fakeFolderDriveItem.id);
    sinon.assert.calledOnceWithMatch(documentDeleteStub, fakeFileDriveItem.id);
    sinon.assert.calledOnceWithExactly(statesDeleteStub, fakeStateRecord._id);
    sinon.assert.notCalled(documentsUpsertStub);
    sinon.assert.notCalled(statesPutStub);
    expect(result).to.deep.equal({
      status: 'success',
      noChange: { total: 0, files: [] },
      upserted: { total: 0, files: [] },
      removed: {
        total: 1,
        files: [
          {
            id: fakeFileDriveItem.id,
            collection: fakeFolderDriveItem.name,
            name: fakeFileDriveItem.name,
          },
        ],
      },
    });
  })

  it('should delete collection if folder is deleted', async () => {
    // Arrange
    const statesGetFoldersStub = sinon.stub(store.states, 'getFolders').resolves([fakeFolderStateRecord]);
    const statesGetByParentIdStub = sinon.stub(store.states, 'getByParentId').resolves([fakeStateRecord]);
    const getFoldersFromRootFolderStub = sinon.stub(sharepointConnector, <any>'getFoldersFromRootFolder')
      .resolves([]);
    const getFilesFromFolderStub = sinon.stub(sharepointConnector, <any>'getFilesFromFolder')
      .resolves([]);

    // Act
    const result = await sharepointConnector.run();

    // Assert
    sinon.assert.calledOnce(getFoldersFromRootFolderStub);
    sinon.assert.calledOnce(statesGetFoldersStub);
    sinon.assert.calledOnceWithExactly(documentsDeleteByCollectionStub, fakeStateRecord._source.parentId);
    sinon.assert.calledOnceWithExactly(statesGetByParentIdStub, fakeStateRecord._source.parentId);
    sinon.assert.calledOnceWithExactly(statesDeleteByCollectionStub, fakeStateRecord._source.parentId);
    sinon.assert.notCalled(getFilesFromFolderStub);
    sinon.assert.notCalled(documentDeleteStub);
    sinon.assert.notCalled(statesDeleteStub);
    sinon.assert.notCalled(documentsUpsertStub);
    sinon.assert.notCalled(statesPutStub);
    expect(result).to.deep.equal({
      status: 'success',
      noChange: { total: 0, files: [] },
      upserted: { total: 0, files: [] },
      removed: {
        total: 1,
        files: [
          {
            id: fakeFileDriveItem.id,
            collection: fakeFolderDriveItem.name,
            name: fakeFileDriveItem.name,
          },
        ],
      },
    });
  })

  it('should no change if link is upserted and not expired', async () => {
   // Arrange
    const statesGetFoldersStub = sinon.stub(store.states, 'getFolders').resolves([fakeFolderStateRecord]);
    const statesGetByParentIdStub = sinon.stub(store.states, 'getByParentId').resolves([fakeStateWebLinkRecord]);
    const getFoldersFromRootFolderStub = sinon.stub(sharepointConnector, <any>'getFoldersFromRootFolder')
      .resolves([fakeFolderDriveItem]);
    const getFilesFromFolderStub = sinon.stub(sharepointConnector, <any>'getFilesFromFolder')
      .resolves([fakeFileWebLinkDriveItem]);

    // Act
    const result = await sharepointConnector.run();

    // Assert
    sinon.assert.calledOnce(getFoldersFromRootFolderStub);
    sinon.assert.calledOnce(statesGetFoldersStub);
    sinon.assert.notCalled(documentsDeleteByCollectionStub);
    sinon.assert.notCalled(statesDeleteByCollectionStub);
    sinon.assert.calledOnceWithExactly(getFilesFromFolderStub, fakeFolderDriveItem.id);
    sinon.assert.calledOnceWithExactly(statesGetByParentIdStub, fakeFolderDriveItem.id);
    sinon.assert.notCalled(documentDeleteStub);
    sinon.assert.notCalled(statesDeleteStub);
    sinon.assert.notCalled(documentsUpsertStub);
    sinon.assert.notCalled(statesPutStub);
    expect(result).to.deep.equal({
      status: 'success',
      noChange: {
        total: 1,
        files: [
          {
            id: fakeFileWebLinkDriveItem.id,
            collection: fakeFolderDriveItem.name,
            name: fakeFileWebLinkDriveItem.name,
          },
        ],
      },
      upserted: { total: 0, files: [] },
      removed: { total: 0, files: [] },
    });
  });


  it('should upsert if link is upserted and expired', async () => {
   // Arrange
    const expiredFakeStateWebLinkRecord = structuredClone(fakeStateWebLinkRecord);
    const expiredDate = new Date()
    expiredDate.setHours(new Date().getHours() - 36);
    expiredFakeStateWebLinkRecord._source.lastUpsertedDateTime = expiredDate.toISOString();

    const statesGetFoldersStub = sinon.stub(store.states, 'getFolders').resolves([fakeFolderStateRecord]);
    const statesGetByParentIdStub = sinon.stub(store.states, 'getByParentId').resolves([expiredFakeStateWebLinkRecord]);
    const getFoldersFromRootFolderStub = sinon.stub(sharepointConnector, <any>'getFoldersFromRootFolder')
      .resolves([fakeFolderDriveItem]);
    const getFilesFromFolderStub = sinon.stub(sharepointConnector, <any>'getFilesFromFolder')
      .resolves([fakeFileWebLinkDriveItem]);

    // Act
    const result = await sharepointConnector.run();

    // Assert
    sinon.assert.calledOnce(getFoldersFromRootFolderStub);
    sinon.assert.calledOnce(statesGetFoldersStub);
    sinon.assert.notCalled(documentsDeleteByCollectionStub);
    sinon.assert.notCalled(statesDeleteByCollectionStub);
    sinon.assert.calledOnceWithExactly(getFilesFromFolderStub, fakeFolderDriveItem.id);
    sinon.assert.calledOnceWithExactly(statesGetByParentIdStub, fakeFolderDriveItem.id);
    sinon.assert.notCalled(documentDeleteStub);
    sinon.assert.notCalled(statesDeleteStub);
    sinon.assert.calledOnceWithMatch(documentsUpsertStub, sinon.match.instanceOf(KnowledgeBaseWebLink));
    sinon.assert.calledOnceWithExactly(statesPutStub, fakeFileWebLinkDriveItem, expiredFakeStateWebLinkRecord._id);
    expect(result).to.deep.equal({
      status: 'success',
      noChange: { total: 0, files: [] },
      upserted: {
        total: 1,
        files: [
          {
            id: fakeFileWebLinkDriveItem.id,
            collection: fakeFolderDriveItem.name,
            name: fakeFileWebLinkDriveItem.name,
          },
        ],
      },
      removed: { total: 0, files: [] },
    });
  });

  it('should treat as modified if file is renamed', async () => {
    // Arrange
    const statesGetFoldersStub = sinon.stub(store.states, 'getFolders').resolves([fakeFolderStateRecord]);
    const statesGetByParentIdStub = sinon.stub(store.states, 'getByParentId').resolves([fakeStateRecord]);
    const getFoldersFromRootFolderStub = sinon.stub(sharepointConnector, <any>'getFoldersFromRootFolder')
      .resolves([fakeFolderDriveItem]);
    const newName = 'new name';
    const renamedFakeFileDriveItem = structuredClone(fakeFileDriveItem);
    renamedFakeFileDriveItem.name = newName;
    renamedFakeFileDriveItem.lastModifiedDateTime.setSeconds(renamedFakeFileDriveItem.lastModifiedDateTime.getSeconds() + 1);
    const getFilesFromFolderStub = sinon.stub(sharepointConnector, <any>'getFilesFromFolder')
      .resolves([renamedFakeFileDriveItem]);

    // Act
    const result = await sharepointConnector.run();

    // Assert
    sinon.assert.calledOnce(getFoldersFromRootFolderStub);
    sinon.assert.calledOnce(statesGetFoldersStub);
    sinon.assert.notCalled(documentsDeleteByCollectionStub);
    sinon.assert.notCalled(statesDeleteByCollectionStub);
    sinon.assert.calledOnceWithExactly(getFilesFromFolderStub, fakeFolderDriveItem.id);
    sinon.assert.calledOnceWithExactly(statesGetByParentIdStub, fakeFolderDriveItem.id);
    sinon.assert.notCalled(documentDeleteStub);
    sinon.assert.notCalled(statesDeleteStub);
    sinon.assert.calledOnceWithMatch(documentsUpsertStub, sinon.match.instanceOf(KnowledgeBasePDFDocument));
    sinon.assert.calledOnceWithExactly(statesPutStub, renamedFakeFileDriveItem, fakeStateRecord._id);
    expect(result).to.deep.equal({
      status: 'success',
      noChange: { total: 0, files: [] },
      upserted: {
        total: 1,
        files: [
          {
            id: renamedFakeFileDriveItem.id,
            collection: fakeFolderDriveItem.name,
            name: newName,
          },
        ],
      },
      removed: { total: 0, files: [] },
    });
  });

  it('should rename the collection of file and file states if folder is renamed', async () => {
    // Arrange
    const statesGetFoldersStub = sinon.stub(store.states, 'getFolders').resolves([fakeFolderStateRecord]);
    const statesGetByParentIdStub = sinon.stub(store.states, 'getByParentId').resolves([fakeStateRecord]);

    const newName = 'new name';
    const renamedFakeFolderDriveItem = structuredClone(fakeFolderDriveItem);
    renamedFakeFolderDriveItem.name = newName;
    const getFoldersFromRootFolderStub = sinon.stub(sharepointConnector, <any>'getFoldersFromRootFolder')
      .resolves([renamedFakeFolderDriveItem]);

    const renamedFolderFakeFileDriveItem = structuredClone(fakeFileDriveItem);
    renamedFolderFakeFileDriveItem.parentReference.name = newName;
    const getFilesFromFolderStub = sinon.stub(sharepointConnector, <any>'getFilesFromFolder')
      .resolves([renamedFolderFakeFileDriveItem]);

    // Act
    const result = await sharepointConnector.run();

    // Assert
    sinon.assert.calledOnce(getFoldersFromRootFolderStub);
    sinon.assert.calledOnce(statesGetFoldersStub);
    sinon.assert.notCalled(documentsDeleteByCollectionStub);
    sinon.assert.notCalled(statesDeleteByCollectionStub);
    sinon.assert.calledOnceWithExactly(documentsRenameCollectionStub, renamedFakeFolderDriveItem.id, newName);
    sinon.assert.calledOnceWithExactly(statesRenameCollectionStub, renamedFakeFolderDriveItem.id, newName);
    sinon.assert.calledOnceWithExactly(getFilesFromFolderStub, fakeFolderDriveItem.id);
    sinon.assert.calledOnceWithExactly(statesGetByParentIdStub, fakeFolderDriveItem.id);
    sinon.assert.notCalled(documentDeleteStub);
    sinon.assert.notCalled(statesDeleteStub);
    sinon.assert.notCalled(documentsUpsertStub);
    sinon.assert.notCalled(statesPutStub);
    expect(result).to.deep.equal({
      status: 'success',
      noChange: {
        total: 1,
        files: [
          {
            id: renamedFolderFakeFileDriveItem.id,
            collection: newName,
            name: renamedFolderFakeFileDriveItem.name,
          },
        ],
      },
      upserted: { total: 0, files: [] },
      removed: { total: 0, files: [] },
    });
  });
});