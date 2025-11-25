import { UpdateByQueryResponse, WriteResponseBase } from "@elastic/elasticsearch/lib/api/types";

export const fakeFolderDriveItem = {
  createdDateTime: new Date('2024-06-31T01:54:20Z'),
  id: 'item-id-folder',
  lastModifiedDateTime: new Date('2024-06-31T01:54:20Z'),
  name: 'some-folder',
  parentReference: {
    id: 'parent-id',
    name: 'parent-name',
  },
  webUrl: 'https://dummy-url',
  folder: {
    childCount: 1,
  },
};

export const fakeFileDriveItem = {
  createdDateTime: new Date('2024-07-31T01:54:20Z'),
  id: 'item-id-file',
  lastModifiedDateTime: new Date('2024-07-31T01:54:20Z'),
  name: 'some-document.pdf',
  parentReference: {
    id: 'item-id-folder',
    name: 'some-folder',
  },
  webUrl: 'https://dummy-url.pdf',
};

export const fakeFileWebLinkDriveItem = {
  createdDateTime: new Date('2024-07-31T01:54:20Z'),
  id: 'item-id-file-2',
  lastModifiedDateTime: new Date('2024-07-31T01:54:20Z'),
  name: 'some-document.url',
  parentReference: {
    id: 'item-id-folder',
    name: 'some-folder',
  },
  webUrl: 'https://dummy-url.url',
};

export const fakeStateRecord = {
  _index: 'dummy-index',
  _id: 'dummy-id',
  _score: 1,
  _source: {
    parentId: fakeFolderDriveItem.id,
    id: fakeFileDriveItem.id,
    collection: fakeFolderDriveItem.name,
    name: fakeFileDriveItem.name,
    webUrl: fakeFileDriveItem.webUrl,
    lastUpsertedDateTime: new Date().toISOString(),
    lastModifiedDateTime: fakeFileDriveItem.lastModifiedDateTime.toISOString(),
  },
};

export const fakeStateWebLinkRecord = {
  _index: 'dummy-index',
  _id: 'dummy-id',
  _score: 1,
  _source: {
    parentId: fakeFolderDriveItem.id,
    id: fakeFileWebLinkDriveItem.id,
    collection: fakeFolderDriveItem.name,
    name: fakeFileWebLinkDriveItem.name,
    webUrl: fakeFileWebLinkDriveItem.webUrl,
    lastUpsertedDateTime: new Date().toISOString(),
    lastModifiedDateTime: fakeFileWebLinkDriveItem.lastModifiedDateTime.toISOString(),
  },
};

export const fakeFolderStateRecord = {
  id: fakeFolderDriveItem.id,
  name: fakeFolderDriveItem.name,
};

export const dummyDocumentsDeleteByCollectionResponse = {
  took: 6,
  timed_out: false,
  total: 0,
  deleted: 1,
};

export const dummyStatesDeleteByCollectionResponse = {
  took: 6,
  timed_out: false,
  total: 0,
  deleted: 1,
};

export const dummyDocumentsDeleteResponse = {
  took: 6,
  timed_out: false,
  total: 0,
  deleted: 1,
};

export const dummyStatesDeleteResponse: WriteResponseBase = {
  _index: 'dummy-index',
  _id: 'dummy-id',
  _version: 1,
  result: 'deleted',
  _shards: {
    total: 2,
    successful: 2,
    failed: 0,
  },
};

export const dummyDocumentsUpsertChunkResponse: WriteResponseBase = {
  _index: 'dummy-index',
  _id: 'dummy-id',
  _version: 1,
  result: 'created',
  _shards: {
    total: 2,
    successful: 2,
    failed: 0,
  },
};

export const dummyStatesPutResponse: WriteResponseBase = {
  _index: 'dummy-index',
  _id: 'dummy-id',
  _version: 1,
  result: 'created',
  _shards: {
    total: 2,
    successful: 2,
    failed: 0
  },
}

export const dummyUpdateByQueryResponse: UpdateByQueryResponse = {
  took: 6,
  timed_out: false,
  total: 0,
  updated: 1,
  deleted: 0,
  batches: 1,
  version_conflicts: 0,
  noops: 0,
  retries: {
    bulk: 0,
    search: 0
  },
  throttled_millis: 0,
  requests_per_second: -1,
  throttled_until_millis: 0,
  failures: [],
};