import { DriveItem } from "@microsoft/msgraph-sdk/models/index.js";

export abstract class KnowledgeBaseDocument {
  static extension: string;
  readonly doc: DriveItem;

  constructor(doc: DriveItem) {
    this.doc = doc;
  }

  abstract getMetadata(chunkContent?: string): Promise<KnowledgeBaseDocumentMetadata>;
  abstract getCollection(): string;
  abstract getParentId(): string;
  abstract getDocId(): string;
  abstract getContent(): Promise<string>;
}

export interface KnowledgeBaseDocumentMetadata {
  title: string;
  webUrl: string;
  startPage?: number;
  endPage?: number;
}