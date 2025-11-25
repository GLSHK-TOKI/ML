import { KnowledgeBaseTextDocument } from './text-document.js';
import { DriveItem } from "@microsoft/msgraph-sdk/models/index.js";

export class KnowledgeBaseRTFDocument extends KnowledgeBaseTextDocument {
  static extension = "rtf";

  constructor(doc: DriveItem) {
    super(doc);
  }

  async getMetadata() {
    return {
      title: this.doc.webUrl?.substring(this.doc.webUrl.lastIndexOf('file=') + 5, this.doc.webUrl.lastIndexOf('&action')) || '',
      webUrl: this.doc.webUrl || ''
    }
  }

  getMimeType() {
    return "application/rtf";
  }
}