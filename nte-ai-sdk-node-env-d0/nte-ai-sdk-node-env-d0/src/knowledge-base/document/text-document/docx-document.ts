import { KnowledgeBaseTextDocument } from './text-document.js';
import { DriveItem } from "@microsoft/msgraph-sdk/models/index.js";

export class KnowledgeBaseDOCXDocument extends KnowledgeBaseTextDocument {
  static extension = 'docx';

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
    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
  }
}