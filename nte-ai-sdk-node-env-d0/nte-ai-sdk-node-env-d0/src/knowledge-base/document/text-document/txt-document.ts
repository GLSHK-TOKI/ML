import { KnowledgeBaseTextDocument } from './text-document.js';
import { DriveItem } from "@microsoft/msgraph-sdk/models/index.js";

export class KnowledgeBaseTXTDocument extends KnowledgeBaseTextDocument {
  static extension = 'txt';

  constructor(doc: DriveItem) {
    super(doc);
  }
  
  getMimeType() {
    return "text/plain";
  }
}