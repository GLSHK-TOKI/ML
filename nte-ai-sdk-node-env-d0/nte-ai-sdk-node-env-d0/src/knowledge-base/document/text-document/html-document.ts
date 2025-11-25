import { KnowledgeBaseTextDocument } from './text-document.js';
import { DriveItem } from "@microsoft/msgraph-sdk/models/index.js";

export class KnowledgeBaseHTMLDocument extends KnowledgeBaseTextDocument {
	static extension = "html";

	constructor(doc: DriveItem) {
		super(doc);
	}

	getMimeType() {
    return "text/html";
  }
}