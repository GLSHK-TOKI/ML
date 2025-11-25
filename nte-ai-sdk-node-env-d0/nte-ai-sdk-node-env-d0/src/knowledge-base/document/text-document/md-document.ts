import { KnowledgeBaseTextDocument } from './text-document.js';
import { DriveItem } from "@microsoft/msgraph-sdk/models/index.js";
import logger from "../../../logger/logger.js";

const _log = logger.child({ module: 'ai-sdk-node.knowledge-base.document.md-document' });

export class KnowledgeBaseMDDocument extends KnowledgeBaseTextDocument {
  static extension = "md";

  constructor(doc: DriveItem) {
    super(doc);
  }

  async processFile(fileUrl: string) {
    try {
      const file = await this.downloadFile(fileUrl);
      const content = file.toString();
      return content;
    } catch (error) {
      const msg = `Failed to upsert file: ${fileUrl} ${error}`;
      _log.error({msg: msg, status_code: 400});
    }
  }

  getMimeType() {
    return "text/markdown";
  }
}