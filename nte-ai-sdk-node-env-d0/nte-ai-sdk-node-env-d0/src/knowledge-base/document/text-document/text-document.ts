import axios from 'axios';
import { DriveItem } from "@microsoft/msgraph-sdk/models/index.js";
import logger from "../../../logger/logger.js";
import textract from "@nosferatu500/textract";
import { KnowledgeBaseDocument } from '../document.js';

const _log = logger.child({ module: 'ai-sdk-node.knowledge-base.document.text-document' });

export class KnowledgeBaseTextDocument extends KnowledgeBaseDocument {
  constructor(doc: DriveItem) {
    super(doc);
  }

  async getMetadata(): Promise<KnowledgeBaseDocumentMetadata> {
    return {
      title: this.doc.webUrl?.substring(this.doc.webUrl.lastIndexOf('/') + 1) || '',
      webUrl: this.doc.webUrl || ''
    }
  }
  // the fucntions below are to be inherited by all file types
  getCollection(): string {
    return this.doc.parentReference?.name || '';
  }

  getParentId(): string {
    return this.doc.parentReference?.id || '';
  }

  getDocId(): string {
    return this.doc.id || '';
  }

  // the functions below are to be inherited by text file types only
  async getContent(): Promise<string> {
    const docWithDownloadUrl = this.doc as DriveItem & { '@microsoft.graph.downloadUrl': string };
    const fileUrl = docWithDownloadUrl['@microsoft.graph.downloadUrl'];
    const response = await this.processFile(fileUrl);
    return response || '';
  }

  getMimeType(): string {
    return '';
  }

  protected async processFile(fileUrl: string) {
    try {
      const file = await this.downloadFile(fileUrl);
      const content = await this.parseFileToText(file);
      return content;
    } catch (error) {
      const msg = `Failed to upsert file: ${fileUrl} ${error}`;
      _log.error({ msg: msg, status_code: 400 });
    }
  }

  protected async downloadFile(fileUrl: string) {
    try {
      const response = await axios.get(fileUrl, { responseType: 'arraybuffer' });
      return response.data;
    } catch (error) {
      const msg = `Failed to download file: ${fileUrl} ${error}`;
      _log.error({ msg: msg, status_code: 400 })
    }
  }

  private parseFileToText(file: string): Promise<string> {
    const docBuffer = Buffer.from(file);
    const mimeType = this.getMimeType();
    return new Promise((resolve, reject) => {
      textract.fromBufferWithMime(mimeType, docBuffer, function (error, text) {
        if (error) {
          const msg = `Failed to parse file content: ${error}`;
          _log.error({ msg: msg, status_code: 400 })
          reject(error)
        } else {
          resolve(text)
        }
      })
    })
  }

}

export interface KnowledgeBaseDocumentMetadata {
  title: string;
  webUrl: string;
}