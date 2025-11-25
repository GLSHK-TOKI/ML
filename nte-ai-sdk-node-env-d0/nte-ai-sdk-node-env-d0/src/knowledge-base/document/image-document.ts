import { KnowledgeBaseDocument, KnowledgeBaseDocumentMetadata } from './document.js';
import { DriveItem } from "@microsoft/msgraph-sdk/models/index.js";
import path from 'path';
import fs from 'fs';
import axios from 'axios';
import logger from '../../logger/logger.js';

const _log = logger.child({ module: 'ai-sdk-node.image-document' });

/**
 * Knowledge Base Image Document for raw image files (.jpg, .jpeg, .png)
 * Handles direct image file processing for vision embeddings
 */
export class KnowledgeBaseImageDocument extends KnowledgeBaseDocument {
  static readonly extension = 'jpg,jpeg,png';
  
  static readonly supportedExtensions = [
    'jpg', 'jpeg', 'png'
  ];

  constructor(file: DriveItem) {
    super(file);
  }

  /**
   * Get metadata for the image document
   */
  async getMetadata(chunkContent?: string): Promise<KnowledgeBaseDocumentMetadata> {
    return {
      title: this.doc.name || 'Untitled Image',
      webUrl: this.doc.webUrl || '',
      startPage: 1,
      endPage: 1
    };
  }

  /**
   * Get collection name (parent folder name)
   */
  getCollection(): string {
    return this.doc.parentReference?.name || '';
  }

  /**
   * Get parent ID (folder ID)
   */
  getParentId(): string {
    return this.doc.parentReference?.id || '';
  }

  /**
   * Get document ID (file ID)
   */
  getDocId(): string {
    return this.doc.id || '';
  }

  /**
   * Get content - for images, return empty string as images don't have text content
   * Image processing happens via vision embeddings, not text chunking
   */
  async getContent(): Promise<string> {
    return '';
  }

  /**
   * Download the image file from SharePoint
   * @param fileUrl - The download URL for the image file
   * @param originalFileName - Optional original filename to use
   * @returns Path to the downloaded image file
   */
  async downloadImageFile(fileUrl: string, originalFileName?: string): Promise<string> {
    try {
      const response = await axios.get(fileUrl, { responseType: 'arraybuffer' });

      // Create image_aisdk directory if it doesn't exist
      const imageAisdkDir = path.join(process.cwd(), 'image_aisdk');
      if (!fs.existsSync(imageAisdkDir)) {
        fs.mkdirSync(imageAisdkDir, { recursive: true });
      }
      
      // Use original filename if provided, otherwise use temp filename
      let fileName: string;
      if (originalFileName) {
        // Sanitize the filename to remove any invalid characters
        fileName = originalFileName.replace(/[<>:"/\\|?*]/g, '_');
        // Add timestamp to avoid conflicts
        const timestamp = Date.now();
        const extensionIndex = fileName.lastIndexOf('.');
        if (extensionIndex > 0) {
          fileName = `${fileName.substring(0, extensionIndex)}_${timestamp}${fileName.substring(extensionIndex)}`;
        } else {
          fileName = `${fileName}_${timestamp}`;
        }
      } else {
        const extension = this.getImageExtension();
        fileName = `temp_image_${Date.now()}_${Math.random().toString(36).substring(2)}.${extension}`;
      }
      
      const filePath = path.join(imageAisdkDir, fileName);
      
      fs.writeFileSync(filePath, response.data);
      
      _log.debug({
        msg: `Successfully downloaded image file to: ${filePath}`,
        status_code: 200
      });
      
      return filePath;
    } catch (error) {
      const msg = `Failed to download image file: ${fileUrl} ${error}`;
      _log.error({ msg: msg, status_code: 500 });
      throw new Error(msg);
    }
  }

  /**
   * Get the image file extension from the file URL
   */
  private getImageExtension(): string {
    const webUrl = this.doc.webUrl || '';
    const lastSlashIndex = webUrl.lastIndexOf('/');
    if (lastSlashIndex === -1) return 'jpg'; // default extension
  
    const fileName = webUrl.substring(lastSlashIndex + 1);
    const dotIndex = fileName.lastIndexOf('.');
    if (dotIndex === -1) return 'jpg'; // default extension
  
    if (fileName.includes('&action')) {
      const actionIndex = fileName.lastIndexOf('&action');
      return fileName.substring(dotIndex + 1, actionIndex).toLowerCase();
    }    

    return fileName.substring(dotIndex + 1).toLowerCase();
  }

  /**
   * Extract and download the image file
   * Downloads the raw image file and returns its path for embedding processing
   * Caller is responsible for cleaning up the returned image file
   * 
   * @returns Promise with array containing the single image file path
   */
  async extractPath(): Promise<string[]> {
    const docWithDownloadUrl = this.doc as DriveItem & { '@microsoft.graph.downloadUrl': string };
    const fileUrl = docWithDownloadUrl['@microsoft.graph.downloadUrl'];

    try {
      // Download the raw image file
      const imagePath = await this.downloadImageFile(fileUrl, this.doc.name || undefined);
      
      _log.debug({
        msg: `Successfully extracted image path: ${imagePath}`,
        status_code: 200
      });
      
      return [imagePath];
    } catch (error) {
      const msg = `Failed to extract image path for ${this.doc.name}: ${error}`;
      _log.error({ msg: msg, status_code: 500, err: error });
      throw error;
    }
  }
}