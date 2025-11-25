import { KnowledgeBaseDocument } from './document.js';
import fs from "fs";
import path from "path";
import os from "os";
import axios from 'axios';
import { DriveItem } from "@microsoft/msgraph-sdk/models/index.js";
import logger from "../../logger/logger.js";
import { Document } from "@langchain/core/documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import * as Jimp from 'jimp';
import * as mupdf from 'mupdf';
const _log = logger.child({ module: 'ai-sdk-node.knowledge-base.document.pdf-document' });

export class KnowledgeBasePDFDocument extends KnowledgeBaseDocument {
  static extension = 'pdf';

  docs: Document[] = []; // Array of Document objects representing each page of the document from PDFLoader
  content: string = ''; // Combined content of the document, separated by "\n\n"

  constructor(doc: DriveItem) {
    super(doc);
  }

  /**
   * Retrieves metadata for the document.
   *
   * @param chunkContent - The content chunk to analyze. If undefined, only the title and web URL are returned.
   * @returns An object containing the document's metadata, including title, web URL, and optionally start and end page numbers.
   */
  async getMetadata(chunkContent?: string) {
    if (chunkContent === undefined) {
      return {
        title: this.doc.webUrl?.substring(this.doc.webUrl.lastIndexOf('/') + 1) || '',
        webUrl: this.doc.webUrl || ''
      }    
    } else {
      const {startPage, endPage} = this.getPageNo(chunkContent)
      return {
        title: this.doc.webUrl?.substring(this.doc.webUrl.lastIndexOf('/') + 1) || '',
        webUrl: this.doc.webUrl || '',
        startPage: startPage,
        endPage: endPage
      }
    }
  }
  
  getCollection(): string {
    return this.doc.parentReference?.name || '';
  }

  getParentId(): string {
    return this.doc.parentReference?.id || '';
  }

  getDocId(): string {
    return this.doc.id || '';
  }

  async getContent(): Promise<string> {
    /**
    * Retrieves the content of the document.
    *
    * This method asynchronously fetches the document, extracts the content from each page,
    * concatenates the content with double newlines, and stores it in the `content` property.
    *
    * @returns {Promise<string>} A promise that resolves to the concatenated content of the document.
    */
    const docs = await this.getDoc();
    const content = docs.map((doc) => doc.pageContent).join("\n\n");
    this.content = content;
    return content;
  }

  async getDoc(): Promise<Document[]> {
    /**
    * Retrieves the document as an array of `Document` objects.
    * 
    * This method fetches the document from a URL obtained from the `DriveItem` object,
    * processes the PDF file, and returns the result as an array of `Document` objects.
    * 
    * @returns {Promise<Document[]>} A promise that resolves to an array of `Document` objects.
    */
    const docWithDownloadUrl = this.doc as DriveItem & { '@microsoft.graph.downloadUrl': string };
    const fileUrl = docWithDownloadUrl['@microsoft.graph.downloadUrl'];
    const response = await this.processPdfFile(fileUrl);
    return response || [];
  }

  async extractPath(): Promise<string[]> {
    /**
    * Extracts and returns the file paths of images extracted from the PDF document.
    * Downloads the PDF file, extracts page images, and cleans up the PDF file.
    * Note: The caller is responsible for cleaning up the extracted image files.
    * 
    * @returns An array containing the file paths of extracted page images.
    */
    const docWithDownloadUrl = this.doc as DriveItem & { '@microsoft.graph.downloadUrl': string };
    const fileUrl = docWithDownloadUrl['@microsoft.graph.downloadUrl'];
    
    // Use default image output directory
    const imageOutputDirectory = '/tmp';
    
    // Download the PDF file
    const pdfPath = await this.downloadFile(fileUrl);
    
    try {
      // Extract page images from the downloaded PDF
      return await KnowledgeBasePDFDocument.extractPageImages(
        pdfPath,
        imageOutputDirectory
      );
    } finally {
      // Clean up the downloaded PDF file
      if (pdfPath && fs.existsSync(pdfPath)) {
        fs.unlinkSync(pdfPath);
        _log.debug({ msg: `Cleaned up PDF file: ${pdfPath}` });
      }
    }
  }

  async processPdfFile(fileUrl: string) {
    let pdfPath: string | undefined;
    try {
      pdfPath = await this.downloadFile(fileUrl);
      if (pdfPath && fs.existsSync(pdfPath)) {
        const docs = await this.parsePdfToDocs(pdfPath);
        return docs;
      } else {
        const msg = `Failed to download PDF file or file doesn't exist: ${fileUrl}`;
        _log.error({msg: msg, status_code: 400});
        throw new Error(msg);
      }
    } finally {
      // Clean up the temporary file
      if (pdfPath && fs.existsSync(pdfPath)) {
        try {
          fs.unlinkSync(pdfPath);
          _log.debug({ msg: `Cleaned up temporary file: ${pdfPath}` });
        } catch (cleanupError) {
          _log.warn({ msg: `Failed to clean up temporary file: ${pdfPath} - ${cleanupError}` });
        }
      }
    }
  }

  async parsePdfToDocs(filePath: string){
    try {
      const loader = new PDFLoader(filePath);
      const docs = await loader.load();
      this.docs = docs;
      return docs
    } catch (error) {
      const msg = `Failed to parse PDF content: ${error}`;
      _log.error({msg: msg, status_code: 400})
      throw error;
    }
  }

  async downloadFile(fileUrl: string) {
    // Download file from fileUrl
    try {
      const response = await axios.get(fileUrl, { responseType: 'arraybuffer' });

      // Create tmp directory if it doesn't exist
      let tmpDir = '/tmp';
      try {
        if (!fs.existsSync(tmpDir)) {
          fs.mkdirSync(tmpDir, { recursive: true });
        }
      } catch (error) {
        // If we can't create in /tmp directory, fall back to OS temp directory
        console.warn(`Failed to create temp directory at ${tmpDir}, falling back to OS temp directory:`, error);
        tmpDir = os.tmpdir();
      }
      
      const fileName = `temp_${Date.now()}_${Math.random().toString(36).substring(2)}.pdf`;
      const filePath = path.join(tmpDir, fileName);
      
      fs.writeFileSync(filePath, response.data);

      return filePath;
    } catch (error) {
      const msg = `Failed to download file: ${fileUrl} ${error}`;
      _log.error({msg: msg, status_code: 400});
      throw error; // Re-throw the error so calling code can handle it
    }
  }

  /**
   * Retrieves the start and end page numbers where a given substring is found within the document.
   *
   * @param substring - The substring to search for within the document content.
   * @returns An object containing the start and end page numbers where the substring is found.
   *          If the substring is not found, both startPage and endPage will be -1.
   */
  getPageNo(substring: string): { startPage: number, endPage: number } {
    const startIndex = this.content.indexOf(substring);
    const endIndex = startIndex + substring.length;
    let startPage = -1;
    let endPage = -1;
    let currentLength = 0;
    for (let pageIdx = 0; pageIdx < this.docs.length; pageIdx++) {
      const doc = this.docs[pageIdx];
      currentLength += doc.pageContent.length + 2;
      if (startPage === -1 && currentLength > startIndex) {
        startPage = pageIdx + 1;
      }
      if (startPage !== -1 && endIndex - currentLength <= doc.pageContent.length) {
        endPage = pageIdx + 1;
        break;
      }
      if (pageIdx === this.docs.length -1) {
        endPage = pageIdx + 1;
        break;
      }
    }
    return { startPage, endPage };
  }

  /**
   * Extract page images from a single PDF file
   * 
   * @param pdfPath Path to the PDF file to process
   * @param outputDirectory Directory where extracted images will be saved
   * @returns Array of extracted image file paths
   */
  static async extractPageImages(
    pdfPath: string, 
    outputDirectory: string
  ): Promise<string[]> {

    if (!fs.existsSync(pdfPath)) {
      _log.warn({ msg: `PDF file not found: ${pdfPath}` });
      return [];
    }

    const allImgPaths: string[] = [];

    try {
      // Use the correct MuPDF API
      const pdfBuffer = fs.readFileSync(pdfPath);
      
      // Create MuPDF buffer from Node.js buffer and open document
      const mupdfBuffer = new mupdf.Buffer(pdfBuffer);
      const doc = mupdf.Document.openDocument(mupdfBuffer, "application/pdf");
      const numPages = doc.countPages();

      const pdfName = path.parse(path.basename(pdfPath)).name;

      // Ensure the output directory exists
      if (!fs.existsSync(outputDirectory)) {
        fs.mkdirSync(outputDirectory, { recursive: true });
      }

      // Process each page
      for (let pageNum = 0; pageNum < numPages; pageNum++) {
        const page = doc.loadPage(pageNum);
        
        // Create identity matrix using static method
        const matrix = mupdf.Matrix.identity;
        
        // Convert page to pixmap with proper parameters
        const pixmap = page.toPixmap(matrix, mupdf.ColorSpace.DeviceRGB, false);
        
        // Get JPEG data as buffer
        const jpegData = pixmap.asJPEG(90);
        
        const pagePath = path.join(outputDirectory, `${pdfName}_page_${pageNum.toString().padStart(3, '0')}.jpg`);
        
        // Write JPEG data to file
        fs.writeFileSync(pagePath, jpegData);
        
        allImgPaths.push(pagePath);
      }
      
    } catch (error) {
      _log.error({ 
        msg: `Error processing PDF ${pdfPath}: ${error}`, 
        status_code: 500 
      });
      throw error;
    }

    return allImgPaths;
  }

  /**
   * Convert a buffer to base64 string with optional format specification
   * 
   * @param buffer Image buffer to convert
   * @param format Optional image format (default: 'jpg')
   * @returns Promise resolving to data URL string with base64 encoded image
   */
  static async base64FromBufferStatic(buffer: Buffer, format: string = 'jpg'): Promise<string> {
    try {
      // Use Jimp to process the image buffer and convert to base64
      const image = await Jimp.Jimp.read(buffer);
      
      // Get the MIME type based on format - use proper types from JimpMime
      let mimeType: keyof typeof Jimp.JimpMime;
      switch (format.toLowerCase()) {
        case 'jpg':
        case 'jpeg':
          mimeType = 'jpeg';
          break;
        case 'png':
          mimeType = 'png';
          break;
        case 'gif':
          mimeType = 'gif';
          break;
        case 'bmp':
          mimeType = 'bmp';
          break;
        default:
          mimeType = 'jpeg';
      }
      
      // Convert to base64 with data URL format
      const base64String = image.getBase64(Jimp.JimpMime[mimeType]);
      
      return base64String;
    } catch (error) {
      // Fallback to direct base64 conversion without Jimp processing
      const fallbackMimeType = format === 'jpg' || format === 'jpeg' ? 'image/jpeg' : `image/${format}`;
      const fallbackBase64 = `data:${fallbackMimeType};base64,${buffer.toString('base64')}`;
      
      return fallbackBase64;
    }
  }


}