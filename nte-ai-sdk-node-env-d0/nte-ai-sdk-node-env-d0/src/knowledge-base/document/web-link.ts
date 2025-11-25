import { DriveItem } from "@microsoft/msgraph-sdk/models/index.js";
import { load } from 'cheerio';
import chromium from '@sparticuz/chromium';
import _puppeteerExtra from 'puppeteer-extra';
const puppeteerExtra = _puppeteerExtra.default; // Workaround for ESM import support
import stealthPlugin from 'puppeteer-extra-plugin-stealth';
import axios from "axios";
import logger from "../../logger/logger.js";

const _log = logger.child({ module: 'ai-sdk-node.knowledge-base.document.web-link' });

import { KnowledgeBaseDocument } from './document.js';
import { PuppeteerError } from "../../exception/index.js";
import { Page } from "puppeteer";

export class KnowledgeBaseWebLink extends KnowledgeBaseDocument {
  static extension = 'url';
  private metadata: { title: string, webUrl: string } | null = null; // Cache metadata to avoid multiple scrape

  constructor(doc: DriveItem) {
    super(doc);
  }

  async getMetadata() {
    if (!this.metadata) {
      this.metadata = await this.cacheMetadata();
    }

    return {
      title: this.metadata.title,
      webUrl: this.metadata.webUrl,
    };
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

  async getContent() {
    const url = await this.getUrlFromFile();
    const response = await this.scrape(url);
    return response || '';
  }

  private async cacheMetadata() {
    return {
      title: await this.getTitleFromPage() || this.doc.webUrl?.substring(this.doc.webUrl.lastIndexOf('/') + 1) || '',
      webUrl: await this.getUrlFromFile() || this.doc.webUrl || '',
    }
  }

  private async getUrlFromFile() {
    const docWithDownloadUrl = this.doc as DriveItem & { '@microsoft.graph.downloadUrl': string };
    const fileUrl = docWithDownloadUrl['@microsoft.graph.downloadUrl'];

    try {
      const response = await axios.get(fileUrl, { responseType: 'arraybuffer' });
      const urlBuffer = response.data;
      const urlFile = urlBuffer.toString('utf8');

      return urlFile.match(/(https?:\/\/[^\s]+)/g)[0];
    } catch (error) {
      const msg = `Failed to download and extract url from file: ${fileUrl} ${error}`;
      _log.error({msg: msg, status_code: 400})
    }
  }

  private async scrape(url: string) {
    const content = await this.fetchWebpageContent(url);
    const mainContent = await this.getMainContentFromHTML(content);
    return mainContent;
  }

  private async fetchWebpageContent(url: string) {
    // Launch the browser and open a new blank page
    const { browser, page } = await this.getBrowserAndPage();

    try {
      // Navigate the page to a URL and get the content
      await page.setViewport({ width: 1080, height: 1024 });
      await this.navigateToURL(page, url);
      return await page.content();
    } catch (error) {
      const msg = `Error fetching webpage content: ${error}`;
      _log.error({msg: msg, status_code: 400})
      throw new PuppeteerError(500, msg);
    } finally {
      // Close the browser
      await browser.close();
    }
  };

  private getMainContentFromHTML = async (content: string) => {
    const $ = load(content);
    $('script, style, head, nav, footer, iframe, img, noscript').remove();
    return $('body').text().replace(/\s+/g, ' ').replace(/[^\x20-\x7E]+/g, '').trim();
  };

  private async getTitleFromPage() {
    // Launch the browser and open a new blank page
    const { browser, page } = await this.getBrowserAndPage();
    const url = await this.getUrlFromFile();

    try {
      // Navigate the page to a URL and get the title
      await page.setViewport({ width: 1080, height: 1024 });
      await this.navigateToURL(page, url);
      return await page.title();
    } catch (error) {
      const msg = `Error getting title from page: ${error}, falling back to URL as title`;
      _log.error({msg: msg, status_code: 400})
      return null;
    } finally {
      // Close the browser
      await browser.close();
    }
  }

  private async getBrowserAndPage() {
    const isOnOpenshift = process.env.CX_POD_ENV;
    try {
      if (isOnOpenshift) {
        return await this.getBrowserAndPageOnOpenshift();
      } else {
        return await this.getBrowserAndPageOnLocal();
      }
    } catch (error) {
      const msg = `Error launching browser: ${error}`;
      logger.error({ msg: msg, status_code: 500 })
      throw new PuppeteerError(500, msg);
    };
  };

  private async navigateToURL(page: Page, url: string) {
    try {
      await page.goto(url, { waitUntil: 'networkidle0'});
    } catch (error) {
      // If the page fails to load, continue to get the partial content from the page
      const msg = `Error fetching when navigating to webpage [${url}], trying to retrieve partial content: ${error}`;
      logger.error({ msg: msg, status_code: 400 })
    }
  }

  private async getBrowserAndPageOnOpenshift() {
    puppeteerExtra.use(stealthPlugin());
    const browser = await puppeteerExtra.launch({
      args: chromium.args,
      defaultViewport: chromium.defaultViewport,
      executablePath: await chromium.executablePath(),
      headless: chromium.headless === 'true' ? true : 
                chromium.headless === 'shell' ? 'shell' : true,
      acceptInsecureCerts: true,
    });
    const page = await browser.newPage();
    return { browser, page };
  }

  private async getBrowserAndPageOnLocal() {
    const puppeteer = await import('puppeteer');
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    const ua =
      'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Mobile Safari/537.36';
    await page.setUserAgent(ua);
    return { browser, page };
  }
}