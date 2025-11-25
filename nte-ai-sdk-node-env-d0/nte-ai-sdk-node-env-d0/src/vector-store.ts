import { Client } from '@elastic/elasticsearch';

export interface VectorStoreOptions {
  /**
   * The host of the Elasticsearch instance.
   */
  host: string;

  /**
   * The basic auth credentials for the Elasticsearch instance.
   */
  basicAuth: {
    username: string;
    password: string;
  };
}

export class VectorStore {
  es: Client;

  constructor(options: VectorStoreOptions) {
    this.es = new Client({
      node: options.host,
      auth: {
        username: options.basicAuth.username,
        password: options.basicAuth.password,
      }
    })
  }
}