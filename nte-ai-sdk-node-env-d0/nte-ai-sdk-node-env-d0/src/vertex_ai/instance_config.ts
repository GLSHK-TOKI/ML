import { BaseInstanceConfig } from '../instance-config.js';

export class VertexAIInstanceConfig extends BaseInstanceConfig {
  location: string;
  project: string;
  credentials_base64: string;

  constructor(options: { location: string; project: string; credentials_base64: string }) {
    super();
    this.location = options.location;
    this.project = options.project;
    this.credentials_base64 = options.credentials_base64;
  }
}