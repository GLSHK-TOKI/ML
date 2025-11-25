export abstract class SourceConnector {
  abstract run(): Promise<SourceConnectorResult>;
}

export interface SourceConnectorResult {
  status: string;
}