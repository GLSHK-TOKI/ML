import { Client } from '@elastic/elasticsearch';
import Mock from '@elastic/elasticsearch-mock';

export const mock = new Mock();
export const client = new Client({
  node: 'http://localhost:9200',
  Connection: mock.getConnection()
});