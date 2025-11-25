import logger from '../services/logger.js';
import { connector } from '../services/ai.js';

export const runSharePointConnector = async () => {
  const _log = logger.child({ method: 'runSharePointConnector' });
  _log.info('Executing runSharePointConnector function');
  
  const result = await connector.run();
  _log.info(result);
};