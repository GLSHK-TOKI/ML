import logger from '../services/logger.js';
import { isHealthy } from '../controllers/health.js';
import { runSharePointConnector } from '../controllers/connector.js';

export const runJobs = async () => {
  const _log = logger.child({ method: 'runJobs' });
  _log.info('Executing runJobs function');

  isHealthy();
  await runSharePointConnector();
};