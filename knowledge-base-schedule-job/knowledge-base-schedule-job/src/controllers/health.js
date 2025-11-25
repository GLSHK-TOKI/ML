import logger from '../services/logger.js';

export const isHealthy = () => {
  const _log = logger.child({ method: 'isHealthy' });
  _log.info('Executing isHealthy function');
  _log.info('Health is OK!');
  return 'OK';
};