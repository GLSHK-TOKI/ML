import 'dotenv/config';
import logger from './src/services/logger.js';
import { runJobs } from './src/jobs/index.js';

(async() => {
  try {
    logger.info('Job execution started');
    await runJobs();
    logger.info('Execution completed');
  } catch (e) {
    logger.error('Execution failed');
    logger.error(e);
  }
})();