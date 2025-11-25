import pino from 'pino';
import dayjs from 'dayjs';

const logger = pino({
  level: process.env.NTE_AISDK_LOG_LEVEL || 'info',
  timestamp: () => `,"time": "${dayjs().format('YYYY-MM-DD HH:mm:ss')}"`
});


export default logger;