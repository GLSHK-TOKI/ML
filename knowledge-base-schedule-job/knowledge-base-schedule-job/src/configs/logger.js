import dayjs from 'dayjs';

export default {
  level: process.env.LOG_LEVEL || 'info',
  // DO NOT use pretty print on testing and production, as the log is not compatible with kibana
  transport: process.env.PINO_PRETTY === 'true' ? {
    target: 'pino-pretty',
    options: {
      colorize: true
    }
  } : false,
  timestamp: () => `,"time": "${dayjs().format('YYYY-MM-DD HH:mm:ss')}"`
};
