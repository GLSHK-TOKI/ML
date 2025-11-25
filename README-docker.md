# Knowledge Base Application - Docker Compose Setup

This Docker Compose configuration provides a complete environment for running the Knowledge Base application with all its dependencies.

## Services Included

- **flask-app**: Main Python Flask web application
- **elasticsearch**: Document storage and search engine
- **cron-job**: Node.js scheduled job processor
- **redis**: Caching and session storage (optional)
- **nginx**: Reverse proxy and load balancer (optional)

## Quick Start

1. **Copy environment variables**:
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file** with your actual configuration values:
   - Azure OpenAI API credentials
   - Azure SharePoint configuration
   - Other service-specific settings

3. **Build and start all services**:
   ```bash
   docker-compose up --build
   ```

4. **Access the application**:
   - Flask App: http://localhost:5000
   - Elasticsearch: http://localhost:9200
   - Nginx (if enabled): http://localhost:80

## Development Commands

### Start services in background:
```bash
docker-compose up -d
```

### View logs:
```bash
docker-compose logs -f flask-app
docker-compose logs -f cron-job
```

### Stop all services:
```bash
docker-compose down
```

### Rebuild specific service:
```bash
docker-compose build flask-app
docker-compose up -d flask-app
```

### Scale services:
```bash
docker-compose up --scale flask-app=3
```

## Service Configuration

### Flask Application
- Port: 5000
- Health check: `/health`
- Environment: Production
- Dependencies: Elasticsearch

### Elasticsearch
- Port: 9200, 9300
- Single-node cluster
- Security disabled for development
- Persistent volume: `elasticsearch_data`

### Cron Job
- Runs Node.js scheduled tasks
- Depends on Flask app and Elasticsearch
- Configurable via environment variables

### Redis (Optional)
- Port: 6379
- Used for caching and session management
- Persistent volume: `redis_data`

### Nginx (Optional)
- Port: 80, 443
- Reverse proxy with rate limiting
- Security headers included
- SSL support ready

## Environment Variables

Key environment variables to configure in `.env`:

```bash
# Azure OpenAI
OPENAI_AZURE_ENDPOINT_1=https://your-resource.openai.azure.com/
OPENAI_API_KEY_1=your-api-key

# Azure SharePoint
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret

# Application
LOG_LEVEL=INFO
CX_POD_ENV=production
```

## Data Persistence

The following volumes are created for data persistence:
- `elasticsearch_data`: Elasticsearch indices and data
- `redis_data`: Redis cache and session data

## Networking

All services communicate through the `knowledge-base-network` bridge network.

## Health Checks

Services include health checks to ensure proper startup order and monitoring:
- Elasticsearch: Cluster health endpoint
- Flask app: Custom health endpoint
- Redis: Ping command

## Customization

To modify the setup:

1. **Remove optional services**: Comment out `redis` and `nginx` services if not needed
2. **Add databases**: Add PostgreSQL or MySQL if required
3. **Environment-specific configs**: Create `docker-compose.prod.yml` for production overrides
4. **Resource limits**: Add memory and CPU limits for production deployment

## Troubleshooting

### Common Issues:

1. **Port conflicts**: Change port mappings if ports are already in use
2. **Memory issues**: Increase Docker memory limits or reduce Elasticsearch heap size
3. **Permission errors**: Ensure Docker has permission to create volumes
4. **Network connectivity**: Check if services can reach each other using service names

### Debug commands:
```bash
# Check service status
docker-compose ps

# Inspect networks
docker network ls
docker network inspect ml_knowledge-base-network

# Check volumes
docker volume ls
docker volume inspect ml_elasticsearch_data
```