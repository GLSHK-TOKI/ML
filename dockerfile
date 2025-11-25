# Multi-stage Dockerfile for Flask Knowledge Base Application

# Build stage for Node.js dependencies (if needed for build process)
FROM node:18-alpine AS node-builder
WORKDIR /app
COPY knowledge-base-schedule-job/knowledge-base-schedule-job/package*.json ./
RUN npm ci --only=production

# Python application stage
FROM python:3.11-slim AS python-app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY knowledge-base-flask/knowledge-base-flask/requirements.txt .
COPY knowledge-base-flask/knowledge-base-flask/pyproject.toml .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY knowledge-base-flask/knowledge-base-flask/ .

# Copy Node.js dependencies if needed
COPY --from=node-builder /app/node_modules ./node_modules

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application using Gunicorn
CMD ["gunicorn", "-c", "gunicorn_config.py", "app.main:app"]
