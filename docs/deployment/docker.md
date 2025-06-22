# Docker Deployment

The RAG Engine provides Docker support for easy containerized deployment with production-ready configurations.

## Docker Images

### Production Image
Multi-stage build optimized for production deployment.

```dockerfile
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000

CMD ["python", "-m", "rag_engine", "serve", "--config", "config/production.json", "--framework", "fastapi", "--host", "0.0.0.0", "--port", "8000"]
```

### Development Image
Simple single-stage build for development.

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "rag_engine", "serve", "--config", "config.json", "--framework", "fastapi", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

## Docker Compose

### Basic Setup
```yaml
version: '3.8'

services:
  rag-engine:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RAG_CONFIG_PATH=/app/config/production.json
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    restart: unless-stopped
```

### With Vector Database
```yaml
version: '3.8'

services:
  rag-engine:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RAG_CONFIG_PATH=/app/config/production.json
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHROMA_HOST=chroma
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    depends_on:
      - chroma
    restart: unless-stopped

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    restart: unless-stopped

volumes:
  chroma_data:
```

### Production Setup with Load Balancer
```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rag-engine-1
      - rag-engine-2
    restart: unless-stopped

  rag-engine-1:
    build: .
    environment:
      - RAG_CONFIG_PATH=/app/config/production.json
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    restart: unless-stopped

  rag-engine-2:
    build: .
    environment:
      - RAG_CONFIG_PATH=/app/config/production.json
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    restart: unless-stopped

  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

## Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key
RAG_CONFIG_PATH=/app/config/production.json

# Optional
ANTHROPIC_API_KEY=your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Database
CHROMA_HOST=localhost
CHROMA_PORT=8001

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=info
```

### Docker Environment File
```bash
# .env file
OPENAI_API_KEY=sk-...
RAG_CONFIG_PATH=/app/config/production.json
LOG_LEVEL=info
WORKERS=4
```

### Production Configuration
```json
{
  "api": {
    "framework": "fastapi",
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "log_level": "info"
  },
  "vectorstore": {
    "type": "chroma",
    "config": {
      "host": "${CHROMA_HOST:-localhost}",
      "port": "${CHROMA_PORT:-8001}",
      "collection_name": "production_docs"
    }
  },
  "llm": {
    "type": "openai",
    "config": {
      "api_key": "${OPENAI_API_KEY}",
      "model": "gpt-3.5-turbo"
    }
  }
}
```

## Building and Running

### Build Image
```bash
# Build production image
docker build -t rag-engine:latest .

# Build with specific tag
docker build -t rag-engine:v1.0.0 .

# Build development image
docker build -f Dockerfile.dev -t rag-engine:dev .
```

### Run Container
```bash
# Basic run
docker run -p 8000:8000 rag-engine:latest

# With environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e RAG_CONFIG_PATH=/app/config/production.json \
  rag-engine:latest

# With volume mounts
docker run -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY=your_key \
  rag-engine:latest

# Interactive development
docker run -it -p 8000:8000 \
  -v $(pwd):/app \
  rag-engine:dev bash
```

### Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f rag-engine

# Scale services
docker-compose up -d --scale rag-engine=3

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

## Health Checks

### Docker Health Check
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

### Compose Health Check
```yaml
services:
  rag-engine:
    build: .
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## Monitoring

### Logging Configuration
```yaml
services:
  rag-engine:
    build: .
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Metrics Collection
```yaml
services:
  rag-engine:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"  # Metrics endpoint

  prometheus:
    image: prom/prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
```

## Security

### Non-root User
```dockerfile
FROM python:3.10-slim

RUN groupadd -r raguser && useradd -r -g raguser raguser

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chown -R raguser:raguser /app

USER raguser
EXPOSE 8000

CMD ["python", "-m", "rag_engine", "serve"]
```

### Secrets Management
```yaml
services:
  rag-engine:
    build: .
    secrets:
      - openai_api_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key

secrets:
  openai_api_key:
    file: ./secrets/openai_api_key.txt
```

### Network Security
```yaml
version: '3.8'

services:
  rag-engine:
    build: .
    networks:
      - internal
    ports:
      - "8000:8000"

  chroma:
    image: chromadb/chroma:latest
    networks:
      - internal
    # No external ports exposed

networks:
  internal:
    driver: bridge
    internal: true
```

## Performance Optimization

### Multi-stage Build
```dockerfile
# Build stage
FROM python:3.10-slim as builder
WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Runtime stage
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
COPY . .
```

### Resource Limits
```yaml
services:
  rag-engine:
    build: .
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Caching
```dockerfile
# Cache pip packages
FROM python:3.10-slim
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .
```

## Troubleshooting

### Common Issues

**1. Permission Denied**
```bash
# Fix file permissions
docker run --rm -v $(pwd):/app alpine chown -R 1000:1000 /app
```

**2. Port Already in Use**
```bash
# Find process using port
docker ps | grep 8000

# Stop conflicting container
docker stop <container_id>
```

**3. Out of Memory**
```yaml
# Increase memory limits
services:
  rag-engine:
    deploy:
      resources:
        limits:
          memory: 8G
```

**4. Slow Startup**
```yaml
# Adjust health check timing
healthcheck:
  start_period: 60s  # Increase startup time
  interval: 60s      # Reduce check frequency
```

### Debugging

```bash
# View container logs
docker logs rag-engine

# Execute commands in container
docker exec -it rag-engine bash

# Inspect container
docker inspect rag-engine

# View resource usage
docker stats rag-engine
```

## Production Considerations

### Data Persistence
```yaml
services:
  rag-engine:
    volumes:
      - app_data:/app/data
      - ./config:/app/config:ro  # Read-only config

volumes:
  app_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /persistent/storage/path
```

### Backup Strategy
```bash
# Backup volumes
docker run --rm -v app_data:/data -v $(pwd):/backup alpine tar czf /backup/app_data.tar.gz /data

# Restore volumes
docker run --rm -v app_data:/data -v $(pwd):/backup alpine tar xzf /backup/app_data.tar.gz -C /
```

### Update Strategy
```bash
# Rolling update
docker-compose pull
docker-compose up -d --no-deps rag-engine

# Blue-green deployment
docker-compose -f docker-compose.blue.yml up -d
# Test new version
docker-compose -f docker-compose.green.yml down
```
