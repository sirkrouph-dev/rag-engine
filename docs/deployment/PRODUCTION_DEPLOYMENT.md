# 🚀 RAG Engine Production Deployment Guide

## Overview

This guide covers deploying the RAG Engine in a production environment with full monitoring, security, and scalability features.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 8GB RAM
- 50GB+ disk space
- Valid SSL certificates (for HTTPS)
- External AI service API keys

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd rag_engine

# Copy environment template
cp docker/.env.example docker/.env

# Edit environment variables
nano docker/.env
```

### 2. Configure Environment Variables

Edit `docker/.env` with your actual values:

```bash
# Required - Database
DB_PASSWORD=your_secure_password_here
REDIS_PASSWORD=your_redis_password_here

# Required - Security
JWT_SECRET_KEY=your_jwt_secret_32_chars_minimum
ENCRYPTION_KEY=your_encryption_key_32_chars

# Required - AI Services
OPENAI_API_KEY=sk-your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX=your_index_name

# Optional - Monitoring
GRAFANA_PASSWORD=your_grafana_admin_password
```

### 3. SSL Certificates (Recommended)

Place your SSL certificates in `docker/nginx/ssl/`:

```bash
mkdir -p docker/nginx/ssl
cp your_cert.pem docker/nginx/ssl/cert.pem
cp your_key.pem docker/nginx/ssl/key.pem
```

### 4. Deploy Production Stack

```bash
# Full production deployment
cd docker
docker-compose -f docker-compose.production.yml up -d

# Or deploy with specific profiles
docker-compose -f docker-compose.production.yml --profile monitoring up -d
```

## Architecture Overview

### Core Services

1. **RAG Engine** - Main application server
   - FastAPI with production middleware
   - Security, monitoring, and error handling
   - Health checks and graceful shutdown

2. **PostgreSQL** - Primary database
   - User data, conversation history
   - Audit logs and system state

3. **Redis** - Caching and session store
   - Response caching
   - Rate limiting counters
   - Session management

### Monitoring Stack

1. **Prometheus** - Metrics collection
   - Application metrics
   - System performance
   - Custom business metrics

2. **Grafana** - Visualization dashboard
   - Real-time monitoring
   - Performance analytics
   - Alert visualization

### Logging Stack

1. **Elasticsearch** - Log storage and search
2. **Logstash** - Log processing pipeline
3. **Kibana** - Log analysis and visualization

### Load Balancing

1. **Nginx** - Reverse proxy and load balancer
   - SSL termination
   - Static file serving
   - Request routing

## Configuration

### Database Initialization

The production database includes:
- Optimized connection pooling
- SSL connections
- Backup configuration
- Monitoring endpoints

### Security Features

- JWT-based authentication
- Rate limiting per endpoint
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration
- Audit logging

### Performance Optimizations

- Redis caching for frequent queries
- Database connection pooling
- Async request handling
- Response compression
- Static file caching

## Monitoring and Alerting

### Key Metrics

1. **Application Metrics**
   - Request rate and latency
   - Error rates by endpoint
   - AI service response times
   - Cache hit rates

2. **System Metrics**
   - CPU and memory usage
   - Disk I/O and space
   - Network traffic
   - Database performance

3. **Business Metrics**
   - Active users
   - Query success rates
   - Feature usage statistics
   - Cost tracking

### Health Checks

- `/health/live` - Service liveness
- `/health/ready` - Service readiness
- `/health/db` - Database connectivity
- `/health/redis` - Cache connectivity
- `/metrics` - Prometheus metrics

### Accessing Monitoring

- **Grafana Dashboard**: http://localhost:3000 (admin/your_password)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601

## Scaling

### Horizontal Scaling

```bash
# Scale the main application
docker-compose -f docker-compose.production.yml up -d --scale rag-engine=3

# Add database replicas
docker-compose -f docker-compose.production.yml -f docker-compose.scale.yml up -d
```

### Vertical Scaling

Edit resource limits in `docker-compose.production.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
docker exec rag-postgres-production pg_dump -U rag_user rag_engine > backup.sql

# Restore from backup
docker exec -i rag-postgres-production psql -U rag_user rag_engine < backup.sql
```

### Redis Backup

```bash
# Redis automatically creates snapshots
# Copy the dump.rdb file from the container
docker cp rag-redis-production:/data/dump.rdb ./redis-backup.rdb
```

## Security Considerations

### Environment Variables

- Never commit `.env` files
- Use secrets management in production
- Rotate keys regularly
- Use strong, unique passwords

### Network Security

- Configure firewall rules
- Use VPC/private networks
- Enable SSL/TLS everywhere
- Implement network segmentation

### Access Control

- Principle of least privilege
- Regular access reviews
- Multi-factor authentication
- API key rotation

## Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   # Check logs
   docker-compose -f docker-compose.production.yml logs rag-engine
   
   # Check dependencies
   docker-compose -f docker-compose.production.yml ps
   ```

2. **Database connection errors**
   ```bash
   # Verify database is healthy
   docker exec rag-postgres-production pg_isready -U rag_user
   
   # Check environment variables
   docker exec rag-engine-production env | grep DB_
   ```

3. **High memory usage**
   ```bash
   # Monitor resource usage
   docker stats
   
   # Adjust memory limits
   # Edit docker-compose.production.yml
   ```

### Log Locations

- Application logs: `/app/logs/` (mounted volume)
- Nginx logs: `/var/log/nginx/` (mounted volume)
- Database logs: Container stdout/stderr
- Redis logs: Container stdout/stderr

### Performance Tuning

1. **Database Optimization**
   - Adjust `shared_buffers` and `effective_cache_size`
   - Monitor slow queries
   - Create appropriate indexes

2. **Redis Optimization**
   - Configure `maxmemory` policies
   - Monitor cache hit rates
   - Adjust TTL values

3. **Application Tuning**
   - Adjust worker processes
   - Configure connection pools
   - Monitor API response times

## Production Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] Security hardening applied
- [ ] Load testing completed

### Post-Deployment

- [ ] Health checks passing
- [ ] Monitoring dashboards operational
- [ ] Log aggregation working
- [ ] Backup verification
- [ ] Security scan passed
- [ ] Performance baseline established

### Ongoing Maintenance

- [ ] Regular security updates
- [ ] Database maintenance
- [ ] Log rotation configured
- [ ] Capacity monitoring
- [ ] Cost optimization
- [ ] Documentation updates

## Support

For production support:
1. Check the monitoring dashboards first
2. Review application logs
3. Consult this deployment guide
4. Check the troubleshooting section
5. Contact support with specific error details

## Updates and Upgrades

### Rolling Updates

```bash
# Update with zero downtime
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d --no-deps rag-engine
```

### Database Migrations

```bash
# Run migrations
docker exec rag-engine-production python -m rag_engine migrate
```

This production deployment provides enterprise-grade reliability, security, and monitoring for the RAG Engine system.
