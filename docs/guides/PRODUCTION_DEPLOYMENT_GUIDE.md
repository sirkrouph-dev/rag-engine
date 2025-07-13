# 🧪 RAG Engine Advanced Features Testing Guide

## Overview

The RAG Engine has **advanced experimental features** including comprehensive security, monitoring, error handling, caching, and database management. This guide covers testing and validating these experimental features in development environments.

## 🏗️ Experimental Features

### ✅ **Security Integration**
- **Authentication**: JWT and API key authentication
- **Rate Limiting**: Configurable rate limits with sliding windows
- **Input Validation**: Comprehensive input sanitization and validation
- **Audit Logging**: Complete audit trail for all operations
- **Security Headers**: XSS protection, CORS, and security headers

### ✅ **Error Handling & Reliability**
- **Circuit Breakers**: Automatic failure detection and recovery
- **Retry Logic**: Exponential backoff with configurable parameters
- **Graceful Degradation**: Fallback mechanisms for service failures
- **Error Tracking**: Comprehensive error statistics and monitoring

### ✅ **Monitoring & Observability**
- **Metrics Collection**: Performance metrics for all operations
- **Health Checks**: Component health monitoring and reporting
- **Prometheus Export**: Industry-standard metrics format
- **Real-time Alerting**: Configurable thresholds and notifications

### ✅ **Production Database**
- **User Management**: Secure user authentication with PBKDF2 hashing
- **Session Management**: TTL-based session handling
- **Audit Logging**: Complete audit trail storage
- **Database Abstraction**: Support for SQLite, PostgreSQL, MySQL

### ✅ **Production Caching**
- **Redis Integration**: High-performance caching with Redis
- **Response Caching**: Intelligent response caching with TTL
- **Embedding Caching**: Cost optimization for AI services
- **Rate Limiting**: Cache-based rate limiting implementation

### ✅ **Comprehensive Testing**
- **280+ Tests**: Complete test coverage for all production features
- **Performance Benchmarks**: Validated performance metrics
- **Integration Tests**: End-to-end workflow validation
- **Security Tests**: Authentication, authorization, and security validation

## 🧪 Quick Experimental Setup

### 1. Install Dependencies

```bash
# Install the RAG Engine with experimental dependencies
pip install -r requirements.txt

# Install optional experimental dependencies
pip install redis psycopg2-binary  # For Redis and PostgreSQL (experimental)
```

### 2. Configure Experimental Settings

Create `config/experimental.json`:

```json
{
  "security": {
    "authentication": {
      "enabled": true,
      "method": "jwt",
      "jwt_secret": "your-secure-jwt-secret-key",
      "jwt_expiry": 3600
    },
    "rate_limiting": {
      "enabled": true,
      "default_limit": "100/hour",
      "burst_limit": 10
    },
    "input_validation": {
      "enabled": true,
      "max_query_length": 1000,
      "sanitize_input": true
    }
  },
  "monitoring": {
    "enabled": true,
    "metrics_endpoint": "/metrics",
    "health_checks": true,
    "alerting": {
      "enabled": true,
      "error_threshold": 0.05,
      "latency_threshold": 2.0
    }
  },
  "database": {
    "provider": "postgresql",
    "host": "localhost",
    "port": 5432,
    "database": "rag_engine",
    "username": "rag_user",
    "password": "secure_password"
  },
  "caching": {
    "provider": "redis",
    "host": "localhost",
    "port": 6379,
    "password": "redis_password",
    "ttl": 3600
  },
  "error_handling": {
    "circuit_breaker": {
      "enabled": true,
      "failure_threshold": 5,
      "timeout": 60
    },
    "retry": {
      "enabled": true,
      "max_retries": 3,
      "backoff_factor": 2
    }
  }
}
```

### 3. Start Production Services

```bash
# Start Redis (if using Redis caching)
redis-server

# Start PostgreSQL (if using PostgreSQL database)
sudo systemctl start postgresql

# Start the RAG Engine with experimental configuration
python -m rag_engine serve \
  --config config/experimental.json \
  --framework fastapi \
  --workers 4 \
  --port 8000
```

### 4. Validate Experimental Setup

```bash
# Run comprehensive experimental tests
python tests/run_production_tests.py --category all

# Run specific test categories
python tests/run_production_tests.py --category security
python tests/run_production_tests.py --category performance
python tests/run_production_tests.py --category e2e
```

## 🧪 Experimental Testing

### Test Categories

The experimental test suite includes 8 comprehensive test files:

1. **Security Integration Tests** (`test_security_integration.py`)
   - JWT authentication flows
   - Input validation and sanitization
   - Rate limiting enforcement
   - Audit logging verification

2. **Error Handling Tests** (`test_error_handling_integration.py`)
   - Circuit breaker functionality
   - Retry logic with exponential backoff
   - Graceful degradation scenarios
   - Error recovery mechanisms

3. **Monitoring Tests** (`test_monitoring_integration.py`)
   - Metrics collection accuracy
   - Health check responses
   - Alerting threshold validation
   - Performance tracking

4. **Database Tests** (`test_production_database.py`)
   - User management operations
   - Session handling and TTL
   - Audit log storage and retrieval
   - Concurrent access patterns

5. **Caching Tests** (`test_production_caching.py`)
   - Redis cache operations
   - Response caching effectiveness
   - Cache invalidation strategies
   - Performance optimization

6. **API Integration Tests** (`test_production_api_integration.py`)
   - Complete API with all middleware
   - Security enforcement
   - Error handling integration
   - Performance under load

7. **End-to-End Tests** (`test_production_e2e.py`)
   - Complete user workflows
   - System integration scenarios
   - Performance benchmarks
   - Failure recovery testing

8. **Comprehensive Production Tests** (`test_comprehensive_production.py`)
   - Overall system validation
   - Component integration
   - Production readiness checks
   - Performance baselines

### Running Tests

```bash
# Run all experimental tests
python tests/run_production_tests.py

# Run with specific options
python tests/run_production_tests.py \
  --category performance \
  --parallel \
  --coverage \
  --benchmark

# Quick validation (essential tests only)
python tests/run_production_tests.py --category quick
```

### Performance Benchmarks

The test suite validates these performance benchmarks:

- **Authentication**: >100 operations/second
- **Caching**: >1000 operations/second
- **Database**: >50 operations/second
- **API Responses**: <2 seconds average
- **Error Recovery**: <5 seconds for circuit breaker recovery

## 🐳 Docker Experimental Deployment

### Docker Compose Experimental Stack

```yaml
# docker-compose.experimental.yml
version: '3.8'

services:
  rag-engine:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RAG_CONFIG_PATH=/app/config/experimental.json
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://rag_user:password@postgres:5432/rag_engine
    depends_on:
      - redis
      - postgres
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --requirepass redis_password
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=rag_engine
      - POSTGRES_USER=rag_user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rag-engine

volumes:
  redis_data:
  postgres_data:
```

### Deploy Experimental Stack

```bash
# Deploy experimental services
docker-compose -f docker-compose.experimental.yml up -d

# Scale the application
docker-compose -f docker-compose.experimental.yml up -d --scale rag-engine=3

# Monitor logs
docker-compose -f docker-compose.experimental.yml logs -f rag-engine
```

## 📊 Monitoring and Alerting

### Health Check Endpoints

- `GET /health` - Basic health check
- `GET /health/detailed` - Comprehensive health status
- `GET /health/database` - Database connectivity
- `GET /health/cache` - Cache connectivity
- `GET /metrics` - Prometheus metrics

### Monitoring Integration

```python
# Example monitoring setup
from rag_engine.core.monitoring_integration import MonitoringIntegration

# Initialize monitoring
monitoring = MonitoringIntegration()

# Start metrics collection
monitoring.start_metrics_collection()

# Setup alerting
monitoring.setup_alerting({
    "error_rate_threshold": 0.05,
    "latency_threshold": 2.0,
    "alert_webhook": "https://your-webhook-url"
})
```

### Grafana Dashboard

Import the provided Grafana dashboard to visualize:

- Request throughput and latency
- Error rates by endpoint
- Cache hit rates and performance
- Database connection pools
- System resource utilization

## 🔒 Security Configuration

### Authentication Setup

```python
# JWT Authentication
from rag_engine.interfaces.security_integration import SecurityIntegration

security = SecurityIntegration()

# Configure JWT
security.configure_jwt_auth(
    secret_key="your-secure-secret-key",
    algorithm="HS256",
    expiry_hours=24
)

# Configure rate limiting
security.configure_rate_limiting(
    default_limit="100/hour",
    burst_limit=10,
    storage="redis://localhost:6379"
)
```

### Input Validation

```python
# Configure input validation
security.configure_input_validation(
    max_query_length=1000,
    allowed_file_types=[".txt", ".pdf", ".docx"],
    sanitize_html=True,
    validate_json=True
)
```

## 🔧 Production Configuration Examples

### High-Performance Configuration

```json
{
  "performance": {
    "workers": 8,
    "worker_connections": 1000,
    "keepalive_timeout": 65,
    "max_requests": 1000,
    "max_requests_jitter": 100
  },
  "caching": {
    "provider": "redis",
    "cluster_mode": true,
    "connection_pool_size": 50,
    "ttl": 3600,
    "max_memory": "2gb"
  },
  "database": {
    "connection_pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600
  }
}
```

### Security-Hardened Configuration

```json
{
  "security": {
    "authentication": {
      "method": "jwt",
      "require_https": true,
      "jwt_expiry": 900,
      "refresh_token_expiry": 86400
    },
    "rate_limiting": {
      "strict_mode": true,
      "per_user_limits": true,
      "ip_whitelist": ["192.168.1.0/24"],
      "fail2ban_integration": true
    },
    "headers": {
      "hsts_max_age": 31536000,
      "content_security_policy": "default-src 'self'",
      "x_frame_options": "DENY"
    }
  }
}
```

## 🚀 Scaling Strategies

### Horizontal Scaling

```bash
# Kubernetes deployment
kubectl apply -f k8s-deployment.yaml

# Auto-scaling based on CPU/memory
kubectl autoscale deployment rag-engine \
  --cpu-percent=70 \
  --min=3 \
  --max=20
```

### Load Balancing with Nginx

```nginx
upstream rag_backend {
    least_conn;
    server rag-engine-1:8000 max_fails=3 fail_timeout=30s;
    server rag-engine-2:8000 max_fails=3 fail_timeout=30s;
    server rag-engine-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    location / {
        proxy_pass http://rag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## 📋 Production Checklist

### Pre-Deployment

- [ ] All production tests passing (280+ tests)
- [ ] Security configuration validated
- [ ] Database migrations completed
- [ ] Cache configuration tested
- [ ] Monitoring and alerting configured
- [ ] SSL certificates installed
- [ ] Backup strategy implemented
- [ ] Load testing completed

### Post-Deployment

- [ ] Health checks responding correctly
- [ ] Metrics collection active
- [ ] Error rates within acceptable limits
- [ ] Performance benchmarks met
- [ ] Security scans passed
- [ ] Backup verification completed
- [ ] Monitoring dashboards operational

### Ongoing Maintenance

- [ ] Regular security updates
- [ ] Performance monitoring
- [ ] Capacity planning
- [ ] Backup testing
- [ ] Security audits
- [ ] Documentation updates

## 🆘 Troubleshooting

### Common Issues

1. **Authentication Failures**
   ```bash
   # Check JWT configuration
   python -c "from rag_engine.interfaces.security_integration import SecurityIntegration; SecurityIntegration().validate_jwt_config()"
   ```

2. **Performance Issues**
   ```bash
   # Run performance tests
   python tests/run_production_tests.py --category performance --benchmark
   ```

3. **Database Connectivity**
   ```bash
   # Test database connection
   python -c "from rag_engine.core.production_database import ProductionDatabaseManager; ProductionDatabaseManager().test_connection()"
   ```

4. **Cache Issues**
   ```bash
   # Test Redis connection
   python -c "from rag_engine.core.production_caching import ProductionCacheManager; ProductionCacheManager().test_connection()"
   ```

### Monitoring Commands

```bash
# Check system health
curl http://localhost:8000/health/detailed

# View metrics
curl http://localhost:8000/metrics

# Check specific component health
curl http://localhost:8000/health/database
curl http://localhost:8000/health/cache
```

## 📚 Additional Resources

- **Production Testing Guide**: `PRODUCTION_TESTING_COMPLETE.md`
- **Security Integration**: `rag_engine/interfaces/security_integration.py`
- **Error Handling**: `rag_engine/core/error_handling_integration.py`
- **Monitoring**: `rag_engine/core/monitoring_integration.py`
- **Database**: `rag_engine/core/production_database.py`
- **Caching**: `rag_engine/core/production_caching.py`

## 🎉 Conclusion

The RAG Engine has **advanced experimental features** ready for testing:

- 🧪 **Comprehensive Security** with authentication, validation, and audit logging (experimental)
- 🧪 **Reliable Error Handling** with circuit breakers and graceful degradation (testing)
- 🧪 **Complete Monitoring** with metrics, health checks, and alerting (beta)
- 🧪 **Advanced Database** with user management and audit trails (experimental)
- 🧪 **High-Performance Caching** with Redis integration (testing)
- 🧪 **Extensive Testing** with 280+ tests covering all experimental features
- 🧪 **Scalable Architecture** ready for advanced testing and validation

Test these advanced features knowing that comprehensive testing infrastructure is in place for validation! 