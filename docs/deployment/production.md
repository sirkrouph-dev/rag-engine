# Production Deployment

This guide covers deploying the RAG Engine in production environments with focus on scalability, reliability, and security.

## Production Architecture

### Single Server Deployment
```
┌─────────────────┐
│   Load Balancer │
│    (nginx)      │
└─────────┬───────┘
          │
┌─────────▼───────┐
│   RAG Engine    │
│  (Multi-worker) │
└─────────┬───────┘
          │
┌─────────▼───────┐
│  Vector Store   │
│   (ChromaDB)    │
└─────────────────┘
```

### Multi-Server Deployment
```
              ┌─────────────────┐
              │   Load Balancer │
              │    (nginx)      │
              └─────────┬───────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼───────┐
│  RAG Engine   │ │RAG Engine │ │  RAG Engine   │
│   Server 1    │ │ Server 2  │ │   Server 3    │
└───────┬───────┘ └─────┬─────┘ └───────┬───────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
              ┌─────────▼─────────┐
              │   Vector Store    │
              │    Cluster        │
              └───────────────────┘
```

## Server Configuration

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB SSD
- Network: 100 Mbps

**Recommended Production:**
- CPU: 8 cores
- RAM: 16GB
- Storage: 100GB SSD
- Network: 1 Gbps

**High-Performance Setup:**
- CPU: 16+ cores
- RAM: 32GB+
- Storage: 500GB+ NVMe SSD
- Network: 10 Gbps

### Operating System Setup

#### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip nginx certbot python3-certbot-nginx

# Create application user
sudo useradd -m -s /bin/bash raguser
sudo usermod -aG sudo raguser

# Setup directories
sudo mkdir -p /opt/rag-engine
sudo chown raguser:raguser /opt/rag-engine
```

#### CentOS/RHEL
```bash
# Update system
sudo yum update -y

# Install dependencies
sudo yum install -y python3 python3-pip nginx certbot python3-certbot-nginx

# Create application user
sudo useradd -m raguser
sudo usermod -aG wheel raguser
```

### Application Setup

#### Installation
```bash
# Switch to application user
sudo su - raguser

# Clone repository
git clone https://github.com/your-org/rag-engine.git /opt/rag-engine
cd /opt/rag-engine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn

# Install as package
pip install -e .
```

#### Configuration
```bash
# Create production config
mkdir -p /opt/rag-engine/config
cp config/example_config.json config/production.json

# Set environment variables
cat >> ~/.bashrc << EOF
export RAG_CONFIG_PATH=/opt/rag-engine/config/production.json
export OPENAI_API_KEY=your_api_key_here
export PYTHONPATH=/opt/rag-engine:$PYTHONPATH
EOF

source ~/.bashrc
```

## Process Management

### Systemd Service

Create `/etc/systemd/system/rag-engine.service`:
```ini
[Unit]
Description=RAG Engine API Server
After=network.target

[Service]
Type=exec
User=raguser
Group=raguser
WorkingDirectory=/opt/rag-engine
Environment=PATH=/opt/rag-engine/venv/bin
Environment=RAG_CONFIG_PATH=/opt/rag-engine/config/production.json
Environment=OPENAI_API_KEY=your_api_key_here
ExecStart=/opt/rag-engine/venv/bin/gunicorn rag_engine.interfaces.api:create_production_app --bind 127.0.0.1:8000 --workers 4 --worker-class uvicorn.workers.UvicornWorker
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### Service Management
```bash
# Enable and start service
sudo systemctl enable rag-engine
sudo systemctl start rag-engine

# Check status
sudo systemctl status rag-engine

# View logs
sudo journalctl -u rag-engine -f

# Restart service
sudo systemctl restart rag-engine
```

### Supervisor (Alternative)

Install and configure Supervisor:
```bash
sudo apt install supervisor

# Create configuration
sudo tee /etc/supervisor/conf.d/rag-engine.conf << EOF
[program:rag-engine]
command=/opt/rag-engine/venv/bin/gunicorn rag_engine.interfaces.api:create_production_app --bind 127.0.0.1:8000 --workers 4 --worker-class uvicorn.workers.UvicornWorker
directory=/opt/rag-engine
user=raguser
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/rag-engine.log
environment=RAG_CONFIG_PATH="/opt/rag-engine/config/production.json",OPENAI_API_KEY="your_api_key_here"
EOF

# Start service
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start rag-engine
```

## Load Balancer Configuration

### Nginx Setup

Create `/etc/nginx/sites-available/rag-engine`:
```nginx
upstream rag_backend {
    # Multiple workers for load balancing
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s backup;
}

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    # API endpoints
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://rag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://rag_backend;
        proxy_set_header Host $host;
        access_log off;
    }
    
    # Static files (if any)
    location /static/ {
        alias /opt/rag-engine/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### Enable Site
```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/rag-engine /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
```

### SSL Certificate
```bash
# Install SSL certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

## Database Setup

### ChromaDB Production

#### Standalone Server
```bash
# Install ChromaDB server
pip install chromadb[server]

# Create systemd service for ChromaDB
sudo tee /etc/systemd/system/chroma.service << EOF
[Unit]
Description=ChromaDB Server
After=network.target

[Service]
Type=exec
User=raguser
Group=raguser
WorkingDirectory=/opt/rag-engine
Environment=PATH=/opt/rag-engine/venv/bin
ExecStart=/opt/rag-engine/venv/bin/chroma run --host 127.0.0.1 --port 8001 --path /opt/rag-engine/data/chroma
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable chroma
sudo systemctl start chroma
```

#### Docker Deployment
```bash
# Run ChromaDB in Docker
docker run -d \
  --name chroma \
  -p 8001:8000 \
  -v /opt/rag-engine/data/chroma:/chroma/chroma \
  chromadb/chroma:latest
```

### PostgreSQL (for metadata)
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE rag_engine;
CREATE USER raguser WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE rag_engine TO raguser;
EOF
```

## Monitoring and Logging

### Application Monitoring

#### Prometheus Metrics
```python
# Add to production config
{
  "monitoring": {
    "enabled": true,
    "metrics_port": 9090,
    "prometheus_endpoint": "/metrics"
  }
}
```

#### Health Checks
```bash
# Create health check script
cat > /opt/rag-engine/health_check.sh << 'EOF'
#!/bin/bash
curl -f http://localhost:8000/health || exit 1
EOF

chmod +x /opt/rag-engine/health_check.sh

# Add to crontab for monitoring
echo "*/5 * * * * /opt/rag-engine/health_check.sh > /dev/null 2>&1" | crontab -
```

### Log Management

#### Centralized Logging
```bash
# Install rsyslog
sudo apt install rsyslog

# Configure log rotation
sudo tee /etc/logrotate.d/rag-engine << EOF
/var/log/rag-engine/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 raguser raguser
    postrotate
        systemctl reload rag-engine
    endscript
}
EOF
```

#### Application Logging Configuration
```json
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": {
      "file": {
        "filename": "/var/log/rag-engine/app.log",
        "max_bytes": 10485760,
        "backup_count": 5
      },
      "syslog": {
        "address": "/dev/log",
        "facility": "local0"
      }
    }
  }
}
```

## Security Hardening

### Firewall Configuration
```bash
# Configure UFW
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow from 127.0.0.1 to any port 8000  # Internal API
```

### API Security

#### API Key Authentication
```python
# Production configuration
{
  "security": {
    "api_key_required": true,
    "rate_limiting": {
      "requests_per_minute": 60,
      "burst_size": 20
    },
    "cors": {
      "allowed_origins": ["https://yourdomain.com"],
      "allowed_methods": ["GET", "POST"],
      "allow_credentials": false
    }
  }
}
```

#### Environment Variables
```bash
# Secure environment file
sudo tee /opt/rag-engine/.env << EOF
RAG_CONFIG_PATH=/opt/rag-engine/config/production.json
OPENAI_API_KEY=your_secure_api_key
DATABASE_URL=postgresql://raguser:secure_password@localhost/rag_engine
SECRET_KEY=your_very_secure_secret_key
EOF

# Secure permissions
sudo chmod 600 /opt/rag-engine/.env
sudo chown raguser:raguser /opt/rag-engine/.env
```

## Backup and Recovery

### Database Backup
```bash
# Create backup script
cat > /opt/rag-engine/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/rag-engine/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup ChromaDB data
tar -czf $BACKUP_DIR/chroma_$DATE.tar.gz /opt/rag-engine/data/chroma

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/rag-engine/config

# Remove old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
EOF

chmod +x /opt/rag-engine/backup.sh

# Schedule backup
echo "0 2 * * * /opt/rag-engine/backup.sh" | crontab -
```

### Recovery Procedures
```bash
# Restore from backup
tar -xzf /opt/rag-engine/backups/chroma_20240101_020000.tar.gz -C /
tar -xzf /opt/rag-engine/backups/config_20240101_020000.tar.gz -C /

# Restart services
sudo systemctl restart rag-engine
sudo systemctl restart chroma
```

## Performance Tuning

### Application Optimization
```json
{
  "performance": {
    "workers": 4,
    "max_requests": 1000,
    "max_requests_jitter": 100,
    "preload_app": true,
    "worker_connections": 1000,
    "keepalive": 2
  }
}
```

### System Optimization
```bash
# Increase file limits
echo "raguser soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "raguser hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Network optimization
echo "net.core.somaxconn = 65536" | sudo tee -a /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65536" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Troubleshooting

### Common Issues

**1. Service Won't Start**
```bash
# Check logs
sudo journalctl -u rag-engine -f

# Check configuration
python -m rag_engine validate-config config/production.json

# Test manually
cd /opt/rag-engine
source venv/bin/activate
python -m rag_engine serve --config config/production.json
```

**2. High Memory Usage**
```bash
# Monitor memory
sudo systemctl status rag-engine
htop

# Adjust worker count
sudo systemctl edit rag-engine
# Add:
# [Service]
# Environment=WORKERS=2
```

**3. SSL Certificate Issues**
```bash
# Check certificate
sudo certbot certificates

# Renew certificate
sudo certbot renew --dry-run

# Force renewal
sudo certbot renew --force-renewal
```

**4. Database Connection Issues**
```bash
# Check ChromaDB status
sudo systemctl status chroma
curl http://localhost:8001/api/v1/heartbeat

# Reset database
sudo systemctl stop chroma
rm -rf /opt/rag-engine/data/chroma/*
sudo systemctl start chroma
```
