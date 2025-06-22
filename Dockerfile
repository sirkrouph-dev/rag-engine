# Dockerfile for RAG Engine
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Expose ports for different frameworks
EXPOSE 8000 8001 8002 8501

# Environment variables
ENV PYTHONPATH=/app
ENV RAG_CONFIG_PATH=/app/config/production.json

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "rag_engine", "serve", "--config", "/app/config/production.json", "--framework", "fastapi", "--host", "0.0.0.0", "--port", "8000"]
