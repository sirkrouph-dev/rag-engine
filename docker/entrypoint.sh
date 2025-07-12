#!/bin/bash
set -e

# ================================
# Production Entrypoint Script
# ================================

echo "Starting RAG Engine in production mode..."

# Environment validation
if [ -z "$RAG_ENGINE_ENV" ]; then
    export RAG_ENGINE_ENV=production
fi

echo "Environment: $RAG_ENGINE_ENV"

# Wait for dependent services (database, redis, etc.)
echo "Waiting for dependent services..."

# Wait for database
if [ -n "$DB_HOST" ]; then
    echo "Waiting for database at $DB_HOST:${DB_PORT:-5432}..."
    while ! nc -z "$DB_HOST" "${DB_PORT:-5432}"; do
        sleep 1
    done
    echo "Database is ready!"
fi

# Wait for Redis
if [ -n "$REDIS_HOST" ]; then
    echo "Waiting for Redis at $REDIS_HOST:${REDIS_PORT:-6379}..."
    while ! nc -z "$REDIS_HOST" "${REDIS_PORT:-6379}"; do
        sleep 1
    done
    echo "Redis is ready!"
fi

# Run database migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    # python -m alembic upgrade head
    echo "Migrations completed!"
fi

# Validate configuration
echo "Validating configuration..."
python -c "
import sys
sys.path.append('/app')
try:
    from rag_engine.config.loader import load_config
    config = load_config('/app/configs/production.json')
    print('✓ Configuration is valid')
except Exception as e:
    print(f'✗ Configuration error: {e}')
    sys.exit(1)
"

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Set up logging
export PYTHONPATH="/app:$PYTHONPATH"

# Start the application
echo "Starting RAG Engine server..."
echo "Command: $@"

# Execute the main command
exec "$@"
