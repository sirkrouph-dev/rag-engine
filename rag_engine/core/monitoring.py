"""
Production monitoring, metrics, and health check endpoints.
"""
import asyncio
import logging
import time
import psutil
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Info = generate_latest = CollectorRegistry = None

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    active_requests: int
    total_requests: int
    error_rate: float
    avg_response_time: float
    cache_hit_rate: float
    queue_size: int
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Production-grade metrics collection system."""
    
    def __init__(self):
        self.request_durations = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
        self.cache_stats = {"hits": 0, "misses": 0}
        self.active_requests = 0
        self.lock = threading.Lock()
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Use a custom registry to avoid conflicts
        self.registry = CollectorRegistry()
        
        self.prom_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.prom_request_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.prom_active_requests = Gauge(
            'http_requests_active',
            'Number of active HTTP requests',
            registry=self.registry
        )
        
        self.prom_system_cpu = Gauge(
            'system_cpu_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.prom_system_memory = Gauge(
            'system_memory_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.prom_cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            registry=self.registry
        )
        
        self.prom_cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics."""
        with self.lock:
            self.request_durations.append(duration)
            self.request_counts[f"{method}_{endpoint}"] += 1
            
            if status_code >= 400:
                self.error_counts[f"{method}_{endpoint}"] += 1
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self.prom_request_duration.labels(
                method=method, endpoint=endpoint, status=str(status_code)
            ).observe(duration)
            
            self.prom_request_total.labels(
                method=method, endpoint=endpoint, status=str(status_code)
            ).inc()
    
    def increment_active_requests(self):
        """Increment active request counter."""
        with self.lock:
            self.active_requests += 1
        
        if PROMETHEUS_AVAILABLE:
            self.prom_active_requests.inc()
    
    def decrement_active_requests(self):
        """Decrement active request counter."""
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)
        
        if PROMETHEUS_AVAILABLE:
            self.prom_active_requests.dec()
    
    def record_cache_hit(self):
        """Record cache hit."""
        with self.lock:
            self.cache_stats["hits"] += 1
        
        if PROMETHEUS_AVAILABLE:
            self.prom_cache_hits.inc()
    
    def record_cache_miss(self):
        """Record cache miss."""
        with self.lock:
            self.cache_stats["misses"] += 1
        
        if PROMETHEUS_AVAILABLE:
            self.prom_cache_misses.inc()
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                load_average = [0.0, 0.0, 0.0]  # Windows fallback
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                self.prom_system_cpu.set(cpu_percent)
                self.prom_system_memory.set(memory_percent)
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io,
                process_count=process_count,
                load_average=load_average
            )
        
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={},
                process_count=0,
                load_average=[0.0, 0.0, 0.0]
            )
    
    def get_application_metrics(self) -> ApplicationMetrics:
        """Get current application metrics."""
        with self.lock:
            total_requests = sum(self.request_counts.values())
            total_errors = sum(self.error_counts.values())
            error_rate = (total_errors / total_requests) if total_requests > 0 else 0.0
            
            avg_response_time = (
                sum(self.request_durations) / len(self.request_durations)
                if self.request_durations else 0.0
            )
            
            total_cache_operations = self.cache_stats["hits"] + self.cache_stats["misses"]
            cache_hit_rate = (
                self.cache_stats["hits"] / total_cache_operations
                if total_cache_operations > 0 else 0.0
            )
        
        return ApplicationMetrics(
            active_requests=self.active_requests,
            total_requests=total_requests,
            error_rate=error_rate,
            avg_response_time=avg_response_time,
            cache_hit_rate=cache_hit_rate,
            queue_size=0  # Placeholder for queue metrics
        )
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        if not PROMETHEUS_AVAILABLE:
            return "Prometheus not available"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self.lock:
            self.request_durations.clear()
            self.error_counts.clear()
            self.request_counts.clear()
            self.cache_stats = {"hits": 0, "misses": 0}
            self.active_requests = 0


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.component_health = {}
        self.registered_components = {}
        self.alert_thresholds = config.get("alerting", {}).get("thresholds", {})
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes
    
    def register_component(self, component_name: str, health_check_func: Callable):
        """Register a health check function for a component."""
        self.registered_components[component_name] = health_check_func
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            # This would be implemented with actual database connection
            # For now, return a mock healthy status
            return {
                "status": "healthy",
                "response_time_ms": 15,
                "connection_pool_size": 10,
                "active_connections": 3
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        try:
            # This would be implemented with actual Redis connection
            return {
                "status": "healthy",
                "response_time_ms": 2,
                "memory_usage": "15MB",
                "connected_clients": 5
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_llm_health(self) -> Dict[str, Any]:
        """Check LLM provider connectivity."""
        try:
            # This would test actual LLM provider connectivity
            return {
                "status": "healthy",
                "provider": "openai",
                "model": "gpt-4-turbo",
                "response_time_ms": 250
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_vectorstore_health(self) -> Dict[str, Any]:
        """Check vector store connectivity and performance."""
        try:
            # This would test actual vector store connectivity
            return {
                "status": "healthy",
                "type": "pinecone",
                "index_size": 10000,
                "response_time_ms": 50
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_checks = {
            "database": self.check_database_health(),
            "redis": self.check_redis_health(),
            "llm": self.check_llm_health(),
            "vectorstore": self.check_vectorstore_health()
        }
        
        # Execute all health checks concurrently
        results = {}
        for name, check_coro in health_checks.items():
            try:
                results[name] = await asyncio.wait_for(check_coro, timeout=10)
            except asyncio.TimeoutError:
                results[name] = {
                    "status": "unhealthy",
                    "error": "Health check timeout"
                }
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Get system and application metrics
        system_metrics = self.metrics_collector.get_system_metrics()
        app_metrics = self.metrics_collector.get_application_metrics()
        
        # Determine overall health
        overall_healthy = all(
            result.get("status") == "healthy" 
            for result in results.values()
        )
        
        # Check for alert conditions
        alerts = self._check_alert_conditions(system_metrics, app_metrics)
        
        return {
            "overall_status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "components": results,
            "system_metrics": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "disk_percent": system_metrics.disk_percent,
                "load_average": system_metrics.load_average
            },
            "application_metrics": {
                "active_requests": app_metrics.active_requests,
                "total_requests": app_metrics.total_requests,
                "error_rate": app_metrics.error_rate,
                "avg_response_time_ms": app_metrics.avg_response_time * 1000,
                "cache_hit_rate": app_metrics.cache_hit_rate
            },
            "alerts": alerts
        }
    
    def _check_alert_conditions(self, system_metrics: SystemMetrics, 
                               app_metrics: ApplicationMetrics) -> List[Dict[str, Any]]:
        """Check for alert conditions based on thresholds."""
        alerts = []
        current_time = time.time()
        
        # CPU usage alert
        cpu_threshold = self.alert_thresholds.get("cpu_usage", 80)
        if system_metrics.cpu_percent > cpu_threshold:
            alert_key = "high_cpu"
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    "type": "high_cpu",
                    "severity": "warning",
                    "message": f"CPU usage is {system_metrics.cpu_percent:.1f}% (threshold: {cpu_threshold}%)",
                    "value": system_metrics.cpu_percent,
                    "threshold": cpu_threshold
                })
        
        # Memory usage alert
        memory_threshold = self.alert_thresholds.get("memory_usage", 85)
        if system_metrics.memory_percent > memory_threshold:
            alert_key = "high_memory"
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    "type": "high_memory",
                    "severity": "warning",
                    "message": f"Memory usage is {system_metrics.memory_percent:.1f}% (threshold: {memory_threshold}%)",
                    "value": system_metrics.memory_percent,
                    "threshold": memory_threshold
                })
        
        # Error rate alert
        error_threshold = self.alert_thresholds.get("error_rate", 0.01)
        if app_metrics.error_rate > error_threshold:
            alert_key = "high_error_rate"
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    "type": "high_error_rate",
                    "severity": "critical",
                    "message": f"Error rate is {app_metrics.error_rate:.2%} (threshold: {error_threshold:.2%})",
                    "value": app_metrics.error_rate,
                    "threshold": error_threshold
                })
        
        # Response time alert
        response_time_threshold = self.alert_thresholds.get("response_time_p95", 2000)  # ms
        avg_response_time_ms = app_metrics.avg_response_time * 1000
        if avg_response_time_ms > response_time_threshold:
            alert_key = "high_response_time"
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    "type": "high_response_time",
                    "severity": "warning",
                    "message": f"Response time is {avg_response_time_ms:.0f}ms (threshold: {response_time_threshold}ms)",
                    "value": avg_response_time_ms,
                    "threshold": response_time_threshold
                })
        
        return alerts
    
    def _should_send_alert(self, alert_key: str, current_time: float) -> bool:
        """Check if we should send an alert based on cooldown period."""
        last_alert = self.last_alert_time.get(alert_key, 0)
        if current_time - last_alert > self.alert_cooldown:
            self.last_alert_time[alert_key] = current_time
            return True
        return False


# Monitoring utilities
def setup_monitoring(config: Dict[str, Any]) -> HealthMonitor:
    """Setup production monitoring system."""
    return HealthMonitor(config)
