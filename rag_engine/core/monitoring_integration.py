"""
Comprehensive monitoring integration for RAG Engine.
Applies metrics collection, health monitoring, and observability throughout the system.
"""
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from functools import wraps
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

from .monitoring import MetricsCollector, SystemMetrics, ApplicationMetrics, HealthMonitor

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    last_call_time: Optional[datetime] = None
    error_rate: float = 0.0
    avg_duration: float = 0.0


class MonitoringIntegration:
    """Comprehensive monitoring integration for the RAG engine."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("monitoring_enabled", True)
        
        # Initialize monitoring components
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor(config)
        
        # Performance tracking
        self.operation_metrics: Dict[str, PerformanceMetrics] = {}
        self.metrics_lock = threading.Lock()
        
        # Component health tracking
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.last_health_check = {}
        
        # Alerting configuration
        self.alert_thresholds = config.get("alert_thresholds", {
            "error_rate": 0.05,  # 5%
            "response_time_p95": 2000,  # 2 seconds in ms
            "memory_usage": 0.85,  # 85%
            "cpu_usage": 0.80  # 80%
        })
        
        # Initialize health checks
        self._register_health_checks()
    
    def _register_health_checks(self):
        """Register health checks for RAG components."""
        
        async def check_llm_health():
            """Check LLM provider health."""
            try:
                # This would make a simple health check call to the LLM provider
                # For now, return a mock healthy status
                return {
                    "status": "healthy",
                    "response_time_ms": 150,
                    "last_successful_call": datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_error_time": datetime.now().isoformat()
                }
        
        async def check_embedding_health():
            """Check embedding provider health."""
            try:
                # This would make a simple health check call to the embedding provider
                return {
                    "status": "healthy",
                    "response_time_ms": 80,
                    "last_successful_call": datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_error_time": datetime.now().isoformat()
                }
        
        async def check_vectorstore_health():
            """Check vector store health."""
            try:
                # This would ping the vector store
                return {
                    "status": "healthy",
                    "response_time_ms": 25,
                    "documents_count": 1000,  # Mock count
                    "last_successful_query": datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_error_time": datetime.now().isoformat()
                }
        
        # Register health checks
        self.health_monitor.register_component("llm", check_llm_health)
        self.health_monitor.register_component("embedding", check_embedding_health)
        self.health_monitor.register_component("vectorstore", check_vectorstore_health)
    
    def create_monitored_method(self, operation_type: str, component_name: str = None):
        """Create a decorator that applies comprehensive monitoring."""
        
        def monitoring_decorator(func: Callable):
            operation_name = f"{operation_type}_{func.__name__}"
            if component_name:
                operation_name = f"{component_name}_{operation_name}"
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)
                
                start_time = time.time()
                success = False
                error = None
                
                try:
                    # Execute the function
                    result = await func(*args, **kwargs)
                    success = True
                    return result
                    
                except Exception as e:
                    error = e
                    success = False
                    raise
                    
                finally:
                    # Record metrics
                    duration = time.time() - start_time
                    self._record_operation_metrics(operation_name, duration, success, error)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                success = False
                error = None
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    success = True
                    return result
                    
                except Exception as e:
                    error = e
                    success = False
                    raise
                    
                finally:
                    # Record metrics
                    duration = time.time() - start_time
                    self._record_operation_metrics(operation_name, duration, success, error)
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return monitoring_decorator
    
    def _record_operation_metrics(self, operation_name: str, duration: float, 
                                success: bool, error: Exception = None):
        """Record metrics for an operation."""
        with self.metrics_lock:
            if operation_name not in self.operation_metrics:
                self.operation_metrics[operation_name] = PerformanceMetrics(operation_name)
            
            metrics = self.operation_metrics[operation_name]
            metrics.total_calls += 1
            metrics.total_duration += duration
            metrics.last_call_time = datetime.now()
            
            if success:
                metrics.successful_calls += 1
            else:
                metrics.failed_calls += 1
                logger.warning(f"Operation {operation_name} failed: {error}")
            
            # Update duration statistics
            metrics.min_duration = min(metrics.min_duration, duration)
            metrics.max_duration = max(metrics.max_duration, duration)
            metrics.avg_duration = metrics.total_duration / metrics.total_calls
            metrics.error_rate = metrics.failed_calls / metrics.total_calls
            
            # Check for alerts
            self._check_operation_alerts(operation_name, metrics)
    
    def _check_operation_alerts(self, operation_name: str, metrics: PerformanceMetrics):
        """Check if operation metrics trigger any alerts."""
        alerts = []
        
        # Error rate alert
        if metrics.error_rate > self.alert_thresholds.get("error_rate", 0.05):
            alerts.append({
                "type": "high_error_rate",
                "operation": operation_name,
                "current_rate": metrics.error_rate,
                "threshold": self.alert_thresholds["error_rate"],
                "severity": "critical" if metrics.error_rate > 0.2 else "warning"
            })
        
        # Response time alert
        response_time_ms = metrics.avg_duration * 1000
        threshold = self.alert_thresholds.get("response_time_p95", 2000)
        if response_time_ms > threshold:
            alerts.append({
                "type": "high_response_time",
                "operation": operation_name,
                "current_time_ms": response_time_ms,
                "threshold_ms": threshold,
                "severity": "warning"
            })
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all monitoring components."""
        
        # Get system metrics
        system_metrics = self.metrics_collector.get_system_metrics()
        
        # Get application metrics
        app_metrics = self.metrics_collector.get_application_metrics()
        
        # Get health status
        health_status = await self.health_monitor.get_comprehensive_health()
        
        # Get operation metrics
        operation_metrics = {}
        with self.metrics_lock:
            for name, metrics in self.operation_metrics.items():
                operation_metrics[name] = {
                    "total_calls": metrics.total_calls,
                    "successful_calls": metrics.successful_calls,
                    "failed_calls": metrics.failed_calls,
                    "error_rate": metrics.error_rate,
                    "avg_duration_ms": metrics.avg_duration * 1000,
                    "min_duration_ms": metrics.min_duration * 1000,
                    "max_duration_ms": metrics.max_duration * 1000,
                    "last_call_time": metrics.last_call_time.isoformat() if metrics.last_call_time else None
                }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "cpu_usage": system_metrics.cpu_usage,
                "memory_usage": system_metrics.memory_usage,
                "disk_usage": system_metrics.disk_usage,
                "network_io": system_metrics.network_io
            },
            "application_metrics": {
                "total_requests": app_metrics.total_requests,
                "error_rate": app_metrics.error_rate,
                "avg_response_time": app_metrics.avg_response_time,
                "active_connections": app_metrics.active_connections
            },
            "health_status": health_status,
            "operation_metrics": operation_metrics
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        lines = []
        
        # Operation metrics
        lines.append("# HELP rag_operation_total Total operations by type")
        lines.append("# TYPE rag_operation_total counter")
        
        with self.metrics_lock:
            for name, metrics in self.operation_metrics.items():
                lines.append(f'rag_operation_total{{operation="{name}",status="success"}} {metrics.successful_calls}')
                lines.append(f'rag_operation_total{{operation="{name}",status="error"}} {metrics.failed_calls}')
        
        lines.append("# HELP rag_operation_duration_seconds Operation duration")
        lines.append("# TYPE rag_operation_duration_seconds histogram")
        
        with self.metrics_lock:
            for name, metrics in self.operation_metrics.items():
                lines.append(f'rag_operation_duration_seconds_sum{{operation="{name}"}} {metrics.total_duration}')
                lines.append(f'rag_operation_duration_seconds_count{{operation="{name}"}} {metrics.total_calls}')
        
        lines.append("# HELP rag_operation_error_rate Error rate by operation")
        lines.append("# TYPE rag_operation_error_rate gauge")
        
        with self.metrics_lock:
            for name, metrics in self.operation_metrics.items():
                lines.append(f'rag_operation_error_rate{{operation="{name}"}} {metrics.error_rate}')
        
        return "\n".join(lines)
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self.metrics_lock:
            self.operation_metrics.clear()
        self.metrics_collector.reset_metrics()
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts based on metrics."""
        alerts = []
        
        # Check operation metrics for alerts
        with self.metrics_lock:
            for name, metrics in self.operation_metrics.items():
                # Error rate alerts
                if metrics.error_rate > self.alert_thresholds.get("error_rate", 0.05):
                    alerts.append({
                        "type": "high_error_rate",
                        "operation": name,
                        "current_value": metrics.error_rate,
                        "threshold": self.alert_thresholds["error_rate"],
                        "severity": "critical" if metrics.error_rate > 0.2 else "warning",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Response time alerts
                response_time_ms = metrics.avg_duration * 1000
                threshold = self.alert_thresholds.get("response_time_p95", 2000)
                if response_time_ms > threshold:
                    alerts.append({
                        "type": "high_response_time",
                        "operation": name,
                        "current_value": response_time_ms,
                        "threshold": threshold,
                        "severity": "warning",
                        "timestamp": datetime.now().isoformat()
                    })
        
        return alerts


# Convenience decorators for common operations
def llm_monitored(monitor: MonitoringIntegration):
    """Decorator for LLM operations with comprehensive monitoring."""
    return monitor.create_monitored_method("llm", "llm_provider")


def embedding_monitored(monitor: MonitoringIntegration):
    """Decorator for embedding operations with comprehensive monitoring."""
    return monitor.create_monitored_method("embedding", "embedding_provider")


def vectorstore_monitored(monitor: MonitoringIntegration):
    """Decorator for vector store operations with comprehensive monitoring."""
    return monitor.create_monitored_method("vectorstore", "vectorstore_provider")


def pipeline_monitored(monitor: MonitoringIntegration):
    """Decorator for pipeline operations with comprehensive monitoring."""
    return monitor.create_monitored_method("pipeline", "rag_pipeline")


# Example usage in RAG components
# class MonitoredRAGExample:
#     """Example of how to apply monitoring to RAG components."""
#     
#     def __init__(self, config: Dict[str, Any]):
#         self.monitor = MonitoringIntegration(config)
#     
#     @llm_monitored(monitor=None)
#     async def generate_response(self, prompt: str) -> str:
#         # Your LLM call here
#         pass
#     
#     @embedding_monitored(monitor=None)
#     async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
#         # Your embedding call here
#         pass
#     
#     @vectorstore_monitored(monitor=None)
#     async def query_vectorstore(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
#         # Your vector store query here
#         pass
#     
#     @pipeline_monitored(monitor=None)
#     async def full_rag_query(self, query: str) -> Dict[str, Any]:
#         """Full RAG query with comprehensive monitoring."""
#         # This would orchestrate the full RAG pipeline
#         # Each step will be monitored individually
#         embeddings = await self.generate_embeddings([query])
#         results = await self.query_vectorstore(embeddings[0])
#         response = await self.generate_response(f"Context: {results[0]['content']}\nQuery: {query}")
#         
#         return {
#             "query": query,
#             "response": response,
#             "sources": results
#         } 