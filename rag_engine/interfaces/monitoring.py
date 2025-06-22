"""
Monitoring and metrics framework for API customization.
"""
import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from threading import Lock
import logging


@dataclass
class MetricsConfig:
    """Configuration for API metrics and monitoring."""
    
    # Basic Metrics
    enable_metrics: bool = True
    metrics_endpoint: str = "/metrics"
    enable_prometheus: bool = False
    
    # Performance Tracking
    track_response_times: bool = True
    track_request_counts: bool = True
    track_error_rates: bool = True
    track_concurrent_users: bool = True
    
    # Custom Metrics
    custom_metrics: List[str] = field(default_factory=list)
    
    # Alerting
    enable_alerting: bool = False
    response_time_threshold: float = 5.0  # seconds
    error_rate_threshold: float = 0.05  # 5%
    
    # Retention
    metrics_retention_minutes: int = 60
    detailed_logs: bool = False


class MetricsCollector:
    """Collects and stores API metrics."""
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.lock = Lock()
        
        # Metrics storage
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.status_codes = defaultdict(int)
        self.concurrent_users = 0
        self.active_requests = set()
          # Time-series data (for retention)
        self.time_series = deque(maxlen=self.config.metrics_retention_minutes)
        self.custom_metrics = defaultdict(list)
        
        # Setup logging
        self.logger = logging.getLogger("api_metrics")
    
    def record_request(self, endpoint: str, method: str, user_id: str = None) -> str:
        """Record a new request and return request ID."""
        request_id = f"{time.time()}_{endpoint}_{method}"
        
        with self.lock:
            self.request_counts[f"{method} {endpoint}"] += 1
            self.active_requests.add(request_id)
            if user_id:
                self.concurrent_users += 1
        
        return request_id
    
    def record_response(self, request_id: str, status_code: int, 
                       response_time: float, endpoint: str = None):
        """Record response metrics."""
        with self.lock:
            self.active_requests.discard(request_id)
            self.status_codes[status_code] += 1
            
            if endpoint and self.config.track_response_times:
                self.response_times[endpoint].append(response_time)
                # Keep only recent response times
                if len(self.response_times[endpoint]) > 100:
                    self.response_times[endpoint] = self.response_times[endpoint][-100:]
            
            if status_code >= 400:
                self.error_counts[endpoint or "unknown"] += 1
    
    def record_custom_metric(self, name: str, value: Any, tags: Dict[str, str] = None):
        """Record custom metric."""
        if name in self.config.custom_metrics:
            metric_data = {
                "value": value,
                "timestamp": time.time(),
                "tags": tags or {}
            }
            self.custom_metrics[name].append(metric_data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self.lock:
            # Calculate derived metrics
            total_requests = sum(self.request_counts.values())
            total_errors = sum(self.error_counts.values())
            error_rate = total_errors / total_requests if total_requests > 0 else 0
            
            # Average response times
            avg_response_times = {}
            for endpoint, times in self.response_times.items():
                if times:
                    avg_response_times[endpoint] = {
                        "avg": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times),
                        "count": len(times)
                    }
            
            return {
                "timestamp": time.time(),
                "request_counts": dict(self.request_counts),
                "error_counts": dict(self.error_counts),
                "status_codes": dict(self.status_codes),
                "error_rate": error_rate,
                "total_requests": total_requests,
                "active_requests": len(self.active_requests),
                "concurrent_users": self.concurrent_users,
                "avg_response_times": avg_response_times,
                "custom_metrics": dict(self.custom_metrics)
            }
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        metrics = self.get_metrics()
        prometheus_output = []
        
        # Request counts
        prometheus_output.append("# HELP api_requests_total Total API requests")
        prometheus_output.append("# TYPE api_requests_total counter")
        for endpoint, count in metrics["request_counts"].items():
            prometheus_output.append(f'api_requests_total{{endpoint="{endpoint}"}} {count}')
        
        # Error rate
        prometheus_output.append("# HELP api_error_rate Current error rate")
        prometheus_output.append("# TYPE api_error_rate gauge")
        prometheus_output.append(f'api_error_rate {metrics["error_rate"]}')
        
        # Active requests
        prometheus_output.append("# HELP api_active_requests Current active requests")
        prometheus_output.append("# TYPE api_active_requests gauge")
        prometheus_output.append(f'api_active_requests {metrics["active_requests"]}')
        
        # Response times
        prometheus_output.append("# HELP api_response_time_avg Average response time in seconds")
        prometheus_output.append("# TYPE api_response_time_avg gauge")
        for endpoint, times in metrics["avg_response_times"].items():
            prometheus_output.append(f'api_response_time_avg{{endpoint="{endpoint}"}} {times["avg"]}')
        
        return "\n".join(prometheus_output)
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self.lock:
            self.request_counts.clear()
            self.response_times.clear()
            self.error_counts.clear()
            self.status_codes.clear()
            self.custom_metrics.clear()
            self.concurrent_users = 0
            self.active_requests.clear()


class PerformanceProfiler:
    """Profiles API performance and identifies bottlenecks."""
    
    def __init__(self):
        self.profiles = {}
        self.lock = Lock()
    
    def start_profiling(self, request_id: str, endpoint: str):
        """Start profiling a request."""
        with self.lock:
            self.profiles[request_id] = {
                "endpoint": endpoint,
                "start_time": time.time(),
                "checkpoints": []
            }
    
    def add_checkpoint(self, request_id: str, name: str):
        """Add a performance checkpoint."""
        if request_id in self.profiles:
            self.profiles[request_id]["checkpoints"].append({
                "name": name,
                "timestamp": time.time()
            })
    
    def end_profiling(self, request_id: str) -> Dict[str, Any]:
        """End profiling and return results."""
        with self.lock:
            if request_id not in self.profiles:
                return {}
            
            profile = self.profiles.pop(request_id)
            profile["end_time"] = time.time()
            profile["total_time"] = profile["end_time"] - profile["start_time"]
            
            # Calculate checkpoint durations
            checkpoints = profile["checkpoints"]
            if checkpoints:
                for i, checkpoint in enumerate(checkpoints):
                    if i == 0:
                        checkpoint["duration"] = checkpoint["timestamp"] - profile["start_time"]
                    else:
                        checkpoint["duration"] = checkpoint["timestamp"] - checkpoints[i-1]["timestamp"]
            
            return profile


class HealthChecker:
    """Monitors API health and provides health check endpoints."""
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.health_checks = {}
        self.last_check = time.time()
    
    def register_health_check(self, name: str, check_function: Callable) -> None:
        """Register a custom health check."""
        self.health_checks[name] = check_function
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Run custom health checks
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                health_status["checks"][name] = {
                    "status": "healthy" if result else "unhealthy",
                    "details": result if isinstance(result, dict) else {}
                }
                if not result:
                    health_status["status"] = "degraded"
            except Exception as e:
                health_status["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        
        return health_status
    
    def check_health(self) -> Dict[str, Any]:
        """Check overall health status."""
        return self.get_health_status()
    
    def check_readiness(self) -> Dict[str, Any]:
        """Check if service is ready to handle requests."""
        health_status = self.get_health_status()
        return {
            "ready": health_status["status"] == "healthy",
            "checks": health_status["checks"],
            "timestamp": time.time()
        }
        


class MonitoringManager:
    """Central monitoring manager for API frameworks."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.profiler = PerformanceProfiler()
        self.health_checker = HealthChecker(config)
        self.alerting_enabled = config.enable_alerting
    
    def create_middleware(self, framework: str) -> Callable:
        """Create framework-specific monitoring middleware."""
        
        def fastapi_middleware(request, call_next):
            """FastAPI monitoring middleware."""
            # Start tracking
            endpoint = request.url.path
            method = request.method
            user_id = getattr(request.state, 'user', {}).get('user_id')
            
            request_id = self.metrics_collector.record_request(endpoint, method, user_id)
            
            if self.config.track_response_times:
                self.profiler.start_profiling(request_id, endpoint)
            
            start_time = time.time()
            
            # Process request
            response = call_next(request)
            
            # Record metrics
            response_time = time.time() - start_time
            self.metrics_collector.record_response(
                request_id, 
                response.status_code, 
                response_time, 
                endpoint
            )
            
            # Add metrics headers
            response.headers["X-Response-Time"] = str(response_time)
            response.headers["X-Request-ID"] = request_id
            
            return response
        
        def flask_middleware(app):
            """Flask monitoring middleware."""
            from flask import request, g, jsonify
            
            @app.before_request
            def before_request():
                g.start_time = time.time()
                g.request_id = self.metrics_collector.record_request(
                    request.endpoint or request.path,
                    request.method,
                    getattr(g, 'user', {}).get('user_id')
                )
            
            @app.after_request
            def after_request(response):
                if hasattr(g, 'start_time'):
                    response_time = time.time() - g.start_time
                    self.metrics_collector.record_response(
                        g.request_id,
                        response.status_code,
                        response_time,
                        request.endpoint or request.path
                    )
                    response.headers["X-Response-Time"] = str(response_time)
                    response.headers["X-Request-ID"] = g.request_id
                
                return response
        
        if framework == "fastapi":
            return fastapi_middleware
        elif framework == "flask":
            return flask_middleware
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def get_metrics_endpoint(self, framework: str) -> Callable:
        """Create metrics endpoint for framework."""
        
        def fastapi_metrics():
            """FastAPI metrics endpoint."""
            if self.config.enable_prometheus:
                from fastapi.responses import PlainTextResponse
                return PlainTextResponse(
                    self.metrics_collector.get_prometheus_metrics(),
                    media_type="text/plain"
                )
            else:
                return self.metrics_collector.get_metrics()
        
        def flask_metrics():
            """Flask metrics endpoint."""
            from flask import jsonify, Response
            if self.config.enable_prometheus:
                return Response(
                    self.metrics_collector.get_prometheus_metrics(),
                    mimetype="text/plain"
                )
            else:
                return jsonify(self.metrics_collector.get_metrics())
        
        if framework == "fastapi":
            return fastapi_metrics
        elif framework == "flask":
            return flask_metrics
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def get_health_endpoint(self, framework: str) -> Callable:
        """Create health check endpoint for framework."""
        
        def health_check():
            return self.health_checker.get_health_status()
        
        return health_check
