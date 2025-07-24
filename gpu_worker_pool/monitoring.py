"""Monitoring and metrics collection for the GPU Worker Pool system."""

import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Deque
from threading import Lock
import asyncio


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str]


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime
    details: Dict[str, Any]


class MetricsCollector(ABC):
    """Abstract base class for metrics collection."""
    
    @abstractmethod
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        pass
    
    @abstractmethod
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        pass
    
    @abstractmethod
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, List[MetricPoint]]:
        """Get all collected metrics."""
        pass
    
    @abstractmethod
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        pass


class InMemoryMetricsCollector(MetricsCollector):
    """In-memory metrics collector for development and testing."""
    
    def __init__(self, max_points_per_metric: int = 1000):
        """Initialize the metrics collector.
        
        Args:
            max_points_per_metric: Maximum number of points to keep per metric
        """
        self.max_points_per_metric = max_points_per_metric
        self._metrics: Dict[str, Deque[MetricPoint]] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self._lock = Lock()
    
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        self._record_metric(f"counter_{name}", value, labels or {})
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        self._record_metric(f"gauge_{name}", value, labels or {})
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        self._record_metric(f"histogram_{name}", value, labels or {})
    
    def _record_metric(self, name: str, value: float, labels: Dict[str, str]) -> None:
        """Record a metric point."""
        with self._lock:
            metric_point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels
            )
            self._metrics[name].append(metric_point)
    
    def get_metrics(self) -> Dict[str, List[MetricPoint]]:
        """Get all collected metrics."""
        with self._lock:
            return {name: list(points) for name, points in self._metrics.items()}
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._metrics.clear()
    
    def get_metric_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for all metrics."""
        with self._lock:
            summary = {}
            for name, points in self._metrics.items():
                if not points:
                    continue
                
                values = [p.value for p in points]
                summary[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": values[-1],
                    "latest_timestamp": points[-1].timestamp.isoformat()
                }
            
            return summary


class StructuredLogger:
    """Structured logger for the GPU Worker Pool system."""
    
    def __init__(self, name: str, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize the structured logger.
        
        Args:
            name: Logger name
            metrics_collector: Optional metrics collector for logging metrics
        """
        self.logger = logging.getLogger(name)
        self.metrics_collector = metrics_collector
        
        # Configure structured logging format
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with structured data."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with structured data."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with structured data."""
        self._log(logging.ERROR, message, **kwargs)
        
        # Record error metric
        if self.metrics_collector:
            self.metrics_collector.record_counter("errors", labels={"level": "error"})
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with structured data."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        """Internal logging method with structured data."""
        if kwargs:
            # Add structured data as JSON
            structured_data = json.dumps(kwargs, default=str)
            full_message = f"{message} | {structured_data}"
        else:
            full_message = message
        
        self.logger.log(level, full_message)
        
        # Record logging metric
        if self.metrics_collector:
            level_name = logging.getLevelName(level).lower()
            self.metrics_collector.record_counter("log_messages", labels={"level": level_name})


class HealthChecker:
    """Health checker for system components."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize the health checker.
        
        Args:
            metrics_collector: Optional metrics collector for health metrics
        """
        self.metrics_collector = metrics_collector
        self._health_checks: Dict[str, callable] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._lock = Lock()
    
    def register_health_check(self, name: str, check_func: callable) -> None:
        """Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns HealthCheckResult
        """
        with self._lock:
            self._health_checks[name] = check_func
    
    def unregister_health_check(self, name: str) -> None:
        """Unregister a health check.
        
        Args:
            name: Name of the health check to remove
        """
        with self._lock:
            self._health_checks.pop(name, None)
            self._last_results.pop(name, None)
    
    async def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks.
        
        Returns:
            Dictionary of health check results
        """
        results = {}
        
        with self._lock:
            checks = dict(self._health_checks)
        
        for name, check_func in checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                if not isinstance(result, HealthCheckResult):
                    result = HealthCheckResult(
                        name=name,
                        status="unhealthy",
                        message="Health check returned invalid result",
                        timestamp=datetime.now(),
                        details={"result": str(result)}
                    )
                
            except Exception as e:
                result = HealthCheckResult(
                    name=name,
                    status="unhealthy",
                    message=f"Health check failed: {e}",
                    timestamp=datetime.now(),
                    details={"error": str(e), "error_type": type(e).__name__}
                )
            
            results[name] = result
            
            # Record health check metric
            if self.metrics_collector:
                status_value = 1.0 if result.status == "healthy" else 0.0
                self.metrics_collector.record_gauge(
                    "health_check_status",
                    status_value,
                    labels={"check_name": name, "status": result.status}
                )
        
        with self._lock:
            self._last_results.update(results)
        
        return results
    
    def get_last_results(self) -> Dict[str, HealthCheckResult]:
        """Get the last health check results.
        
        Returns:
            Dictionary of last health check results
        """
        with self._lock:
            return dict(self._last_results)
    
    def get_overall_health(self) -> HealthCheckResult:
        """Get overall system health based on all checks.
        
        Returns:
            Overall health check result
        """
        with self._lock:
            results = dict(self._last_results)
        
        if not results:
            return HealthCheckResult(
                name="overall",
                status="unhealthy",
                message="No health checks registered",
                timestamp=datetime.now(),
                details={}
            )
        
        unhealthy_checks = [name for name, result in results.items() if result.status == "unhealthy"]
        degraded_checks = [name for name, result in results.items() if result.status == "degraded"]
        
        if unhealthy_checks:
            status = "unhealthy"
            message = f"Unhealthy checks: {', '.join(unhealthy_checks)}"
        elif degraded_checks:
            status = "degraded"
            message = f"Degraded checks: {', '.join(degraded_checks)}"
        else:
            status = "healthy"
            message = "All health checks passing"
        
        return HealthCheckResult(
            name="overall",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details={
                "total_checks": len(results),
                "healthy_checks": len([r for r in results.values() if r.status == "healthy"]),
                "degraded_checks": len(degraded_checks),
                "unhealthy_checks": len(unhealthy_checks)
            }
        )


class PerformanceMonitor:
    """Performance monitoring for the GPU Worker Pool system."""
    
    def __init__(self, metrics_collector: MetricsCollector, logger: StructuredLogger):
        """Initialize the performance monitor.
        
        Args:
            metrics_collector: Metrics collector for recording performance data
            logger: Structured logger for performance logging
        """
        self.metrics_collector = metrics_collector
        self.logger = logger
        self._request_times: Dict[str, float] = {}
        self._lock = Lock()
    
    def start_request_timer(self, request_id: str) -> None:
        """Start timing a request.
        
        Args:
            request_id: Unique identifier for the request
        """
        with self._lock:
            self._request_times[request_id] = time.time()
    
    def end_request_timer(self, request_id: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """End timing a request and record metrics.
        
        Args:
            request_id: Unique identifier for the request
            labels: Optional labels for the metric
            
        Returns:
            Request duration in seconds, or None if timer wasn't started
        """
        with self._lock:
            start_time = self._request_times.pop(request_id, None)
        
        if start_time is None:
            return None
        
        duration = time.time() - start_time
        
        # Record metrics
        self.metrics_collector.record_histogram("request_duration", duration, labels)
        self.metrics_collector.record_counter("requests_total", labels=labels)
        
        # Log performance data
        self.logger.info(
            "Request completed",
            request_id=request_id,
            duration_seconds=duration,
            labels=labels or {}
        )
        
        return duration
    
    def record_gpu_assignment(self, gpu_id: int, worker_id: str, assignment_duration: float) -> None:
        """Record GPU assignment metrics.
        
        Args:
            gpu_id: ID of the assigned GPU
            worker_id: ID of the worker
            assignment_duration: Time taken to assign GPU in seconds
        """
        labels = {"gpu_id": str(gpu_id)}
        
        self.metrics_collector.record_counter("gpu_assignments_total", labels=labels)
        self.metrics_collector.record_histogram("gpu_assignment_duration", assignment_duration, labels)
        
        self.logger.info(
            "GPU assignment completed",
            gpu_id=gpu_id,
            worker_id=worker_id,
            assignment_duration_seconds=assignment_duration
        )
    
    def record_gpu_release(self, gpu_id: int, worker_id: str, usage_duration: float) -> None:
        """Record GPU release metrics.
        
        Args:
            gpu_id: ID of the released GPU
            worker_id: ID of the worker
            usage_duration: Time GPU was used in seconds
        """
        labels = {"gpu_id": str(gpu_id)}
        
        self.metrics_collector.record_counter("gpu_releases_total", labels=labels)
        self.metrics_collector.record_histogram("gpu_usage_duration", usage_duration, labels)
        
        self.logger.info(
            "GPU release completed",
            gpu_id=gpu_id,
            worker_id=worker_id,
            usage_duration_seconds=usage_duration
        )
    
    def record_worker_blocked(self, worker_id: str, reason: str, queue_size: int) -> None:
        """Record worker blocking metrics.
        
        Args:
            worker_id: ID of the blocked worker
            reason: Reason for blocking
            queue_size: Current queue size
        """
        labels = {"reason": reason}
        
        self.metrics_collector.record_counter("workers_blocked_total", labels=labels)
        self.metrics_collector.record_gauge("worker_queue_size", queue_size)
        
        self.logger.info(
            "Worker blocked",
            worker_id=worker_id,
            reason=reason,
            queue_size=queue_size
        )
    
    def record_worker_unblocked(self, worker_id: str, wait_duration: float, queue_size: int) -> None:
        """Record worker unblocking metrics.
        
        Args:
            worker_id: ID of the unblocked worker
            wait_duration: Time worker was blocked in seconds
            queue_size: Current queue size after unblocking
        """
        self.metrics_collector.record_counter("workers_unblocked_total")
        self.metrics_collector.record_histogram("worker_wait_duration", wait_duration)
        self.metrics_collector.record_gauge("worker_queue_size", queue_size)
        
        self.logger.info(
            "Worker unblocked",
            worker_id=worker_id,
            wait_duration_seconds=wait_duration,
            queue_size=queue_size
        )
    
    def record_service_error(self, error_type: str, error_message: str, component: str) -> None:
        """Record service error metrics.
        
        Args:
            error_type: Type of error
            error_message: Error message
            component: Component where error occurred
        """
        labels = {"error_type": error_type, "component": component}
        
        self.metrics_collector.record_counter("service_errors_total", labels=labels)
        
        self.logger.error(
            "Service error occurred",
            error_type=error_type,
            error_message=error_message,
            component=component
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics.
        
        Returns:
            Dictionary with performance summary
        """
        metrics = self.metrics_collector.get_metrics()
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "metrics_count": len(metrics),
            "total_data_points": sum(len(points) for points in metrics.values())
        }
        
        # Add specific performance metrics if available
        if hasattr(self.metrics_collector, 'get_metric_summary'):
            summary["metric_summary"] = self.metrics_collector.get_metric_summary()
        
        return summary


def create_monitoring_system() -> Dict[str, Any]:
    """Create a complete monitoring system with all components.
    
    Returns:
        Dictionary containing all monitoring components
    """
    metrics_collector = InMemoryMetricsCollector()
    logger = StructuredLogger("gpu_worker_pool", metrics_collector)
    health_checker = HealthChecker(metrics_collector)
    performance_monitor = PerformanceMonitor(metrics_collector, logger)
    
    return {
        "metrics_collector": metrics_collector,
        "logger": logger,
        "health_checker": health_checker,
        "performance_monitor": performance_monitor
    }