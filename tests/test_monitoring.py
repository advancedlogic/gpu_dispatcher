"""Tests for monitoring and logging capabilities."""

import asyncio
import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

from gpu_worker_pool.monitoring import (
    InMemoryMetricsCollector, StructuredLogger, HealthChecker, PerformanceMonitor,
    MetricPoint, HealthCheckResult, create_monitoring_system
)


class TestInMemoryMetricsCollector:
    """Test the in-memory metrics collector."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a metrics collector for testing."""
        return InMemoryMetricsCollector(max_points_per_metric=100)
    
    def test_record_counter(self, metrics_collector):
        """Test recording counter metrics."""
        metrics_collector.record_counter("test_counter", 1.0, {"label": "value"})
        metrics_collector.record_counter("test_counter", 2.0, {"label": "value2"})
        
        metrics = metrics_collector.get_metrics()
        assert "counter_test_counter" in metrics
        assert len(metrics["counter_test_counter"]) == 2
        
        # Check first metric point
        first_point = metrics["counter_test_counter"][0]
        assert first_point.value == 1.0
        assert first_point.labels == {"label": "value"}
        assert isinstance(first_point.timestamp, datetime)
    
    def test_record_gauge(self, metrics_collector):
        """Test recording gauge metrics."""
        metrics_collector.record_gauge("test_gauge", 42.5, {"gpu_id": "0"})
        
        metrics = metrics_collector.get_metrics()
        assert "gauge_test_gauge" in metrics
        assert len(metrics["gauge_test_gauge"]) == 1
        
        point = metrics["gauge_test_gauge"][0]
        assert point.value == 42.5
        assert point.labels == {"gpu_id": "0"}
    
    def test_record_histogram(self, metrics_collector):
        """Test recording histogram metrics."""
        metrics_collector.record_histogram("test_histogram", 0.123, {"operation": "gpu_assignment"})
        
        metrics = metrics_collector.get_metrics()
        assert "histogram_test_histogram" in metrics
        assert len(metrics["histogram_test_histogram"]) == 1
        
        point = metrics["histogram_test_histogram"][0]
        assert point.value == 0.123
        assert point.labels == {"operation": "gpu_assignment"}
    
    def test_max_points_per_metric(self):
        """Test that metrics collector respects max points limit."""
        collector = InMemoryMetricsCollector(max_points_per_metric=3)
        
        # Add more points than the limit
        for i in range(5):
            collector.record_counter("test_counter", float(i))
        
        metrics = collector.get_metrics()
        points = metrics["counter_test_counter"]
        
        # Should only keep the last 3 points
        assert len(points) == 3
        assert points[0].value == 2.0  # First kept point
        assert points[1].value == 3.0
        assert points[2].value == 4.0  # Last point
    
    def test_clear_metrics(self, metrics_collector):
        """Test clearing all metrics."""
        metrics_collector.record_counter("test_counter", 1.0)
        metrics_collector.record_gauge("test_gauge", 2.0)
        
        assert len(metrics_collector.get_metrics()) == 2
        
        metrics_collector.clear_metrics()
        assert len(metrics_collector.get_metrics()) == 0
    
    def test_get_metric_summary(self, metrics_collector):
        """Test getting metric summary statistics."""
        # Add some test data
        for i in range(5):
            metrics_collector.record_gauge("test_gauge", float(i * 10))
        
        summary = metrics_collector.get_metric_summary()
        assert "gauge_test_gauge" in summary
        
        gauge_summary = summary["gauge_test_gauge"]
        assert gauge_summary["count"] == 5
        assert gauge_summary["min"] == 0.0
        assert gauge_summary["max"] == 40.0
        assert gauge_summary["avg"] == 20.0
        assert gauge_summary["latest"] == 40.0


class TestStructuredLogger:
    """Test the structured logger."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a mock metrics collector."""
        return Mock()
    
    @pytest.fixture
    def structured_logger(self, metrics_collector):
        """Create a structured logger for testing."""
        return StructuredLogger("test_logger", metrics_collector)
    
    def test_info_logging(self, structured_logger, caplog):
        """Test info level logging."""
        with caplog.at_level(logging.INFO):
            structured_logger.info("Test message", key1="value1", key2=42)
        
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelname == "INFO"
        assert "Test message" in record.message
        assert '"key1": "value1"' in record.message
        assert '"key2": 42' in record.message
    
    def test_error_logging_with_metrics(self, structured_logger, metrics_collector, caplog):
        """Test error logging records metrics."""
        with caplog.at_level(logging.ERROR):
            structured_logger.error("Error occurred", error_code=500)
        
        # Should have logged the error
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "ERROR"
        
        # Should have recorded both log message metric and error metric
        assert metrics_collector.record_counter.call_count == 2
        calls = metrics_collector.record_counter.call_args_list
        
        # Check that both expected calls were made
        log_call = calls[0]
        error_call = calls[1]
        
        assert log_call[0] == ("log_messages",)
        assert log_call[1] == {"labels": {"level": "error"}}
        
        assert error_call[0] == ("errors",)
        assert error_call[1] == {"labels": {"level": "error"}}
    
    def test_logging_without_structured_data(self, structured_logger, caplog):
        """Test logging without additional structured data."""
        with caplog.at_level(logging.INFO):
            structured_logger.info("Simple message")
        
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "Simple message"
    
    def test_logging_with_metrics_collection(self, structured_logger, metrics_collector, caplog):
        """Test that all log levels record metrics."""
        with caplog.at_level(logging.DEBUG):
            structured_logger.debug("Debug message")
            structured_logger.info("Info message")
            structured_logger.warning("Warning message")
            structured_logger.error("Error message")
        
        # Should have recorded metrics for each log level
        expected_calls = [
            (("log_messages",), {"labels": {"level": "debug"}}),
            (("log_messages",), {"labels": {"level": "info"}}),
            (("log_messages",), {"labels": {"level": "warning"}}),
            (("log_messages",), {"labels": {"level": "error"}}),
            (("errors",), {"labels": {"level": "error"}})  # Additional error metric
        ]
        
        assert metrics_collector.record_counter.call_count == 5


class TestHealthChecker:
    """Test the health checker."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a mock metrics collector."""
        return Mock()
    
    @pytest.fixture
    def health_checker(self, metrics_collector):
        """Create a health checker for testing."""
        return HealthChecker(metrics_collector)
    
    def test_register_health_check(self, health_checker):
        """Test registering a health check."""
        def dummy_check():
            return HealthCheckResult("test", "healthy", "OK", datetime.now(), {})
        
        health_checker.register_health_check("test_check", dummy_check)
        
        # Check should be registered
        assert "test_check" in health_checker._health_checks
    
    def test_unregister_health_check(self, health_checker):
        """Test unregistering a health check."""
        def dummy_check():
            return HealthCheckResult("test", "healthy", "OK", datetime.now(), {})
        
        health_checker.register_health_check("test_check", dummy_check)
        health_checker.unregister_health_check("test_check")
        
        # Check should be removed
        assert "test_check" not in health_checker._health_checks
    
    @pytest.mark.asyncio
    async def test_run_health_checks_sync(self, health_checker, metrics_collector):
        """Test running synchronous health checks."""
        def healthy_check():
            return HealthCheckResult("healthy_check", "healthy", "All good", datetime.now(), {})
        
        def unhealthy_check():
            return HealthCheckResult("unhealthy_check", "unhealthy", "Something wrong", datetime.now(), {})
        
        health_checker.register_health_check("healthy", healthy_check)
        health_checker.register_health_check("unhealthy", unhealthy_check)
        
        results = await health_checker.run_health_checks()
        
        assert len(results) == 2
        assert results["healthy"].status == "healthy"
        assert results["unhealthy"].status == "unhealthy"
        
        # Should have recorded metrics
        assert metrics_collector.record_gauge.call_count == 2
    
    @pytest.mark.asyncio
    async def test_run_health_checks_async(self, health_checker):
        """Test running asynchronous health checks."""
        async def async_check():
            await asyncio.sleep(0.01)  # Simulate async work
            return HealthCheckResult("async_check", "healthy", "Async OK", datetime.now(), {})
        
        health_checker.register_health_check("async", async_check)
        
        results = await health_checker.run_health_checks()
        
        assert len(results) == 1
        assert results["async"].status == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self, health_checker):
        """Test handling of exceptions in health checks."""
        def failing_check():
            raise ValueError("Health check failed")
        
        health_checker.register_health_check("failing", failing_check)
        
        results = await health_checker.run_health_checks()
        
        assert len(results) == 1
        result = results["failing"]
        assert result.status == "unhealthy"
        assert "Health check failed" in result.message
        assert "ValueError" in result.details["error_type"]
    
    @pytest.mark.asyncio
    async def test_invalid_health_check_result(self, health_checker):
        """Test handling of invalid health check results."""
        def invalid_check():
            return "not a health check result"
        
        health_checker.register_health_check("invalid", invalid_check)
        
        results = await health_checker.run_health_checks()
        
        assert len(results) == 1
        result = results["invalid"]
        assert result.status == "unhealthy"
        assert "invalid result" in result.message
    
    def test_get_overall_health_all_healthy(self, health_checker):
        """Test overall health when all checks are healthy."""
        # Simulate some healthy results
        health_checker._last_results = {
            "check1": HealthCheckResult("check1", "healthy", "OK", datetime.now(), {}),
            "check2": HealthCheckResult("check2", "healthy", "OK", datetime.now(), {})
        }
        
        overall = health_checker.get_overall_health()
        assert overall.status == "healthy"
        assert "All health checks passing" in overall.message
    
    def test_get_overall_health_with_degraded(self, health_checker):
        """Test overall health with degraded checks."""
        health_checker._last_results = {
            "check1": HealthCheckResult("check1", "healthy", "OK", datetime.now(), {}),
            "check2": HealthCheckResult("check2", "degraded", "Slow", datetime.now(), {})
        }
        
        overall = health_checker.get_overall_health()
        assert overall.status == "degraded"
        assert "check2" in overall.message
    
    def test_get_overall_health_with_unhealthy(self, health_checker):
        """Test overall health with unhealthy checks."""
        health_checker._last_results = {
            "check1": HealthCheckResult("check1", "healthy", "OK", datetime.now(), {}),
            "check2": HealthCheckResult("check2", "unhealthy", "Failed", datetime.now(), {})
        }
        
        overall = health_checker.get_overall_health()
        assert overall.status == "unhealthy"
        assert "check2" in overall.message


class TestPerformanceMonitor:
    """Test the performance monitor."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a mock metrics collector."""
        return Mock()
    
    @pytest.fixture
    def structured_logger(self):
        """Create a mock structured logger."""
        return Mock()
    
    @pytest.fixture
    def performance_monitor(self, metrics_collector, structured_logger):
        """Create a performance monitor for testing."""
        return PerformanceMonitor(metrics_collector, structured_logger)
    
    def test_request_timer(self, performance_monitor, metrics_collector, structured_logger):
        """Test request timing functionality."""
        import time
        
        request_id = "test_request_123"
        labels = {"operation": "gpu_assignment"}
        
        # Start timer
        performance_monitor.start_request_timer(request_id)
        
        # Simulate some work
        time.sleep(0.01)
        
        # End timer
        duration = performance_monitor.end_request_timer(request_id, labels)
        
        # Should return a duration
        assert duration is not None
        assert duration > 0
        
        # Should have recorded metrics
        metrics_collector.record_histogram.assert_called_once_with("request_duration", duration, labels)
        metrics_collector.record_counter.assert_called_once_with("requests_total", labels=labels)
        
        # Should have logged
        structured_logger.info.assert_called_once()
    
    def test_request_timer_not_started(self, performance_monitor):
        """Test ending timer that was never started."""
        duration = performance_monitor.end_request_timer("nonexistent_request")
        assert duration is None
    
    def test_record_gpu_assignment(self, performance_monitor, metrics_collector, structured_logger):
        """Test recording GPU assignment metrics."""
        gpu_id = 0
        worker_id = "worker_123"
        assignment_duration = 0.5
        
        performance_monitor.record_gpu_assignment(gpu_id, worker_id, assignment_duration)
        
        # Should have recorded metrics
        expected_labels = {"gpu_id": "0"}
        metrics_collector.record_counter.assert_called_once_with("gpu_assignments_total", labels=expected_labels)
        metrics_collector.record_histogram.assert_called_once_with("gpu_assignment_duration", assignment_duration, expected_labels)
        
        # Should have logged
        structured_logger.info.assert_called_once()
    
    def test_record_gpu_release(self, performance_monitor, metrics_collector, structured_logger):
        """Test recording GPU release metrics."""
        gpu_id = 1
        worker_id = "worker_456"
        usage_duration = 120.0
        
        performance_monitor.record_gpu_release(gpu_id, worker_id, usage_duration)
        
        # Should have recorded metrics
        expected_labels = {"gpu_id": "1"}
        metrics_collector.record_counter.assert_called_once_with("gpu_releases_total", labels=expected_labels)
        metrics_collector.record_histogram.assert_called_once_with("gpu_usage_duration", usage_duration, expected_labels)
        
        # Should have logged
        structured_logger.info.assert_called_once()
    
    def test_record_worker_blocked(self, performance_monitor, metrics_collector, structured_logger):
        """Test recording worker blocking metrics."""
        worker_id = "worker_789"
        reason = "No GPUs available"
        queue_size = 5
        
        performance_monitor.record_worker_blocked(worker_id, reason, queue_size)
        
        # Should have recorded metrics
        expected_labels = {"reason": reason}
        metrics_collector.record_counter.assert_called_once_with("workers_blocked_total", labels=expected_labels)
        metrics_collector.record_gauge.assert_called_once_with("worker_queue_size", queue_size)
        
        # Should have logged
        structured_logger.info.assert_called_once()
    
    def test_record_worker_unblocked(self, performance_monitor, metrics_collector, structured_logger):
        """Test recording worker unblocking metrics."""
        worker_id = "worker_101"
        wait_duration = 30.0
        queue_size = 3
        
        performance_monitor.record_worker_unblocked(worker_id, wait_duration, queue_size)
        
        # Should have recorded metrics
        metrics_collector.record_counter.assert_called_once_with("workers_unblocked_total")
        metrics_collector.record_histogram.assert_called_once_with("worker_wait_duration", wait_duration)
        metrics_collector.record_gauge.assert_called_once_with("worker_queue_size", queue_size)
        
        # Should have logged
        structured_logger.info.assert_called_once()
    
    def test_record_service_error(self, performance_monitor, metrics_collector, structured_logger):
        """Test recording service error metrics."""
        error_type = "ConnectionError"
        error_message = "Failed to connect to GPU service"
        component = "gpu_monitor"
        
        performance_monitor.record_service_error(error_type, error_message, component)
        
        # Should have recorded metrics
        expected_labels = {"error_type": error_type, "component": component}
        metrics_collector.record_counter.assert_called_once_with("service_errors_total", labels=expected_labels)
        
        # Should have logged error
        structured_logger.error.assert_called_once()
    
    def test_get_performance_summary(self, performance_monitor, metrics_collector):
        """Test getting performance summary."""
        # Mock metrics data
        mock_metrics = {
            "counter_requests_total": [
                MetricPoint(datetime.now(), 1.0, {}),
                MetricPoint(datetime.now(), 1.0, {})
            ]
        }
        metrics_collector.get_metrics.return_value = mock_metrics
        
        # Mock metric summary if available
        metrics_collector.get_metric_summary = Mock(return_value={"test": "summary"})
        
        summary = performance_monitor.get_performance_summary()
        
        assert "timestamp" in summary
        assert summary["metrics_count"] == 1
        assert summary["total_data_points"] == 2
        assert summary["metric_summary"] == {"test": "summary"}


class TestMonitoringSystemIntegration:
    """Test the complete monitoring system integration."""
    
    def test_create_monitoring_system(self):
        """Test creating a complete monitoring system."""
        system = create_monitoring_system()
        
        # Should contain all components
        assert "metrics_collector" in system
        assert "logger" in system
        assert "health_checker" in system
        assert "performance_monitor" in system
        
        # Components should be properly initialized
        assert isinstance(system["metrics_collector"], InMemoryMetricsCollector)
        assert isinstance(system["logger"], StructuredLogger)
        assert isinstance(system["health_checker"], HealthChecker)
        assert isinstance(system["performance_monitor"], PerformanceMonitor)
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_workflow(self):
        """Test end-to-end monitoring workflow."""
        system = create_monitoring_system()
        
        metrics_collector = system["metrics_collector"]
        logger = system["logger"]
        health_checker = system["health_checker"]
        performance_monitor = system["performance_monitor"]
        
        # Record some metrics
        performance_monitor.record_gpu_assignment(0, "worker_1", 0.1)
        performance_monitor.record_worker_blocked("worker_2", "No GPUs", 1)
        
        # Register a health check
        def test_health_check():
            return HealthCheckResult("test", "healthy", "OK", datetime.now(), {})
        
        health_checker.register_health_check("test", test_health_check)
        
        # Run health checks
        health_results = await health_checker.run_health_checks()
        assert len(health_results) == 1
        assert health_results["test"].status == "healthy"
        
        # Get metrics
        metrics = metrics_collector.get_metrics()
        assert len(metrics) > 0
        
        # Get performance summary
        summary = performance_monitor.get_performance_summary()
        assert "timestamp" in summary
        assert summary["metrics_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])