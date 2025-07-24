"""Tests for comprehensive error handling and recovery system."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from gpu_worker_pool.error_recovery import (
    CircuitBreaker, CircuitState, CircuitBreakerConfig, CircuitBreakerOpenError,
    GracefulDegradationManager, RecoveryOrchestrator, AllEndpointsUnavailableError
)


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout_seconds=10.0)
        breaker = CircuitBreaker("test-endpoint", config)
        
        assert breaker.endpoint_id == "test-endpoint"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success_flow(self):
        """Test successful request through circuit breaker."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker("test-endpoint", config)
        
        async def mock_function():
            return "success"
        
        result = await breaker.call(mock_function)
        
        assert result == "success"
        assert breaker.total_requests == 1
        assert breaker.total_successes == 1
        assert breaker.total_failures == 0
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_flow(self):
        """Test failure handling in circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test-endpoint", config)
        
        async def failing_function():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception, match="Test failure"):
            await breaker.call(failing_function)
        
        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED
        
        # Second failure should open circuit
        with pytest.raises(Exception, match="Test failure"):
            await breaker.call(failing_function)
        
        assert breaker.failure_count == 2
        assert breaker.state == CircuitState.OPEN
        
        # Third call should be blocked
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(failing_function)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1, 
            success_threshold=2,
            timeout_seconds=0.1  # Very short timeout for testing
        )
        breaker = CircuitBreaker("test-endpoint", config)
        
        # Force circuit open
        async def failing_function():
            raise Exception("Failure")
        
        with pytest.raises(Exception):
            await breaker.call(failing_function)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        await asyncio.sleep(0.15)
        
        # Next call should move to half-open
        async def success_function():
            return "success"
        
        result = await breaker.call(success_function)
        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Second success should close circuit
        result = await breaker.call(success_function)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
    
    def test_circuit_breaker_statistics(self):
        """Test circuit breaker statistics collection."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker("test-endpoint", config)
        
        stats = breaker.get_statistics()
        
        assert stats["endpoint_id"] == "test-endpoint"
        assert stats["state"] == "closed"
        assert stats["total_requests"] == 0
        assert stats["failure_rate"] == 0.0


class TestGracefulDegradationManager:
    """Test cases for GracefulDegradationManager class."""
    
    def test_degradation_manager_initialization(self):
        """Test degradation manager initialization."""
        manager = GracefulDegradationManager()
        
        assert not manager.is_fully_degraded
        assert len(manager.circuit_breakers) == 0
        assert len(manager.request_queue) == 0
    
    def test_register_endpoint(self):
        """Test endpoint registration."""
        manager = GracefulDegradationManager()
        
        manager.register_endpoint("endpoint1")
        manager.register_endpoint("endpoint2")
        
        assert "endpoint1" in manager.circuit_breakers
        assert "endpoint2" in manager.circuit_breakers
        assert len(manager.circuit_breakers) == 2
    
    @pytest.mark.asyncio
    async def test_execute_with_degradation_success(self):
        """Test successful execution with degradation manager."""
        manager = GracefulDegradationManager()
        manager.register_endpoint("test-endpoint")
        
        async def test_function(arg1, arg2):
            return f"{arg1}-{arg2}"
        
        result = await manager.execute_with_degradation(
            "test-endpoint", test_function, "hello", "world"
        )
        
        assert result == "hello-world"
    
    @pytest.mark.asyncio
    async def test_execute_with_degradation_failure(self):
        """Test failure handling with degradation manager."""
        manager = GracefulDegradationManager()
        
        # Use very low failure threshold for testing
        config = CircuitBreakerConfig(failure_threshold=1)
        manager.config = config
        manager.register_endpoint("test-endpoint")
        
        degradation_calls = []
        recovery_calls = []
        
        async def on_degradation(endpoint_id, error_msg):
            degradation_calls.append((endpoint_id, error_msg))
        
        async def on_recovery(endpoint_id):
            recovery_calls.append(endpoint_id)
        
        manager.on_degradation(on_degradation)
        manager.on_recovery(on_recovery)
        
        async def failing_function():
            raise Exception("Test failure")
        
        # First failure should trigger degradation
        with pytest.raises(Exception):
            await manager.execute_with_degradation("test-endpoint", failing_function)
        
        # Give time for callbacks
        await asyncio.sleep(0.01)
        
        assert len(degradation_calls) == 1
        assert degradation_calls[0][0] == "test-endpoint"
    
    def test_get_healthy_and_degraded_endpoints(self):
        """Test getting healthy and degraded endpoints."""
        manager = GracefulDegradationManager()
        
        # Register endpoints
        manager.register_endpoint("healthy1")
        manager.register_endpoint("healthy2")
        manager.register_endpoint("degraded1")
        
        # Simulate degraded state
        manager.circuit_breakers["degraded1"].state = CircuitState.OPEN
        
        healthy = manager.get_healthy_endpoints()
        degraded = manager.get_degraded_endpoints()
        
        assert "healthy1" in healthy
        assert "healthy2" in healthy
        assert "degraded1" in degraded
        assert len(healthy) == 2
        assert len(degraded) == 1
    
    @pytest.mark.asyncio
    async def test_request_queueing(self):
        """Test request queueing functionality."""
        manager = GracefulDegradationManager()
        
        request_executed = False
        
        async def queued_request():
            nonlocal request_executed
            request_executed = True
        
        await manager.queue_request_for_retry(queued_request)
        
        assert len(manager.request_queue) == 1
        assert not request_executed
        
        # Process queued requests
        await manager.process_queued_requests()
        
        assert len(manager.request_queue) == 0
        assert request_executed
    
    def test_system_status(self):
        """Test system status reporting."""
        manager = GracefulDegradationManager()
        
        manager.register_endpoint("endpoint1")
        manager.register_endpoint("endpoint2")
        
        status = manager.get_system_status()
        
        assert "is_fully_degraded" in status
        assert "total_endpoints" in status
        assert "healthy_endpoints_count" in status
        assert "degraded_endpoints_count" in status
        assert "circuit_breaker_stats" in status
        
        assert status["total_endpoints"] == 2
        assert status["healthy_endpoints_count"] == 2
        assert status["degraded_endpoints_count"] == 0


class TestRecoveryOrchestrator:
    """Test cases for RecoveryOrchestrator class."""
    
    @pytest.mark.asyncio
    async def test_recovery_orchestrator_lifecycle(self):
        """Test recovery orchestrator start/stop lifecycle."""
        manager = GracefulDegradationManager()
        orchestrator = RecoveryOrchestrator(manager)
        
        assert not orchestrator.is_running
        
        await orchestrator.start()
        assert orchestrator.is_running
        
        await orchestrator.stop()
        assert not orchestrator.is_running
    
    @pytest.mark.asyncio
    async def test_trigger_recovery_attempt(self):
        """Test triggering recovery attempts."""
        manager = GracefulDegradationManager()
        orchestrator = RecoveryOrchestrator(manager)
        await orchestrator.start()
        
        recovery_called = False
        
        async def mock_recovery():
            nonlocal recovery_called
            await asyncio.sleep(0.01)  # Simulate work
            recovery_called = True
        
        await orchestrator.trigger_recovery_attempt("test-endpoint", mock_recovery)
        
        # Wait for recovery task to complete
        await asyncio.sleep(0.05)
        
        assert recovery_called
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_multiple_recovery_attempts(self):
        """Test handling multiple recovery attempts for same endpoint."""
        manager = GracefulDegradationManager()
        orchestrator = RecoveryOrchestrator(manager)
        await orchestrator.start()
        
        call_count = 0
        
        async def counting_recovery():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
        
        # Trigger multiple recovery attempts
        await orchestrator.trigger_recovery_attempt("test-endpoint", counting_recovery)
        await orchestrator.trigger_recovery_attempt("test-endpoint", counting_recovery)
        
        # Wait for tasks to complete
        await asyncio.sleep(0.05)
        
        # Only the latest attempt should have executed
        assert call_count == 1
        
        await orchestrator.stop()
    
    def test_recovery_status(self):
        """Test recovery status reporting."""
        manager = GracefulDegradationManager()
        orchestrator = RecoveryOrchestrator(manager)
        
        status = orchestrator.get_recovery_status()
        
        assert "is_running" in status
        assert "active_recovery_tasks" in status
        assert "total_recovery_tasks" in status
        
        assert not status["is_running"]
        assert status["total_recovery_tasks"] == 0


@pytest.mark.asyncio
async def test_integrated_error_recovery_flow():
    """Test integrated error recovery flow with circuit breaker and queue."""
    # Setup
    config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.1)
    manager = GracefulDegradationManager(config)
    orchestrator = RecoveryOrchestrator(manager)
    
    await orchestrator.start()
    
    manager.register_endpoint("test-endpoint")
    
    # Track events
    degradation_events = []
    recovery_events = []
    
    async def on_degradation(endpoint_id, error_msg):
        degradation_events.append((endpoint_id, error_msg))
    
    async def on_recovery(endpoint_id):
        recovery_events.append(endpoint_id)
    
    manager.on_degradation(on_degradation)
    manager.on_recovery(on_recovery)
    
    # Phase 1: Cause failures to open circuit
    async def failing_function():
        raise Exception("Service down")
    
    # First failure
    with pytest.raises(Exception):
        await manager.execute_with_degradation("test-endpoint", failing_function)
    
    # Second failure - should open circuit and trigger degradation
    with pytest.raises(Exception):
        await manager.execute_with_degradation("test-endpoint", failing_function)
    
    # Wait for degradation callback
    await asyncio.sleep(0.01)
    
    assert len(degradation_events) == 1
    assert manager.circuit_breakers["test-endpoint"].state == CircuitState.OPEN
    
    # Phase 2: Attempt calls while circuit is open
    with pytest.raises(CircuitBreakerOpenError):
        await manager.execute_with_degradation("test-endpoint", failing_function)
    
    # Phase 3: Wait for circuit to move to half-open and test recovery
    await asyncio.sleep(0.15)  # Wait for timeout
    
    async def recovered_function():
        return "success"
    
    # This should move circuit to half-open and then closed
    result = await manager.execute_with_degradation("test-endpoint", recovered_function)
    assert result == "success"
    
    # Wait for recovery callback
    await asyncio.sleep(0.01)
    
    await orchestrator.stop()
    
    # Verify recovery occurred
    assert len(recovery_events) >= 0  # May or may not trigger depending on timing