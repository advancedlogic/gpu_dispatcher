"""Comprehensive error handling and recovery system for multi-endpoint GPU pool."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5           # Failures before opening
    success_threshold: int = 3           # Successes needed to close from half-open
    timeout_seconds: float = 30.0        # Time to wait before trying half-open
    max_timeout_seconds: float = 300.0   # Maximum timeout duration
    backoff_multiplier: float = 2.0      # Exponential backoff multiplier


@dataclass
class FailureRecord:
    """Record of a failure event."""
    timestamp: datetime
    error_type: str
    error_message: str
    endpoint_id: Optional[str] = None


class CircuitBreaker:
    """Per-endpoint circuit breaker for graceful degradation."""
    
    def __init__(self, endpoint_id: str, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker for an endpoint.
        
        Args:
            endpoint_id: Unique identifier for the endpoint
            config: Circuit breaker configuration
        """
        self.endpoint_id = endpoint_id
        self.config = config
        self.state = CircuitState.CLOSED
        
        # Failure tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
        
        # Exponential backoff state
        self.current_timeout = config.timeout_seconds
        self.consecutive_timeout_increases = 0
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.failure_history: List[FailureRecord] = []
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function call through the circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original function exceptions
        """
        self.total_requests += 1
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker for {self.endpoint_id} moved to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker for {self.endpoint_id} is OPEN. "
                    f"Next attempt at {self.next_attempt_time}"
                )
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self) -> None:
        """Handle successful request."""
        self.total_successes += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.current_timeout = self.config.timeout_seconds
                self.consecutive_timeout_increases = 0
                logger.info(f"Circuit breaker for {self.endpoint_id} CLOSED - service recovered")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    async def _on_failure(self, error: Exception) -> None:
        """Handle failed request."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        # Record failure for analysis
        failure_record = FailureRecord(
            timestamp=self.last_failure_time,
            error_type=type(error).__name__,
            error_message=str(error),
            endpoint_id=self.endpoint_id
        )
        self.failure_history.append(failure_record)
        
        # Keep only recent failures (last 100)
        if len(self.failure_history) > 100:
            self.failure_history = self.failure_history[-100:]
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self._calculate_next_attempt_time()
            logger.warning(
                f"Circuit breaker for {self.endpoint_id} OPENED after {self.failure_count} failures. "
                f"Next attempt at {self.next_attempt_time}"
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.next_attempt_time is None:
            return True
        return datetime.now() >= self.next_attempt_time
    
    def _calculate_next_attempt_time(self) -> None:
        """Calculate when to next attempt to reset the circuit."""
        # Exponential backoff with jitter
        import random
        
        if self.consecutive_timeout_increases > 0:
            self.current_timeout = min(
                self.current_timeout * self.config.backoff_multiplier,
                self.config.max_timeout_seconds
            )
        
        # Add jitter (±20% of timeout)
        jitter = self.current_timeout * 0.2 * (random.random() - 0.5)
        timeout_with_jitter = self.current_timeout + jitter
        
        self.next_attempt_time = datetime.now() + timedelta(seconds=timeout_with_jitter)
        self.consecutive_timeout_increases += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "endpoint_id": self.endpoint_id,
            "state": self.state.value,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "failure_rate": (self.total_failures / self.total_requests * 100) if self.total_requests > 0 else 0.0,
            "current_failure_count": self.failure_count,
            "current_success_count": self.success_count,
            "current_timeout": self.current_timeout,
            "next_attempt_time": self.next_attempt_time.isoformat() if self.next_attempt_time else None,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "recent_failures": len(self.failure_history),
            "consecutive_timeout_increases": self.consecutive_timeout_increases
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and blocking requests."""
    pass


class GracefulDegradationManager:
    """Manages graceful degradation across multiple endpoints."""
    
    def __init__(self, circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize graceful degradation manager.
        
        Args:
            circuit_breaker_config: Configuration for circuit breakers
        """
        self.config = circuit_breaker_config or CircuitBreakerConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.request_queue: List[Callable] = []
        self.queue_lock = asyncio.Lock()
        self.degradation_callbacks: List[Callable[[str, str], None]] = []
        self.recovery_callbacks: List[Callable[[str], None]] = []
        
        # System state
        self.is_fully_degraded = False
        self.degradation_start_time: Optional[datetime] = None
        self.last_healthy_endpoints: Set[str] = set()
        
    def register_endpoint(self, endpoint_id: str) -> None:
        """Register an endpoint for circuit breaker management."""
        if endpoint_id not in self.circuit_breakers:
            self.circuit_breakers[endpoint_id] = CircuitBreaker(endpoint_id, self.config)
            logger.debug(f"Registered circuit breaker for endpoint: {endpoint_id}")
    
    def on_degradation(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for when an endpoint becomes degraded."""
        self.degradation_callbacks.append(callback)
    
    def on_recovery(self, callback: Callable[[str], None]) -> None:
        """Register callback for when an endpoint recovers."""
        self.recovery_callbacks.append(callback)
    
    async def execute_with_degradation(self, endpoint_id: str, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with graceful degradation support.
        
        Args:
            endpoint_id: Endpoint identifier
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            AllEndpointsUnavailableError: If all endpoints are unavailable
        """
        if endpoint_id not in self.circuit_breakers:
            self.register_endpoint(endpoint_id)
        
        circuit_breaker = self.circuit_breakers[endpoint_id]
        old_state = circuit_breaker.state
        
        try:
            result = await circuit_breaker.call(func, *args, **kwargs)
            
            # Check for recovery
            if old_state != CircuitState.CLOSED and circuit_breaker.state == CircuitState.CLOSED:
                await self._notify_recovery(endpoint_id)
            
            return result
            
        except CircuitBreakerOpenError:
            # Circuit is open, check if we need to trigger degradation
            await self._check_system_degradation()
            raise
        
        except Exception as e:
            # Function failed, check if circuit state changed
            if old_state == CircuitState.CLOSED and circuit_breaker.state == CircuitState.OPEN:
                await self._notify_degradation(endpoint_id, str(e))
            
            await self._check_system_degradation()
            raise
    
    async def queue_request_for_retry(self, request_func: Callable) -> None:
        """Queue a request for retry when endpoints become available."""
        async with self.queue_lock:
            self.request_queue.append(request_func)
            logger.debug(f"Queued request for retry. Queue size: {len(self.request_queue)}")
    
    async def process_queued_requests(self) -> None:
        """Process queued requests when endpoints become available."""
        if not self.request_queue:
            return
        
        async with self.queue_lock:
            requests_to_process = self.request_queue.copy()
            self.request_queue.clear()
        
        logger.info(f"Processing {len(requests_to_process)} queued requests")
        
        for request_func in requests_to_process:
            try:
                await request_func()
            except Exception as e:
                logger.error(f"Failed to process queued request: {e}")
                # Re-queue failed requests
                async with self.queue_lock:
                    self.request_queue.append(request_func)
    
    def get_healthy_endpoints(self) -> List[str]:
        """Get list of endpoints that are currently healthy."""
        healthy = []
        for endpoint_id, breaker in self.circuit_breakers.items():
            if breaker.state == CircuitState.CLOSED:
                healthy.append(endpoint_id)
        return healthy
    
    def get_degraded_endpoints(self) -> List[str]:
        """Get list of endpoints that are currently degraded."""
        degraded = []
        for endpoint_id, breaker in self.circuit_breakers.items():
            if breaker.state in [CircuitState.OPEN, CircuitState.HALF_OPEN]:
                degraded.append(endpoint_id)
        return degraded
    
    async def _check_system_degradation(self) -> None:
        """Check if the entire system is in a degraded state."""
        healthy_endpoints = set(self.get_healthy_endpoints())
        total_endpoints = len(self.circuit_breakers)
        
        if total_endpoints == 0:
            return
        
        # Check if system just became fully degraded
        if len(healthy_endpoints) == 0 and not self.is_fully_degraded:
            self.is_fully_degraded = True
            self.degradation_start_time = datetime.now()
            logger.critical("System fully degraded - no healthy endpoints available")
            
        # Check if system recovered from full degradation
        elif len(healthy_endpoints) > 0 and self.is_fully_degraded:
            self.is_fully_degraded = False
            degradation_duration = datetime.now() - self.degradation_start_time if self.degradation_start_time else None
            logger.info(f"System recovered from full degradation. Duration: {degradation_duration}")
            
            # Process any queued requests
            await self.process_queued_requests()
        
        self.last_healthy_endpoints = healthy_endpoints
    
    async def _notify_degradation(self, endpoint_id: str, error_message: str) -> None:
        """Notify registered callbacks about endpoint degradation."""
        for callback in self.degradation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(endpoint_id, error_message)
                else:
                    callback(endpoint_id, error_message)
            except Exception as e:
                logger.error(f"Error in degradation callback: {e}")
    
    async def _notify_recovery(self, endpoint_id: str) -> None:
        """Notify registered callbacks about endpoint recovery."""
        for callback in self.recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(endpoint_id)
                else:
                    callback(endpoint_id)
            except Exception as e:
                logger.error(f"Error in recovery callback: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system degradation status."""
        healthy_endpoints = self.get_healthy_endpoints()
        degraded_endpoints = self.get_degraded_endpoints()
        
        return {
            "is_fully_degraded": self.is_fully_degraded,
            "degradation_start_time": self.degradation_start_time.isoformat() if self.degradation_start_time else None,
            "total_endpoints": len(self.circuit_breakers),
            "healthy_endpoints_count": len(healthy_endpoints),
            "degraded_endpoints_count": len(degraded_endpoints),
            "healthy_endpoints": healthy_endpoints,
            "degraded_endpoints": degraded_endpoints,
            "queued_requests": len(self.request_queue),
            "circuit_breaker_stats": {
                endpoint_id: breaker.get_statistics()
                for endpoint_id, breaker in self.circuit_breakers.items()
            }
        }


class AllEndpointsUnavailableError(Exception):
    """Raised when all endpoints are unavailable."""
    pass


class RecoveryOrchestrator:
    """Orchestrates recovery operations across the multi-endpoint system."""
    
    def __init__(self, degradation_manager: GracefulDegradationManager):
        """
        Initialize recovery orchestrator.
        
        Args:
            degradation_manager: The degradation manager to work with
        """
        self.degradation_manager = degradation_manager
        self.recovery_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        
    async def start(self) -> None:
        """Start the recovery orchestrator."""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Recovery orchestrator started")
    
    async def stop(self) -> None:
        """Stop the recovery orchestrator and cancel all recovery tasks."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all recovery tasks
        for endpoint_id, task in self.recovery_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.debug(f"Cancelled recovery task for endpoint: {endpoint_id}")
        
        self.recovery_tasks.clear()
        logger.info("Recovery orchestrator stopped")
    
    async def trigger_recovery_attempt(self, endpoint_id: str, recovery_func: Callable) -> None:
        """
        Trigger a recovery attempt for a specific endpoint.
        
        Args:
            endpoint_id: Endpoint to recover
            recovery_func: Function to execute for recovery
        """
        if not self.is_running:
            logger.warning(f"Recovery orchestrator not running, ignoring recovery attempt for {endpoint_id}")
            return
        
        # Cancel existing recovery task if running
        if endpoint_id in self.recovery_tasks:
            existing_task = self.recovery_tasks[endpoint_id]
            if not existing_task.done():
                existing_task.cancel()
        
        # Start new recovery task
        task = asyncio.create_task(self._recovery_task(endpoint_id, recovery_func))
        self.recovery_tasks[endpoint_id] = task
        logger.info(f"Started recovery task for endpoint: {endpoint_id}")
    
    async def _recovery_task(self, endpoint_id: str, recovery_func: Callable) -> None:
        """Execute recovery task for an endpoint."""
        try:
            await recovery_func()
            logger.info(f"Recovery successful for endpoint: {endpoint_id}")
        except asyncio.CancelledError:
            logger.debug(f"Recovery task cancelled for endpoint: {endpoint_id}")
            raise
        except Exception as e:
            logger.error(f"Recovery failed for endpoint {endpoint_id}: {e}")
        finally:
            # Clean up task reference
            self.recovery_tasks.pop(endpoint_id, None)
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get status of all recovery operations."""
        return {
            "is_running": self.is_running,
            "active_recovery_tasks": {
                endpoint_id: {
                    "is_running": not task.done(),
                    "is_cancelled": task.cancelled(),
                    "is_done": task.done()
                }
                for endpoint_id, task in self.recovery_tasks.items()
            },
            "total_recovery_tasks": len(self.recovery_tasks)
        }