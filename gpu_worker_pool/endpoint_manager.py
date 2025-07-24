"""Endpoint management for multi-endpoint GPU Worker Pool."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
from urllib.parse import urlparse

from .models import EndpointInfo, create_endpoint_id_from_url
from .http_client import AsyncGPUStatsHTTPClient, ServiceUnavailableError, RetryableError
from .config import MultiEndpointConfigurationManager
from .error_recovery import (
    GracefulDegradationManager, RecoveryOrchestrator, 
    CircuitBreakerConfig, AllEndpointsUnavailableError
)

logger = logging.getLogger(__name__)


class EndpointManager(ABC):
    """Abstract base class for endpoint management."""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the endpoint manager."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the endpoint manager and cleanup resources."""
        pass
    
    @abstractmethod
    def get_healthy_endpoints(self) -> List[EndpointInfo]:
        """Get list of currently healthy endpoints."""
        pass
    
    @abstractmethod
    def get_all_endpoints(self) -> List[EndpointInfo]:
        """Get list of all configured endpoints."""
        pass
    
    @abstractmethod
    def get_endpoint_by_id(self, endpoint_id: str) -> Optional[EndpointInfo]:
        """Get endpoint information by ID."""
        pass
    
    @abstractmethod
    def is_endpoint_healthy(self, endpoint_id: str) -> bool:
        """Check if a specific endpoint is healthy."""
        pass


class AsyncEndpointManager(EndpointManager):
    """Async endpoint manager with health monitoring and connection lifecycle management."""
    
    def __init__(self, config: MultiEndpointConfigurationManager):
        """Initialize the endpoint manager.
        
        Args:
            config: Multi-endpoint configuration manager
        """
        self.config = config
        self._endpoints: Dict[str, EndpointInfo] = {}
        self._http_clients: Dict[str, AsyncGPUStatsHTTPClient] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._endpoints_lock = asyncio.Lock()
        
        # Health check state
        self._last_health_check: Optional[datetime] = None
        self._health_check_failures: Dict[str, int] = {}
        
        # Error recovery and graceful degradation
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=10.0,
            max_timeout_seconds=120.0,
            backoff_multiplier=1.5
        )
        self.degradation_manager = GracefulDegradationManager(circuit_config)
        self.recovery_orchestrator = RecoveryOrchestrator(self.degradation_manager)
        
        # Register for degradation/recovery notifications
        self.degradation_manager.on_degradation(self._on_endpoint_degradation)
        self.degradation_manager.on_recovery(self._on_endpoint_recovery)
        
        # Initialize endpoints from configuration
        self._initialize_endpoints()
    
    def _initialize_endpoints(self) -> None:
        """Initialize endpoint information from configuration."""
        endpoints_urls = self.config.get_service_endpoints()
        
        for url in endpoints_urls:
            try:
                endpoint_id = create_endpoint_id_from_url(url)
                
                # Avoid duplicate endpoint IDs
                original_endpoint_id = endpoint_id
                counter = 1
                while endpoint_id in self._endpoints:
                    endpoint_id = f"{original_endpoint_id}_{counter}"
                    counter += 1
                
                endpoint_info = EndpointInfo(
                    endpoint_id=endpoint_id,
                    url=url,
                    is_healthy=False,  # Will be determined by health checks
                    last_seen=datetime.now(),
                    total_gpus=0,
                    available_gpus=0,
                    response_time_ms=0.0
                )
                
                self._endpoints[endpoint_id] = endpoint_info
                self._health_check_failures[endpoint_id] = 0
                
                logger.info(f"Initialized endpoint {endpoint_id}: {url}")
                
            except Exception as e:
                logger.error(f"Failed to initialize endpoint {url}: {e}")
                continue
        
        if not self._endpoints:
            raise ValueError("No valid endpoints configured")
        
        logger.info(f"Initialized {len(self._endpoints)} endpoints")
    
    async def start(self) -> None:
        """Start the endpoint manager and health monitoring."""
        if self._is_running:
            logger.warning("Endpoint manager is already running")
            return
        
        logger.info("Starting endpoint manager")
        
        # Start recovery orchestrator
        await self.recovery_orchestrator.start()
        
        # Create HTTP clients for all endpoints
        for endpoint_id, endpoint_info in self._endpoints.items():
            client = AsyncGPUStatsHTTPClient(
                endpoint=endpoint_info.url,
                timeout=self.config.get_endpoint_timeout(),
                max_retries=self.config.get_endpoint_max_retries()
            )
            self._http_clients[endpoint_id] = client
            
            # Register endpoint with degradation manager
            self.degradation_manager.register_endpoint(endpoint_id)
        
        # Start health monitoring
        self._is_running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Perform initial health check
        await self._perform_health_checks()
        
        logger.info("Endpoint manager started successfully")
    
    async def stop(self) -> None:
        """Stop the endpoint manager and cleanup resources."""
        if not self._is_running:
            return
        
        logger.info("Stopping endpoint manager")
        
        self._is_running = False
        
        # Stop recovery orchestrator
        await self.recovery_orchestrator.stop()
        
        # Stop health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        
        # Close all HTTP clients
        for client in self._http_clients.values():
            await client.close()
        self._http_clients.clear()
        
        logger.info("Endpoint manager stopped")
    
    def get_healthy_endpoints(self) -> List[EndpointInfo]:
        """Get list of currently healthy endpoints."""
        return [endpoint for endpoint in self._endpoints.values() if endpoint.is_healthy]
    
    def get_all_endpoints(self) -> List[EndpointInfo]:
        """Get list of all configured endpoints."""
        return list(self._endpoints.values())
    
    def get_endpoint_by_id(self, endpoint_id: str) -> Optional[EndpointInfo]:
        """Get endpoint information by ID."""
        return self._endpoints.get(endpoint_id)
    
    def is_endpoint_healthy(self, endpoint_id: str) -> bool:
        """Check if a specific endpoint is healthy."""
        endpoint = self._endpoints.get(endpoint_id)
        return endpoint.is_healthy if endpoint else False
    
    def get_http_client(self, endpoint_id: str) -> Optional[AsyncGPUStatsHTTPClient]:
        """Get HTTP client for a specific endpoint."""
        return self._http_clients.get(endpoint_id)
    
    async def _health_check_loop(self) -> None:
        """Main health check loop."""
        while self._is_running:
            try:
                await self._perform_health_checks()
                
                # Wait for next health check interval
                await asyncio.sleep(self.config.get_endpoint_health_check_interval())
                
            except asyncio.CancelledError:
                logger.debug("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}", exc_info=True)
                # Continue with shorter delay on error
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all endpoints concurrently."""
        if not self._endpoints:
            return
        
        logger.debug(f"Performing health checks on {len(self._endpoints)} endpoints")
        
        # Create health check tasks for all endpoints
        tasks = []
        for endpoint_id in self._endpoints.keys():
            task = asyncio.create_task(self._check_endpoint_health(endpoint_id))
            tasks.append(task)
        
        # Wait for all health checks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        healthy_count = 0
        for i, result in enumerate(results):
            endpoint_id = list(self._endpoints.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Health check failed for endpoint {endpoint_id}: {result}")
                await self._mark_endpoint_unhealthy(endpoint_id, str(result))
            elif result:
                healthy_count += 1
        
        self._last_health_check = datetime.now()
        logger.debug(f"Health check completed: {healthy_count}/{len(self._endpoints)} endpoints healthy")
    
    async def _check_endpoint_health(self, endpoint_id: str) -> bool:
        """Check health of a specific endpoint.
        
        Args:
            endpoint_id: ID of the endpoint to check
            
        Returns:
            True if endpoint is healthy, False otherwise
        """
        endpoint = self._endpoints.get(endpoint_id)
        client = self._http_clients.get(endpoint_id)
        
        if not endpoint or not client:
            logger.error(f"Endpoint or client not found for {endpoint_id}")
            return False
        
        try:
            # Use circuit breaker for health check
            result = await self.degradation_manager.execute_with_degradation(
                endpoint_id,
                self._perform_raw_health_check,
                endpoint_id, client
            )
            return result
            
        except Exception as e:
            logger.debug(f"Health check failed for {endpoint_id} (expected with circuit breaker): {e}")
            return False
    
    async def _perform_raw_health_check(self, endpoint_id: str, client: AsyncGPUStatsHTTPClient) -> bool:
        """Perform the actual health check without circuit breaker logic.
        
        Args:
            endpoint_id: ID of the endpoint to check
            client: HTTP client for the endpoint
            
        Returns:
            True if endpoint is healthy, False otherwise
            
        Raises:
            Exception: Any error encountered during health check
        """
        start_time = time.time()
        
        # Fetch GPU stats to check endpoint health
        gpu_stats = await client.fetch_gpu_stats()
        
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        if gpu_stats is not None:
            # Endpoint is healthy
            await self._mark_endpoint_healthy(endpoint_id, gpu_stats, response_time_ms)
            return True
        else:
            # Endpoint returned None (unhealthy)
            await self._mark_endpoint_unhealthy(endpoint_id, "Received None response")
            raise ServiceUnavailableError("GPU stats returned None")
    
    async def _mark_endpoint_healthy(self, endpoint_id: str, gpu_stats, response_time_ms: float) -> None:
        """Mark an endpoint as healthy and update its information."""
        async with self._endpoints_lock:
            endpoint = self._endpoints.get(endpoint_id)
            if not endpoint:
                return
            
            was_unhealthy = not endpoint.is_healthy
            
            # Update endpoint information
            endpoint.is_healthy = True
            endpoint.last_seen = datetime.now()
            endpoint.total_gpus = gpu_stats.gpu_count
            endpoint.available_gpus = len([gpu for gpu in gpu_stats.gpus_summary 
                                         if gpu.utilization_percent < self.config.get_utilization_threshold() 
                                         and gpu.memory_usage_percent < self.config.get_memory_threshold()])
            endpoint.response_time_ms = response_time_ms
            
            # Reset failure count
            self._health_check_failures[endpoint_id] = 0
            
            if was_unhealthy:
                logger.info(f"Endpoint {endpoint_id} recovered: {endpoint.total_gpus} total GPUs, "
                          f"{endpoint.available_gpus} available, {response_time_ms:.1f}ms response time")
            else:
                logger.debug(f"Endpoint {endpoint_id} healthy: {endpoint.total_gpus} total GPUs, "
                           f"{endpoint.available_gpus} available, {response_time_ms:.1f}ms response time")
    
    async def _mark_endpoint_unhealthy(self, endpoint_id: str, reason: str) -> None:
        """Mark an endpoint as unhealthy."""
        async with self._endpoints_lock:
            endpoint = self._endpoints.get(endpoint_id)
            if not endpoint:
                return
            
            was_healthy = endpoint.is_healthy
            
            # Update endpoint status
            endpoint.is_healthy = False
            endpoint.total_gpus = 0
            endpoint.available_gpus = 0
            
            # Increment failure count
            self._health_check_failures[endpoint_id] = self._health_check_failures.get(endpoint_id, 0) + 1
            
            if was_healthy:
                logger.warning(f"Endpoint {endpoint_id} became unhealthy: {reason}")
            else:
                failure_count = self._health_check_failures[endpoint_id]
                logger.debug(f"Endpoint {endpoint_id} still unhealthy ({failure_count} failures): {reason}")
    
    def get_endpoint_statistics(self) -> Dict[str, any]:
        """Get comprehensive endpoint statistics for monitoring."""
        healthy_endpoints = self.get_healthy_endpoints()
        all_endpoints = self.get_all_endpoints()
        
        total_gpus = sum(endpoint.total_gpus for endpoint in healthy_endpoints)
        total_available_gpus = sum(endpoint.available_gpus for endpoint in healthy_endpoints)
        
        avg_response_time = 0.0
        if healthy_endpoints:
            avg_response_time = sum(endpoint.response_time_ms for endpoint in healthy_endpoints) / len(healthy_endpoints)
        
        return {
            "total_endpoints": len(all_endpoints),
            "healthy_endpoints": len(healthy_endpoints),
            "unhealthy_endpoints": len(all_endpoints) - len(healthy_endpoints),
            "total_gpus": total_gpus,
            "total_available_gpus": total_available_gpus,
            "average_response_time_ms": avg_response_time,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
            "endpoints": [
                {
                    "endpoint_id": endpoint.endpoint_id,
                    "url": endpoint.url,
                    "is_healthy": endpoint.is_healthy,
                    "last_seen": endpoint.last_seen.isoformat(),
                    "total_gpus": endpoint.total_gpus,
                    "available_gpus": endpoint.available_gpus,
                    "response_time_ms": endpoint.response_time_ms,
                    "failure_count": self._health_check_failures.get(endpoint.endpoint_id, 0)
                }
                for endpoint in all_endpoints
            ]
        }
    
    async def _on_endpoint_degradation(self, endpoint_id: str, error_message: str) -> None:
        """Handle endpoint degradation notification."""
        logger.warning(f"Endpoint {endpoint_id} degraded: {error_message}")
        
        # Mark endpoint as unhealthy if not already
        async with self._endpoints_lock:
            endpoint = self._endpoints.get(endpoint_id)
            if endpoint and endpoint.is_healthy:
                endpoint.is_healthy = False
                endpoint.last_seen = datetime.now()
        
        # Trigger recovery attempt
        await self.recovery_orchestrator.trigger_recovery_attempt(
            endpoint_id,
            lambda: self._attempt_endpoint_recovery(endpoint_id)
        )
    
    async def _on_endpoint_recovery(self, endpoint_id: str) -> None:
        """Handle endpoint recovery notification."""
        logger.info(f"Endpoint {endpoint_id} recovered")
        
        # Perform a health check to update endpoint information
        await self._check_endpoint_health(endpoint_id)
    
    async def _attempt_endpoint_recovery(self, endpoint_id: str) -> None:
        """Attempt to recover a failed endpoint."""
        logger.info(f"Attempting recovery for endpoint: {endpoint_id}")
        
        try:
            # Perform health check
            success = await self._check_endpoint_health(endpoint_id)
            
            if success:
                logger.info(f"Recovery successful for endpoint: {endpoint_id}")
            else:
                logger.warning(f"Recovery attempt unsuccessful for endpoint: {endpoint_id}")
                raise ServiceUnavailableError(f"Endpoint {endpoint_id} still unhealthy after recovery attempt")
            
        except Exception as e:
            logger.error(f"Recovery attempt failed for endpoint {endpoint_id}: {e}")
            raise
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get comprehensive degradation and recovery status."""
        system_status = self.degradation_manager.get_system_status()
        recovery_status = self.recovery_orchestrator.get_recovery_status()
        
        return {
            "degradation_manager": system_status,
            "recovery_orchestrator": recovery_status,
            "endpoint_health_summary": {
                "total_endpoints": len(self._endpoints),
                "healthy_count": len(self.get_healthy_endpoints()),
                "degraded_count": len(system_status.get("degraded_endpoints", [])),
                "queued_requests": system_status.get("queued_requests", 0)
            }
        }
    
    async def queue_request_when_degraded(self, request_func: Callable) -> None:
        """Queue a request for retry when system is degraded."""
        await self.degradation_manager.queue_request_for_retry(request_func)


class MockEndpointManager(EndpointManager):
    """Mock endpoint manager for testing purposes."""
    
    def __init__(self, mock_endpoints: Optional[List[EndpointInfo]] = None):
        """Initialize mock endpoint manager.
        
        Args:
            mock_endpoints: List of mock endpoints to use
        """
        self.mock_endpoints = mock_endpoints or []
        self._is_running = False
        self.start_call_count = 0
        self.stop_call_count = 0
    
    async def start(self) -> None:
        """Mock start method."""
        self._is_running = True
        self.start_call_count += 1
    
    async def stop(self) -> None:
        """Mock stop method."""
        self._is_running = False
        self.stop_call_count += 1
    
    def get_healthy_endpoints(self) -> List[EndpointInfo]:
        """Get mock healthy endpoints."""
        return [endpoint for endpoint in self.mock_endpoints if endpoint.is_healthy]
    
    def get_all_endpoints(self) -> List[EndpointInfo]:
        """Get all mock endpoints."""
        return self.mock_endpoints.copy()
    
    def get_endpoint_by_id(self, endpoint_id: str) -> Optional[EndpointInfo]:
        """Get mock endpoint by ID."""
        for endpoint in self.mock_endpoints:
            if endpoint.endpoint_id == endpoint_id:
                return endpoint
        return None
    
    def is_endpoint_healthy(self, endpoint_id: str) -> bool:
        """Check if mock endpoint is healthy."""
        endpoint = self.get_endpoint_by_id(endpoint_id)
        return endpoint.is_healthy if endpoint else False
    
    def add_mock_endpoint(self, endpoint: EndpointInfo) -> None:
        """Add a mock endpoint for testing."""
        self.mock_endpoints.append(endpoint)
    
    def set_endpoint_health(self, endpoint_id: str, is_healthy: bool, available_gpus: Optional[int] = None, response_time_ms: Optional[float] = None) -> None:
        """Set the health status of a mock endpoint."""
        endpoint = self.get_endpoint_by_id(endpoint_id)
        if endpoint:
            endpoint.is_healthy = is_healthy
            if available_gpus is not None:
                endpoint.available_gpus = available_gpus
            if response_time_ms is not None:
                endpoint.response_time_ms = response_time_ms
    
    def get_http_client(self, endpoint_id: str):
        """Get mock HTTP client for endpoint (returns mock for testing)."""
        from unittest.mock import AsyncMock
        if self.get_endpoint_by_id(endpoint_id):
            # Return a mock client
            return AsyncMock()
        return None