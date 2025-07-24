"""Simple client interface for the GPU Worker Pool system."""

import asyncio
import logging
from typing import Optional, Dict, Any, AsyncContextManager, List, Union, Callable
from contextlib import asynccontextmanager

from .models import GPUAssignment, PoolStatus, MultiEndpointPoolStatus, GlobalGPUAssignment
from .worker_pool_manager import AsyncWorkerPoolManager
from .config import EnvironmentConfigurationManager, MultiEndpointConfigurationManager, EnvironmentMultiEndpointConfigurationManager
from .gpu_monitor import AsyncGPUMonitor
from .multi_endpoint_gpu_monitor import MultiEndpointGPUMonitor
from .gpu_allocator import ThresholdBasedGPUAllocator
from .worker_queue import FIFOWorkerQueue
from .http_client import AsyncGPUStatsHTTPClient
from .multi_endpoint_http_client import AsyncMultiEndpointHTTPClientPool
from .endpoint_manager import AsyncEndpointManager
from .load_balancer import MultiEndpointLoadBalancer
from .unified_metrics import UnifiedPoolMetrics, MetricsFormatter

logger = logging.getLogger(__name__)


class GPUWorkerPoolClient:
    """
    Client interface for the GPU Worker Pool system with multi-endpoint support.
    
    This class provides an easy-to-use wrapper around the worker pool manager
    with context manager support for automatic resource cleanup. It supports
    both single-endpoint (backward compatibility) and multi-endpoint configurations.
    
    Example usage:
        # Single endpoint (backward compatibility)
        client = GPUWorkerPoolClient(service_endpoint="http://localhost:8000")
        await client.start()
        try:
            assignment = await client.request_gpu()
            # Use the GPU...
            await client.release_gpu(assignment)
        finally:
            await client.stop()
        
        # Multi-endpoint using environment variables
        # Set GPU_STATS_SERVICE_ENDPOINTS="http://server1:8000,http://server2:8000"
        client = GPUWorkerPoolClient()
        await client.start()
        try:
            assignment = await client.request_gpu()  # Returns global GPU assignment
            # Use the GPU...
            await client.release_gpu(assignment)
        finally:
            await client.stop()
        
        # Context manager usage (recommended)
        async with GPUWorkerPoolClient() as client:
            assignment = await client.request_gpu()
            # Use the GPU...
            await client.release_gpu(assignment)
    """
    
    def __init__(self,
                 service_endpoint: Optional[str] = None,
                 service_endpoints: Optional[str] = None,
                 memory_threshold: Optional[float] = None,
                 utilization_threshold: Optional[float] = None,
                 polling_interval: Optional[int] = None,
                 worker_timeout: float = 300.0,
                 request_timeout: float = 30.0,
                 load_balancing_strategy: Optional[str] = None):
        """
        Initialize the GPU Worker Pool client.
        
        Args:
            service_endpoint: Single GPU statistics service endpoint (overrides env var)
            service_endpoints: Comma-separated list of endpoints for multi-endpoint mode (overrides env var)
            memory_threshold: Memory usage threshold percentage (overrides env var)
            utilization_threshold: GPU utilization threshold percentage (overrides env var)
            polling_interval: Statistics polling interval in seconds (overrides env var)
            worker_timeout: Timeout for worker GPU requests in seconds
            request_timeout: Timeout for individual HTTP requests in seconds
            load_balancing_strategy: Load balancing strategy for multi-endpoint mode (overrides env var)
        """
        self.service_endpoint = service_endpoint
        self.service_endpoints = service_endpoints
        self.memory_threshold = memory_threshold
        self.utilization_threshold = utilization_threshold
        self.polling_interval = polling_interval
        self.worker_timeout = worker_timeout
        self.request_timeout = request_timeout
        self.load_balancing_strategy = load_balancing_strategy
        
        # Internal components (initialized in start())
        self._config: Optional[EnvironmentConfigurationManager] = None
        self._multi_config: Optional[MultiEndpointConfigurationManager] = None
        self._http_client: Optional[AsyncGPUStatsHTTPClient] = None
        self._multi_http_client: Optional[AsyncMultiEndpointHTTPClientPool] = None
        self._endpoint_manager: Optional[EndpointManager] = None
        self._load_balancer: Optional[MultiEndpointLoadBalancer] = None
        self._gpu_monitor: Optional[AsyncGPUMonitor] = None
        self._multi_gpu_monitor: Optional[MultiEndpointGPUMonitor] = None
        self._gpu_allocator: Optional[ThresholdBasedGPUAllocator] = None
        self._worker_queue: Optional[FIFOWorkerQueue] = None
        self._pool_manager: Optional[AsyncWorkerPoolManager] = None
        
        # Unified metrics aggregator
        self._unified_metrics: Optional[UnifiedPoolMetrics] = None
        
        # Mode detection (detect mode at initialization)
        self._is_multi_endpoint_mode = self._detect_multi_endpoint_mode()
        
        # Initialize unified metrics aggregator
        self._unified_metrics = UnifiedPoolMetrics(self._is_multi_endpoint_mode)
        
        # State tracking
        self._is_started = False
        
        logger.info("GPUWorkerPoolClient initialized")
    
    async def start(self) -> None:
        """
        Start the GPU worker pool client and all underlying components.
        
        This method must be called before using the client to request GPUs.
        
        Raises:
            RuntimeError: If the client is already started
            Exception: If initialization fails
        """
        if self._is_started:
            raise RuntimeError("Client is already started")
        
        logger.info("Starting GPU worker pool client")
        
        try:
            if self._is_multi_endpoint_mode:
                logger.info("Starting in multi-endpoint mode")
                await self._start_multi_endpoint_mode()
            else:
                logger.info("Starting in single-endpoint mode (backward compatibility)")
                await self._start_single_endpoint_mode()
            
            self._is_started = True
            mode = "multi-endpoint" if self._is_multi_endpoint_mode else "single-endpoint"
            logger.info(f"GPU worker pool client started successfully in {mode} mode")
            
        except Exception as e:
            logger.error(f"Failed to start GPU worker pool client: {e}")
            # Cleanup on failure
            await self._cleanup()
            raise
    
    async def stop(self) -> None:
        """
        Stop the GPU worker pool client and cleanup all resources.
        
        This method should be called when done using the client to ensure
        proper cleanup of resources and background tasks.
        """
        if not self._is_started:
            logger.debug("Client is not started, nothing to stop")
            return
        
        logger.info("Stopping GPU worker pool client")
        
        try:
            await self._cleanup()
            self._is_started = False
            logger.info("GPU worker pool client stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during client shutdown: {e}")
            raise
    
    async def request_gpu(self, timeout: Optional[float] = None) -> GPUAssignment:
        """
        Request a GPU assignment for the current worker.
        
        This method will either immediately assign an available GPU or block
        until a GPU becomes available, respecting the configured resource thresholds.
        
        In multi-endpoint mode, returns a GPUAssignment with global GPU ID.
        In single-endpoint mode, returns a GPUAssignment with local GPU ID.
        
        Args:
            timeout: Optional timeout in seconds. If None, uses client's worker_timeout
            
        Returns:
            GPUAssignment containing the assigned GPU ID and assignment details.
            In multi-endpoint mode, gpu_id will be in format 'endpoint_id:local_gpu_id'.
            
        Raises:
            RuntimeError: If the client is not started
            WorkerTimeoutError: If the request times out
            Exception: If GPU assignment fails
        """
        if not self._is_started:
            raise RuntimeError("Client must be started before requesting GPU")
        
        if not self._pool_manager:
            raise RuntimeError("Pool manager is not initialized")
        
        logger.debug(f"Requesting GPU assignment in {'multi-endpoint' if self._is_multi_endpoint_mode else 'single-endpoint'} mode")
        
        try:
            assignment = await self._pool_manager.request_gpu(timeout=timeout)
            
            # Log appropriate message based on mode
            if self._is_multi_endpoint_mode:
                logger.info(f"Global GPU assignment successful: GPU {assignment.gpu_id} assigned to worker {assignment.worker_id}")
            else:
                logger.info(f"GPU assignment successful: GPU {assignment.gpu_id} assigned to worker {assignment.worker_id}")
            
            return assignment
            
        except Exception as e:
            logger.error(f"GPU request failed: {e}")
            raise
    
    async def release_gpu(self, assignment: GPUAssignment) -> None:
        """
        Release a GPU assignment.
        
        This method should be called when the worker is done using the GPU
        to make it available for other workers. Supports both local GPU IDs
        (single-endpoint mode) and global GPU IDs (multi-endpoint mode).
        
        Args:
            assignment: The GPU assignment to release. Can contain either
                       local GPU ID or global GPU ID depending on mode.
            
        Raises:
            RuntimeError: If the client is not started
            ValueError: If the assignment is invalid
            Exception: If GPU release fails
        """
        if not self._is_started:
            raise RuntimeError("Client must be started before releasing GPU")
        
        if not self._pool_manager:
            raise RuntimeError("Pool manager is not initialized")
        
        gpu_id_type = "global" if self._is_multi_endpoint_mode else "local"
        logger.debug(f"Releasing {gpu_id_type} GPU assignment: GPU {assignment.gpu_id} from worker {assignment.worker_id}")
        
        try:
            await self._pool_manager.release_gpu(assignment)
            logger.info(f"{gpu_id_type.capitalize()} GPU release successful: GPU {assignment.gpu_id} released from worker {assignment.worker_id}")
            
        except Exception as e:
            logger.error(f"GPU release failed: {e}")
            raise
    
    def get_pool_status(self) -> Union[PoolStatus, MultiEndpointPoolStatus]:
        """
        Get the current unified status of the GPU worker pool.
        
        Returns:
            PoolStatus (single-endpoint mode) or MultiEndpointPoolStatus (multi-endpoint mode)
            containing current pool metrics with proper aggregation and endpoint health information
            
        Raises:
            RuntimeError: If the client is not started
        """
        if not self._is_started:
            raise RuntimeError("Client must be started before getting pool status")
        
        if not self._pool_manager or not self._unified_metrics:
            raise RuntimeError("Pool manager or unified metrics is not initialized")
        
        # Get base pool status from pool manager
        base_status = self._pool_manager.get_pool_status()
        
        # Gather additional information for multi-endpoint mode
        endpoints = None
        endpoint_health = None
        
        if self._is_multi_endpoint_mode:
            if self._endpoint_manager:
                endpoints = self._endpoint_manager.get_all_endpoints()
            
            if self._multi_gpu_monitor:
                endpoint_health = self._multi_gpu_monitor.get_endpoint_health_status()
        
        # Create unified status
        return self._unified_metrics.create_unified_pool_status(
            pool_status=base_status,
            endpoints=endpoints,
            endpoint_health=endpoint_health
        )
    
    def get_multi_endpoint_pool_status(self) -> Optional[MultiEndpointPoolStatus]:
        """
        Get the current multi-endpoint pool status with per-endpoint details.
        
        This method is only available in multi-endpoint mode and provides
        detailed information about each endpoint's health and GPU availability.
        
        Returns:
            MultiEndpointPoolStatus with detailed endpoint information,
            or None if not in multi-endpoint mode
            
        Raises:
            RuntimeError: If the client is not started
        """
        if not self._is_started:
            raise RuntimeError("Client must be started before getting multi-endpoint status")
        
        if not self._is_multi_endpoint_mode:
            return None
        
        if not self._multi_gpu_monitor:
            raise RuntimeError("Multi-endpoint GPU monitor is not initialized")
        
        # Get current worker statistics from pool manager
        pool_status = self._pool_manager.get_pool_status()
        
        # Create multi-endpoint pool status with comprehensive information
        return self._multi_gpu_monitor.create_multi_endpoint_pool_status(
            active_workers=pool_status.active_workers,
            blocked_workers=pool_status.blocked_workers,
            gpu_assignments=getattr(pool_status, 'gpu_assignments', {})
        )
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get unified detailed pool metrics for monitoring and debugging.
        
        In multi-endpoint mode, includes comprehensive per-endpoint breakdown,
        health status, connectivity information, and load balancing effectiveness metrics.
        
        Returns:
            Dictionary with comprehensive unified pool metrics
            
        Raises:
            RuntimeError: If the client is not started
        """
        if not self._is_started:
            raise RuntimeError("Client must be started before getting metrics")
        
        if not self._pool_manager or not self._unified_metrics:
            raise RuntimeError("Pool manager or unified metrics is not initialized")
        
        # Get base pool metrics from pool manager
        base_metrics = self._pool_manager.get_detailed_pool_metrics()
        
        # Gather additional information for unified metrics
        endpoints = None
        endpoint_health = None
        load_balancer_info = None
        monitor_status = None
        
        if self._is_multi_endpoint_mode:
            # Collect multi-endpoint specific information
            if self._endpoint_manager:
                endpoints = self._endpoint_manager.get_all_endpoints()
            
            if self._multi_gpu_monitor:
                endpoint_health = self._multi_gpu_monitor.get_endpoint_health_status()
                monitor_status = self._multi_gpu_monitor.get_monitor_status()
            
            if self._load_balancer:
                load_balancer_info = {
                    'strategy': self._load_balancer.get_strategy_name(),
                    'strategy_description': getattr(self._load_balancer, 'get_strategy_description', lambda: 'No description available')()
                }
                
                # Add endpoint weights if available
                if hasattr(self._load_balancer, 'get_endpoint_weights'):
                    load_balancer_info['endpoint_weights'] = self._load_balancer.get_endpoint_weights()
                
                # Add recent selection history if available
                if hasattr(self._load_balancer, 'get_selection_history'):
                    load_balancer_info['recent_selections'] = self._load_balancer.get_selection_history()
        
        # Create unified detailed metrics
        return self._unified_metrics.create_unified_detailed_metrics(
            base_metrics=base_metrics,
            endpoints=endpoints,
            endpoint_health=endpoint_health,
            load_balancer_info=load_balancer_info,
            monitor_status=monitor_status
        )
    
    def _detect_multi_endpoint_mode(self) -> bool:
        """Detect whether to use multi-endpoint or single-endpoint mode.
        
        Returns:
            True if multi-endpoint mode should be used, False for single-endpoint mode
        """
        # Check if explicit multi-endpoint configuration is provided
        if self.service_endpoints:
            return True
        
        # Check environment variables for multi-endpoint configuration
        import os
        env_endpoints = os.getenv('GPU_STATS_SERVICE_ENDPOINTS')
        if env_endpoints and ',' in env_endpoints:
            return True
        
        # Default to single-endpoint mode for backward compatibility
        return False
    
    async def _start_single_endpoint_mode(self) -> None:
        """Initialize components for single-endpoint mode."""
        # Initialize single-endpoint configuration
        self._config = EnvironmentConfigurationManager()
        
        # Apply parameter overrides if provided
        if self.service_endpoint:
            self._config._service_endpoint = self.service_endpoint
        if self.memory_threshold is not None:
            self._config._memory_threshold = self.memory_threshold
        if self.utilization_threshold is not None:
            self._config._utilization_threshold = self.utilization_threshold
        if self.polling_interval is not None:
            self._config._polling_interval = self.polling_interval
        
        # Initialize HTTP client
        self._http_client = AsyncGPUStatsHTTPClient(
            endpoint=self._config.get_service_endpoint(),
            timeout=self.request_timeout
        )
        
        # Initialize GPU monitor
        self._gpu_monitor = AsyncGPUMonitor(
            http_client=self._http_client,
            config=self._config
        )
        
        # Initialize GPU allocator
        self._gpu_allocator = ThresholdBasedGPUAllocator(self._config)
        
        # Initialize worker queue
        self._worker_queue = FIFOWorkerQueue()
        
        # Initialize worker pool manager
        self._pool_manager = AsyncWorkerPoolManager(
            config=self._config,
            gpu_monitor=self._gpu_monitor,
            gpu_allocator=self._gpu_allocator,
            worker_queue=self._worker_queue,
            worker_timeout=self.worker_timeout
        )
        
        # Start the pool manager (this will start all components)
        await self._pool_manager.start()
    
    async def _start_multi_endpoint_mode(self) -> None:
        """Initialize components for multi-endpoint mode."""
        # Initialize multi-endpoint configuration
        self._multi_config = EnvironmentMultiEndpointConfigurationManager()
        
        # Apply parameter overrides if provided
        if self.service_endpoints:
            self._multi_config._service_endpoints = self.service_endpoints.split(',')
        if self.memory_threshold is not None:
            self._multi_config._memory_threshold = self.memory_threshold
        if self.utilization_threshold is not None:
            self._multi_config._utilization_threshold = self.utilization_threshold
        if self.polling_interval is not None:
            self._multi_config._polling_interval = self.polling_interval
        if self.load_balancing_strategy:
            self._multi_config._load_balancing_strategy = self.load_balancing_strategy
        
        # Initialize endpoint manager
        self._endpoint_manager = AsyncEndpointManager(
            config=self._multi_config
        )
        
        # Initialize multi-endpoint HTTP client pool
        self._multi_http_client = AsyncMultiEndpointHTTPClientPool(
            endpoint_manager=self._endpoint_manager
        )
        
        # Initialize load balancer
        self._load_balancer = MultiEndpointLoadBalancer(
            endpoint_manager=self._endpoint_manager,
            strategy_name=self._multi_config.get_load_balancing_strategy()
        )
        
        # Initialize multi-endpoint GPU monitor
        self._multi_gpu_monitor = MultiEndpointGPUMonitor(
            endpoint_manager=self._endpoint_manager,
            http_client_pool=self._multi_http_client,
            config=self._multi_config
        )
        
        # Initialize GPU allocator (using multi-endpoint config)
        self._gpu_allocator = ThresholdBasedGPUAllocator(self._multi_config)
        
        # Initialize worker queue
        self._worker_queue = FIFOWorkerQueue()
        
        # Initialize worker pool manager with multi-endpoint monitor
        self._pool_manager = AsyncWorkerPoolManager(
            config=self._multi_config,
            gpu_monitor=self._multi_gpu_monitor,
            gpu_allocator=self._gpu_allocator,
            worker_queue=self._worker_queue,
            worker_timeout=self.worker_timeout
        )
        
        # Start the endpoint manager and multi-endpoint components
        await self._endpoint_manager.start()
        await self._pool_manager.start()
    
    async def _cleanup(self) -> None:
        """Internal cleanup method."""
        if self._pool_manager:
            try:
                await self._pool_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping pool manager: {e}")
        
        if self._endpoint_manager:
            try:
                await self._endpoint_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping endpoint manager: {e}")
        
        if self._multi_http_client:
            try:
                await self._multi_http_client.close()
            except Exception as e:
                logger.error(f"Error closing multi-endpoint HTTP client: {e}")
        
        # Reset all components
        self._pool_manager = None
        self._worker_queue = None
        self._gpu_allocator = None
        self._gpu_monitor = None
        self._multi_gpu_monitor = None
        self._load_balancer = None
        self._endpoint_manager = None
        self._multi_http_client = None
        self._http_client = None
        self._multi_config = None
        self._config = None
        self._unified_metrics = None
    
    async def __aenter__(self) -> 'GPUWorkerPoolClient':
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
    
    def is_multi_endpoint_mode(self) -> bool:
        """
        Check if the client is operating in multi-endpoint mode.
        
        Returns:
            True if in multi-endpoint mode, False if in single-endpoint mode
        """
        return self._is_multi_endpoint_mode
    
    def get_endpoints_info(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get information about all configured endpoints.
        
        This method is only available in multi-endpoint mode.
        
        Returns:
            List of endpoint information dictionaries, or None if not in multi-endpoint mode
        """
        if not self._is_multi_endpoint_mode or not self._endpoint_manager:
            return None
        
        return [
            {
                'endpoint_id': ep.endpoint_id,
                'url': ep.url,
                'is_healthy': ep.is_healthy,
                'total_gpus': ep.total_gpus,
                'available_gpus': ep.available_gpus,
                'response_time_ms': ep.response_time_ms,
                'last_seen': ep.last_seen.isoformat()
            }
            for ep in self._endpoint_manager.get_all_endpoints()
        ]
    
    def get_error_recovery_status(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive error recovery and degradation status.
        
        This method is only available in multi-endpoint mode and provides
        detailed information about circuit breaker states, degradation status,
        and recovery operations.
        
        Returns:
            Dictionary with error recovery status, or None if not in multi-endpoint mode
            
        Raises:
            RuntimeError: If the client is not started
        """
        if not self._is_started:
            raise RuntimeError("Client must be started before getting error recovery status")
        
        if not self._is_multi_endpoint_mode or not self._endpoint_manager:
            return None
        
        # Get degradation status from endpoint manager
        return self._endpoint_manager.get_degradation_status()
    
    async def trigger_endpoint_recovery(self, endpoint_id: str) -> bool:
        """
        Manually trigger recovery attempt for a specific endpoint.
        
        This method is only available in multi-endpoint mode and allows
        manual intervention to attempt recovery of a failed endpoint.
        
        Args:
            endpoint_id: ID of the endpoint to recover
            
        Returns:
            True if recovery was triggered successfully, False otherwise
            
        Raises:
            RuntimeError: If the client is not started or not in multi-endpoint mode
        """
        if not self._is_started:
            raise RuntimeError("Client must be started before triggering recovery")
        
        if not self._is_multi_endpoint_mode or not self._endpoint_manager:
            raise RuntimeError("Endpoint recovery is only available in multi-endpoint mode")
        
        try:
            await self._endpoint_manager.recovery_orchestrator.trigger_recovery_attempt(
                endpoint_id,
                lambda: self._endpoint_manager._attempt_endpoint_recovery(endpoint_id)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to trigger recovery for endpoint {endpoint_id}: {e}")
            return False
    
    async def queue_request_for_retry(self, request_func: Callable) -> None:
        """
        Queue a request for retry when system is in degraded state.
        
        This method is only available in multi-endpoint mode and allows
        queueing requests that can be retried when endpoints recover.
        
        Args:
            request_func: Async function to execute when endpoints recover
            
        Raises:
            RuntimeError: If the client is not started or not in multi-endpoint mode
        """
        if not self._is_started:
            raise RuntimeError("Client must be started before queueing requests")
        
        if not self._is_multi_endpoint_mode or not self._endpoint_manager:
            raise RuntimeError("Request queueing is only available in multi-endpoint mode")
        
        await self._endpoint_manager.queue_request_when_degraded(request_func)
    
    def print_error_recovery_summary(self) -> None:
        """
        Print a summary of error recovery status to the console.
        
        This is a convenience method for quick error recovery status checks.
        
        Raises:
            RuntimeError: If the client is not started
        """
        if not self._is_started:
            raise RuntimeError("Client must be started before printing recovery status")
        
        if not self._is_multi_endpoint_mode:
            print("\n=== Error Recovery Status ===")
            print("Mode: Single-endpoint (error recovery not applicable)")
            print()
            return
        
        try:
            recovery_status = self.get_error_recovery_status()
            if not recovery_status:
                print("\n=== Error Recovery Status ===")
                print("Error recovery status not available")
                print()
                return
            
            degradation_info = recovery_status.get('degradation_manager', {})
            recovery_info = recovery_status.get('recovery_orchestrator', {})
            health_summary = recovery_status.get('endpoint_health_summary', {})
            
            print("\n=== Error Recovery Status ===")
            print(f"System Status: {'DEGRADED' if degradation_info.get('is_fully_degraded', False) else 'OPERATIONAL'}")
            print(f"Total Endpoints: {health_summary.get('total_endpoints', 0)}")
            print(f"Healthy Endpoints: {health_summary.get('healthy_count', 0)}")
            print(f"Degraded Endpoints: {health_summary.get('degraded_count', 0)}")
            print(f"Queued Requests: {health_summary.get('queued_requests', 0)}")
            print(f"Recovery Operations Active: {len(recovery_info.get('active_recovery_tasks', {}))}")
            
            # Show circuit breaker states
            circuit_stats = degradation_info.get('circuit_breaker_stats', {})
            if circuit_stats:
                print("\n--- Circuit Breaker Status ---")
                for endpoint_id, stats in circuit_stats.items():
                    state = stats.get('state', 'unknown').upper()
                    failure_rate = stats.get('failure_rate', 0.0)
                    print(f"{endpoint_id}: {state} (Failure Rate: {failure_rate:.1f}%)")
            
            print()
            
        except Exception as e:
            print(f"Error getting recovery status: {e}")
    
    def get_formatted_metrics(self, format_type: str = 'console') -> str:
        """
        Get formatted metrics output for display.
        
        Args:
            format_type: Output format ('console' for human-readable, 'json' for JSON)
            
        Returns:
            Formatted metrics string
            
        Raises:
            RuntimeError: If the client is not started
            ValueError: If format_type is not supported
        """
        if format_type not in ['console', 'json']:
            raise ValueError(f"Unsupported format type: {format_type}. Use 'console' or 'json'.")
        
        metrics = self.get_detailed_metrics()
        
        if format_type == 'console':
            return MetricsFormatter.format_for_console(metrics)
        elif format_type == 'json':
            import json
            return json.dumps(MetricsFormatter.format_for_json(metrics), indent=2)
    
    def get_health_summary(self) -> Optional[str]:
        """
        Get a formatted health summary for all endpoints.
        
        This method is only available in multi-endpoint mode.
        
        Returns:
            Formatted health summary string, or None if not in multi-endpoint mode
            
        Raises:
            RuntimeError: If the client is not started
        """
        if not self._is_started:
            raise RuntimeError("Client must be started before getting health summary")
        
        if not self._is_multi_endpoint_mode or not self._multi_gpu_monitor:
            return None
        
        endpoint_health = self._multi_gpu_monitor.get_endpoint_health_status()
        return MetricsFormatter.format_health_summary(endpoint_health)
    
    def print_status_summary(self) -> None:
        """
        Print a concise status summary to the console.
        
        This is a convenience method for quick status checks.
        
        Raises:
            RuntimeError: If the client is not started
        """
        if not self._is_started:
            raise RuntimeError("Client must be started before printing status")
        
        try:
            pool_status = self.get_pool_status()
            
            print("\n=== GPU Worker Pool Status ===")
            print(f"Mode: {'Multi-endpoint' if self._is_multi_endpoint_mode else 'Single-endpoint'}")
            
            if isinstance(pool_status, MultiEndpointPoolStatus):
                print(f"Endpoints: {pool_status.healthy_endpoints}/{pool_status.total_endpoints} healthy")
                print(f"Total GPUs: {pool_status.total_gpus}")
                print(f"Available GPUs: {pool_status.available_gpus}")
                print(f"Utilization: {((pool_status.total_gpus - pool_status.available_gpus) / pool_status.total_gpus * 100) if pool_status.total_gpus > 0 else 0:.1f}%")
            else:
                print(f"Total GPUs: {pool_status.total_gpus}")
                print(f"Available GPUs: {pool_status.available_gpus}")
                print(f"Utilization: {((pool_status.total_gpus - pool_status.available_gpus) / pool_status.total_gpus * 100) if pool_status.total_gpus > 0 else 0:.1f}%")
            
            print(f"Active Workers: {pool_status.active_workers}")
            print(f"Blocked Workers: {pool_status.blocked_workers}")
            print("\n")
            
        except Exception as e:
            print(f"Error getting status: {e}")


@asynccontextmanager
async def gpu_worker_pool_client(**kwargs) -> AsyncContextManager[GPUWorkerPoolClient]:
    """
    Async context manager factory for GPUWorkerPoolClient.
    
    This is a convenience function that creates and manages a GPUWorkerPoolClient
    instance with automatic startup and cleanup.
    
    Args:
        **kwargs: Arguments passed to GPUWorkerPoolClient constructor
        
    Yields:
        GPUWorkerPoolClient: Started client instance
        
    Example:
        async with gpu_worker_pool_client() as client:
            assignment = await client.request_gpu()
            # Use the GPU...
            await client.release_gpu(assignment)
    """
    client = GPUWorkerPoolClient(**kwargs)
    async with client:
        yield client


class GPUContextManager:
    """
    Context manager for automatic GPU assignment and release.
    
    This class provides a convenient way to request and automatically release
    GPU assignments using Python's context manager protocol. Supports both
    local GPU IDs (single-endpoint mode) and global GPU IDs (multi-endpoint mode).
    
    Example:
        # Single-endpoint mode
        async with GPUWorkerPoolClient() as client:
            async with GPUContextManager(client) as gpu_id:
                # gpu_id is a local integer (e.g., 0, 1, 2)
                print(f"Using GPU {gpu_id}")
        
        # Multi-endpoint mode
        async with GPUWorkerPoolClient() as client:
            async with GPUContextManager(client) as gpu_id:
                # gpu_id is a global identifier (e.g., "server1:0", "server2:1")
                print(f"Using global GPU {gpu_id}")
    """
    
    def __init__(self, client: GPUWorkerPoolClient, timeout: Optional[float] = None):
        """
        Initialize the GPU context manager.
        
        Args:
            client: Started GPUWorkerPoolClient instance
            timeout: Optional timeout for GPU request
        """
        self.client = client
        self.timeout = timeout
        self._assignment: Optional[GPUAssignment] = None
    
    async def __aenter__(self) -> str:
        """
        Request GPU assignment on context entry.
        
        Returns:
            str: The assigned GPU ID. In single-endpoint mode, this will be
                 a string representation of an integer (e.g., "0"). In
                 multi-endpoint mode, this will be a global identifier
                 (e.g., "server1:0").
        """
        self._assignment = await self.client.request_gpu(timeout=self.timeout)
        return str(self._assignment.gpu_id)
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release GPU assignment on context exit."""
        if self._assignment:
            await self.client.release_gpu(self._assignment)
            self._assignment = None