"""Multi-endpoint GPU monitoring system with aggregated statistics and health tracking."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Dict, Any
from datetime import datetime, timedelta

from .models import GPUStats, GlobalGPUInfo, EndpointInfo, MultiEndpointPoolStatus
from .gpu_monitor import GPUMonitor
from .endpoint_manager import EndpointManager
from .multi_endpoint_http_client import MultiEndpointHTTPClientPool
from .config import MultiEndpointConfigurationManager

logger = logging.getLogger(__name__)


class MultiEndpointGPUMonitor(GPUMonitor):
    """GPU monitor that aggregates statistics from multiple endpoints with health tracking."""
    
    def __init__(self, 
                 endpoint_manager: EndpointManager,
                 http_client_pool: MultiEndpointHTTPClientPool,
                 config: MultiEndpointConfigurationManager,
                 max_retry_delay: float = 60.0,
                 backoff_multiplier: float = 2.0):
        """Initialize the multi-endpoint GPU monitor.
        
        Args:
            endpoint_manager: Manager for endpoint health and connections
            http_client_pool: HTTP client pool for fetching from multiple endpoints
            config: Configuration manager for polling settings
            max_retry_delay: Maximum delay between retries in seconds
            backoff_multiplier: Multiplier for exponential backoff
        """
        self.endpoint_manager = endpoint_manager
        self.http_client_pool = http_client_pool
        self.config = config
        self.max_retry_delay = max_retry_delay
        self.backoff_multiplier = backoff_multiplier
        
        # Current aggregated state
        self._current_global_gpus: Optional[List[GlobalGPUInfo]] = None
        self._current_aggregated_stats: Optional[GPUStats] = None
        self._endpoint_health_status: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks and monitoring state
        self._callbacks: List[Callable[[GPUStats], None]] = []
        self._polling_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._stats_lock = asyncio.Lock()
        
        # Retry and failure tracking
        self._consecutive_failures = 0
        self._last_success_time: Optional[datetime] = None
        self._last_failure_time: Optional[datetime] = None
        self._endpoint_last_success: Dict[str, datetime] = {}
        self._endpoint_consecutive_failures: Dict[str, int] = {}
    
    async def start(self) -> None:
        """Start the multi-endpoint GPU monitoring polling loop."""
        if self._is_running:
            logger.warning("Multi-endpoint GPU monitor is already running")
            return
        
        self._is_running = True
        self._consecutive_failures = 0
        self._polling_task = asyncio.create_task(self._polling_loop())
        logger.info(f"Multi-endpoint GPU monitor started with {self.config.get_polling_interval()}s interval")
    
    async def stop(self) -> None:
        """Stop the multi-endpoint GPU monitoring polling loop."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
        
        await self.http_client_pool.close()
        logger.info("Multi-endpoint GPU monitor stopped")
    
    def get_current_stats(self) -> Optional[GPUStats]:
        """Get the most recent aggregated GPU statistics."""
        return self._current_aggregated_stats
    
    def get_current_global_gpus(self) -> Optional[List[GlobalGPUInfo]]:
        """Get the most recent global GPU information from all endpoints."""
        return self._current_global_gpus
    
    def get_endpoint_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed health status for all endpoints."""
        return self._endpoint_health_status.copy()
    
    def on_stats_update(self, callback: Callable[[GPUStats], None]) -> None:
        """Register a callback for GPU stats updates."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug(f"Registered stats update callback: {callback.__name__}")
    
    def remove_stats_callback(self, callback: Callable[[GPUStats], None]) -> None:
        """Remove a previously registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug(f"Removed stats update callback: {callback.__name__}")
    
    async def _polling_loop(self) -> None:
        """Main polling loop with graceful degradation and error handling."""
        while self._is_running:
            try:
                # Fetch aggregated GPU statistics from all healthy endpoints
                global_gpus = await self.http_client_pool.fetch_aggregated_gpu_stats()
                
                if global_gpus is not None and len(global_gpus) > 0:
                    await self._handle_successful_fetch(global_gpus)
                else:
                    await self._handle_failed_fetch("No GPU data available from any endpoint")
                
                # Update endpoint health status
                await self._update_endpoint_health_status()
                
                # Wait for next polling interval
                await asyncio.sleep(self.config.get_polling_interval())
                
            except asyncio.CancelledError:
                logger.debug("Multi-endpoint polling loop cancelled")
                break
                
            except Exception as e:
                # Handle any unexpected errors
                await self._handle_failed_fetch(f"Unexpected error: {e}")
                logger.error(f"Unexpected error in multi-endpoint polling loop: {e}", exc_info=True)
                
                # Calculate retry delay with exponential backoff
                retry_delay = self._calculate_retry_delay()
                logger.info(f"Retrying in {retry_delay:.1f} seconds")
                
                try:
                    await asyncio.sleep(retry_delay)
                except asyncio.CancelledError:
                    break
    
    async def _handle_successful_fetch(self, global_gpus: List[GlobalGPUInfo]) -> None:
        """Handle successful GPU stats fetch from multiple endpoints."""
        async with self._stats_lock:
            self._current_global_gpus = global_gpus
            
            # Create aggregated GPUStats from global GPU information
            self._current_aggregated_stats = self._create_aggregated_stats(global_gpus)
            
            self._last_success_time = datetime.now()
            
            # Reset failure count on success
            if self._consecutive_failures > 0:
                logger.info(f"Multi-endpoint GPU stats service recovered after {self._consecutive_failures} failures")
                self._consecutive_failures = 0
        
        # Notify callbacks with aggregated stats
        if self._current_aggregated_stats:
            await self._notify_callbacks(self._current_aggregated_stats)
        
        logger.debug(f"Updated aggregated GPU stats: {len(global_gpus)} GPUs from multiple endpoints, "
                    f"avg utilization: {self._current_aggregated_stats.average_utilization_percent:.1f}%")
    
    async def _handle_failed_fetch(self, error_message: str) -> None:
        """Handle failed GPU stats fetch from endpoints."""
        self._consecutive_failures += 1
        self._last_failure_time = datetime.now()
        
        logger.error(f"Failed to fetch aggregated GPU stats (attempt {self._consecutive_failures}): {error_message}")
        
        # Log additional context for persistent failures
        if self._consecutive_failures >= 3:
            time_since_success = "never" if self._last_success_time is None else \
                str(datetime.now() - self._last_success_time)
            logger.warning(f"Multi-endpoint GPU stats service has been failing for {time_since_success}")
            
            # Check if any endpoints are still healthy
            healthy_endpoints = self.endpoint_manager.get_healthy_endpoints()
            if not healthy_endpoints:
                logger.error("No healthy endpoints available - system is in degraded mode")
            else:
                logger.info(f"Still have {len(healthy_endpoints)} healthy endpoints available")
    
    async def _update_endpoint_health_status(self) -> None:
        """Update detailed health status for all endpoints."""
        all_endpoints = self.endpoint_manager.get_all_endpoints()
        current_time = datetime.now()
        
        for endpoint in all_endpoints:
            endpoint_id = endpoint.endpoint_id
            
            # Track per-endpoint success/failure and connectivity
            if endpoint.is_healthy:
                if endpoint_id not in self._endpoint_last_success or \
                   self._endpoint_last_success[endpoint_id] < endpoint.last_seen:
                    self._endpoint_last_success[endpoint_id] = endpoint.last_seen
                    self._endpoint_consecutive_failures[endpoint_id] = 0
            else:
                self._endpoint_consecutive_failures[endpoint_id] = \
                    self._endpoint_consecutive_failures.get(endpoint_id, 0) + 1
            
            # Calculate connectivity metrics
            time_since_last_seen = current_time - endpoint.last_seen
            time_since_last_success = (
                current_time - self._endpoint_last_success[endpoint_id]
                if endpoint_id in self._endpoint_last_success else None
            )
            
            # Determine connectivity status
            connectivity_status = self._determine_connectivity_status(
                endpoint.is_healthy, 
                time_since_last_seen, 
                time_since_last_success,
                self._endpoint_consecutive_failures.get(endpoint_id, 0)
            )
            
            # Update comprehensive health status information
            self._endpoint_health_status[endpoint_id] = {
                "endpoint_id": endpoint_id,
                "url": endpoint.url,
                "is_healthy": endpoint.is_healthy,
                "connectivity_status": connectivity_status,
                "last_seen": endpoint.last_seen.isoformat(),
                "last_seen_seconds_ago": int(time_since_last_seen.total_seconds()),
                "total_gpus": endpoint.total_gpus,
                "available_gpus": endpoint.available_gpus,
                "gpu_utilization_rate": (
                    (endpoint.total_gpus - endpoint.available_gpus) / endpoint.total_gpus * 100
                    if endpoint.total_gpus > 0 else 0.0
                ),
                "response_time_ms": endpoint.response_time_ms,
                "consecutive_failures": self._endpoint_consecutive_failures.get(endpoint_id, 0),
                "last_successful_communication": (
                    self._endpoint_last_success[endpoint_id].isoformat() 
                    if endpoint_id in self._endpoint_last_success else None
                ),
                "last_successful_communication_seconds_ago": (
                    int(time_since_last_success.total_seconds())
                    if time_since_last_success else None
                ),
                "time_since_last_success": (
                    str(time_since_last_success).split('.')[0]  # Remove microseconds
                    if time_since_last_success else "never"
                ),
                "health_score": self._calculate_endpoint_health_score(
                    endpoint.is_healthy,
                    endpoint.response_time_ms,
                    self._endpoint_consecutive_failures.get(endpoint_id, 0),
                    time_since_last_success
                )
            }
    
    def _determine_connectivity_status(self, is_healthy: bool, time_since_last_seen: timedelta, 
                                     time_since_last_success: Optional[timedelta], 
                                     consecutive_failures: int) -> str:
        """Determine the connectivity status of an endpoint.
        
        Args:
            is_healthy: Whether endpoint is currently healthy
            time_since_last_seen: Time since last communication attempt
            time_since_last_success: Time since last successful communication
            consecutive_failures: Number of consecutive failures
            
        Returns:
            String describing connectivity status
        """
        if is_healthy:
            return "healthy"
        
        # Check if endpoint has never been successfully contacted
        if time_since_last_success is None:
            return "unreachable"
        
        # Check for recent failures
        if time_since_last_seen.total_seconds() < 60:  # Less than 1 minute
            if consecutive_failures < 3:
                return "intermittent"
            else:
                return "failing"
        
        # Check for longer outages
        if time_since_last_success.total_seconds() < 300:  # Less than 5 minutes
            return "temporarily_unavailable"
        elif time_since_last_success.total_seconds() < 1800:  # Less than 30 minutes
            return "degraded"
        else:
            return "offline"
    
    def _calculate_endpoint_health_score(self, is_healthy: bool, response_time_ms: float,
                                       consecutive_failures: int, 
                                       time_since_last_success: Optional[timedelta]) -> float:
        """Calculate a health score (0-100) for an endpoint.
        
        Args:
            is_healthy: Whether endpoint is currently healthy
            response_time_ms: Current response time in milliseconds
            consecutive_failures: Number of consecutive failures
            time_since_last_success: Time since last successful communication
            
        Returns:
            Health score from 0 (completely unhealthy) to 100 (perfect health)
        """
        if not is_healthy and time_since_last_success is None:
            return 0.0  # Never been reachable
        
        score = 100.0
        
        # Deduct points for being unhealthy
        if not is_healthy:
            score -= 50.0
        
        # Deduct points for slow response times
        if response_time_ms > 1000:  # > 1 second
            score -= min(30.0, (response_time_ms - 1000) / 100)
        elif response_time_ms > 500:  # > 500ms
            score -= min(15.0, (response_time_ms - 500) / 50)
        
        # Deduct points for consecutive failures
        if consecutive_failures > 0:
            score -= min(25.0, consecutive_failures * 5)
        
        # Deduct points for time since last success
        if time_since_last_success:
            minutes_since_success = time_since_last_success.total_seconds() / 60
            if minutes_since_success > 5:
                score -= min(20.0, (minutes_since_success - 5) / 2)
        
        return max(0.0, score)
    
    def _create_aggregated_stats(self, global_gpus: List[GlobalGPUInfo]) -> GPUStats:
        """Create aggregated GPUStats from global GPU information.
        
        Args:
            global_gpus: List of global GPU information from all endpoints
            
        Returns:
            Aggregated GPUStats object
        """
        if not global_gpus:
            # Return empty stats if no GPUs available
            from .models import GPUInfo
            return GPUStats(
                gpu_count=0,
                total_memory_mb=0,
                total_used_memory_mb=0,
                average_utilization_percent=0.0,
                gpus_summary=[],
                total_memory_usage_percent=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        # Convert GlobalGPUInfo to GPUInfo for the summary
        from .models import GPUInfo
        gpus_summary = []
        
        for global_gpu in global_gpus:
            gpu_info = GPUInfo(
                gpu_id=len(gpus_summary),  # Use sequential IDs for aggregated view
                name=f"{global_gpu.endpoint_id}:{global_gpu.name}",  # Include endpoint in name
                memory_usage_percent=global_gpu.memory_usage_percent,
                utilization_percent=global_gpu.utilization_percent
            )
            gpus_summary.append(gpu_info)
        
        # Calculate aggregated statistics
        gpu_count = len(global_gpus)
        
        # Estimate memory (8GB per GPU as default)
        estimated_memory_per_gpu = 8192  # MB
        total_memory_mb = gpu_count * estimated_memory_per_gpu
        
        # Calculate total used memory based on usage percentages
        total_used_memory_mb = sum(
            int(gpu.memory_usage_percent / 100 * estimated_memory_per_gpu)
            for gpu in global_gpus
        )
        
        # Calculate average utilization
        average_utilization_percent = (
            sum(gpu.utilization_percent for gpu in global_gpus) / gpu_count
            if gpu_count > 0 else 0.0
        )
        
        # Calculate total memory usage percentage
        total_memory_usage_percent = (
            (total_used_memory_mb / total_memory_mb * 100)
            if total_memory_mb > 0 else 0.0
        )
        
        return GPUStats(
            gpu_count=gpu_count,
            total_memory_mb=total_memory_mb,
            total_used_memory_mb=total_used_memory_mb,
            average_utilization_percent=average_utilization_percent,
            gpus_summary=gpus_summary,
            total_memory_usage_percent=total_memory_usage_percent,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_retry_delay(self) -> float:
        """Calculate retry delay using exponential backoff."""
        if self._consecutive_failures <= 1:
            return self.config.get_polling_interval()
        
        # Exponential backoff: base_delay * multiplier^(failures-1)
        base_delay = self.config.get_polling_interval()
        delay = base_delay * (self.backoff_multiplier ** (self._consecutive_failures - 1))
        
        # Cap at maximum retry delay
        return min(delay, self.max_retry_delay)
    
    async def _notify_callbacks(self, stats: GPUStats) -> None:
        """Notify all registered callbacks of stats update."""
        if not self._callbacks:
            return
        
        # Run callbacks concurrently but handle errors individually
        tasks = []
        for callback in self._callbacks.copy():  # Copy to avoid modification during iteration
            task = asyncio.create_task(self._safe_callback_execution(callback, stats))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_callback_execution(self, callback: Callable[[GPUStats], None], stats: GPUStats) -> None:
        """Execute callback safely with error handling."""
        try:
            # Check if callback is async or sync
            if asyncio.iscoroutinefunction(callback):
                await callback(stats)
            else:
                callback(stats)
        except Exception as e:
            logger.error(f"Error in stats update callback {callback.__name__}: {e}")
    
    def get_monitor_status(self) -> Dict[str, Any]:
        """Get current monitor status for debugging and monitoring."""
        healthy_endpoints = self.endpoint_manager.get_healthy_endpoints()
        all_endpoints = self.endpoint_manager.get_all_endpoints()
        
        return {
            "is_running": self._is_running,
            "consecutive_failures": self._consecutive_failures,
            "last_success_time": self._last_success_time.isoformat() if self._last_success_time else None,
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "has_current_stats": self._current_aggregated_stats is not None,
            "has_global_gpu_data": self._current_global_gpus is not None,
            "total_global_gpus": len(self._current_global_gpus) if self._current_global_gpus else 0,
            "callback_count": len(self._callbacks),
            "polling_interval": self.config.get_polling_interval(),
            "endpoint_summary": {
                "total_endpoints": len(all_endpoints),
                "healthy_endpoints": len(healthy_endpoints),
                "unhealthy_endpoints": len(all_endpoints) - len(healthy_endpoints)
            },
            "endpoint_health_details": self._endpoint_health_status
        }
    
    def create_multi_endpoint_pool_status(self, active_workers: int = 0, blocked_workers: int = 0, 
                                        gpu_assignments: Optional[Dict[str, List]] = None) -> MultiEndpointPoolStatus:
        """Create a MultiEndpointPoolStatus object with current monitoring data.
        
        Args:
            active_workers: Number of active workers
            blocked_workers: Number of blocked workers
            gpu_assignments: Current GPU assignments by global GPU ID
            
        Returns:
            MultiEndpointPoolStatus object with aggregated information
        """
        all_endpoints = self.endpoint_manager.get_all_endpoints()
        healthy_endpoints = self.endpoint_manager.get_healthy_endpoints()
        
        total_gpus = sum(endpoint.total_gpus for endpoint in healthy_endpoints)
        available_gpus = sum(endpoint.available_gpus for endpoint in healthy_endpoints)
        
        return MultiEndpointPoolStatus(
            total_endpoints=len(all_endpoints),
            healthy_endpoints=len(healthy_endpoints),
            total_gpus=total_gpus,
            available_gpus=available_gpus,
            active_workers=active_workers,
            blocked_workers=blocked_workers,
            endpoints=all_endpoints,
            gpu_assignments=gpu_assignments or {}
        )


class MockMultiEndpointGPUMonitor(MultiEndpointGPUMonitor):
    """Mock multi-endpoint GPU monitor for testing purposes."""
    
    def __init__(self, mock_global_gpus: Optional[List[GlobalGPUInfo]] = None,
                 mock_endpoints: Optional[List[EndpointInfo]] = None):
        """Initialize mock monitor.
        
        Args:
            mock_global_gpus: Mock global GPU data to return
            mock_endpoints: Mock endpoint information
        """
        # Don't call super().__init__ to avoid dependency requirements
        self.mock_global_gpus = mock_global_gpus or []
        self.mock_endpoints = mock_endpoints or []
        self._callbacks: List[Callable[[GPUStats], None]] = []
        self._is_running = False
        self.start_call_count = 0
        self.stop_call_count = 0
        self._endpoint_health_status: Dict[str, Dict[str, Any]] = {}
        
        # Create mock aggregated stats if we have mock data
        if self.mock_global_gpus:
            self._current_aggregated_stats = self._create_aggregated_stats(self.mock_global_gpus)
            self._current_global_gpus = self.mock_global_gpus
        else:
            self._current_aggregated_stats = None
            self._current_global_gpus = None
    
    async def start(self) -> None:
        """Mock start method."""
        self._is_running = True
        self.start_call_count += 1
        
        # Trigger callbacks if we have mock stats
        if self._current_aggregated_stats:
            await self._notify_callbacks(self._current_aggregated_stats)
    
    async def stop(self) -> None:
        """Mock stop method."""
        self._is_running = False
        self.stop_call_count += 1
    
    def get_current_stats(self) -> Optional[GPUStats]:
        """Get mock aggregated GPU statistics."""
        return self._current_aggregated_stats
    
    def get_current_global_gpus(self) -> Optional[List[GlobalGPUInfo]]:
        """Get mock global GPU information."""
        return self._current_global_gpus
    
    def get_endpoint_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get mock endpoint health status."""
        return self._endpoint_health_status.copy()
    
    def on_stats_update(self, callback: Callable[[GPUStats], None]) -> None:
        """Register mock callback."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def remove_stats_callback(self, callback: Callable[[GPUStats], None]) -> None:
        """Remove mock callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def trigger_stats_update(self, global_gpus: List[GlobalGPUInfo]) -> None:
        """Manually trigger stats update for testing."""
        self.mock_global_gpus = global_gpus
        self._current_global_gpus = global_gpus
        self._current_aggregated_stats = self._create_aggregated_stats(global_gpus)
        
        if self._current_aggregated_stats:
            await self._notify_callbacks(self._current_aggregated_stats)
    
    def set_endpoint_health_status(self, endpoint_id: str, health_info: Dict[str, Any]) -> None:
        """Set mock endpoint health status for testing."""
        self._endpoint_health_status[endpoint_id] = health_info