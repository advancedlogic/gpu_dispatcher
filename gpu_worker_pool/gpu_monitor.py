"""GPU monitoring system with polling and callback support."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Dict, Any
from datetime import datetime, timedelta
import aiohttp

from .models import GPUStats
from .http_client import GPUStatsHTTPClient, AsyncGPUStatsHTTPClient, ServiceUnavailableError, RetryableError
from .config import ConfigurationManager

logger = logging.getLogger(__name__)


class GPUMonitor(ABC):
    """Abstract base class for GPU monitoring."""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the GPU monitoring system."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the GPU monitoring system."""
        pass
    
    @abstractmethod
    def get_current_stats(self) -> Optional[GPUStats]:
        """Get the most recent GPU statistics."""
        pass
    
    @abstractmethod
    def on_stats_update(self, callback: Callable[[GPUStats], None]) -> None:
        """Register a callback for GPU stats updates."""
        pass
    
    @abstractmethod
    def remove_stats_callback(self, callback: Callable[[GPUStats], None]) -> None:
        """Remove a previously registered callback."""
        pass


class AsyncGPUMonitor(GPUMonitor):
    """Async GPU monitor with polling and exponential backoff."""
    
    def __init__(self, 
                 http_client: GPUStatsHTTPClient,
                 config: ConfigurationManager,
                 max_retry_delay: float = 60.0,
                 backoff_multiplier: float = 2.0):
        """Initialize the GPU monitor.
        
        Args:
            http_client: HTTP client for fetching GPU stats
            config: Configuration manager for polling settings
            max_retry_delay: Maximum delay between retries in seconds
            backoff_multiplier: Multiplier for exponential backoff
        """
        self.http_client = http_client
        self.config = config
        self.max_retry_delay = max_retry_delay
        self.backoff_multiplier = backoff_multiplier
        
        self._current_stats: Optional[GPUStats] = None
        self._callbacks: List[Callable[[GPUStats], None]] = []
        self._polling_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._stats_lock = asyncio.Lock()
        
        # Retry state
        self._consecutive_failures = 0
        self._last_success_time: Optional[datetime] = None
        self._last_failure_time: Optional[datetime] = None
    
    async def start(self) -> None:
        """Start the GPU monitoring polling loop."""
        if self._is_running:
            logger.warning("GPU monitor is already running")
            return
        
        self._is_running = True
        self._consecutive_failures = 0
        self._polling_task = asyncio.create_task(self._polling_loop())
        logger.info(f"GPU monitor started with {self.config.get_polling_interval()}s interval")
    
    async def stop(self) -> None:
        """Stop the GPU monitoring polling loop."""
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
        
        await self.http_client.close()
        logger.info("GPU monitor stopped")
    
    def get_current_stats(self) -> Optional[GPUStats]:
        """Get the most recent GPU statistics."""
        return self._current_stats
    
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
                # Fetch GPU statistics
                stats = await self.http_client.fetch_gpu_stats()
                
                if stats is not None:
                    await self._handle_successful_fetch(stats)
                else:
                    await self._handle_failed_fetch("Received None stats from HTTP client")
                
                # Wait for next polling interval
                await asyncio.sleep(self.config.get_polling_interval())
                
            except asyncio.CancelledError:
                logger.debug("Polling loop cancelled")
                break
                
            except ServiceUnavailableError as e:
                # Service is unavailable - implement graceful degradation
                await self._handle_service_unavailable(str(e))
                
                # Use longer delay for service unavailable
                retry_delay = min(self.config.get_polling_interval() * 3, self.max_retry_delay)
                logger.info(f"Service unavailable, retrying in {retry_delay:.1f} seconds")
                
                try:
                    await asyncio.sleep(retry_delay)
                except asyncio.CancelledError:
                    break
                    
            except RetryableError as e:
                # Retryable error - use exponential backoff
                await self._handle_failed_fetch(f"Retryable error: {e}")
                
                retry_delay = self._calculate_retry_delay()
                logger.info(f"Retrying in {retry_delay:.1f} seconds")
                
                try:
                    await asyncio.sleep(retry_delay)
                except asyncio.CancelledError:
                    break
                    
            except ValueError as e:
                # Invalid response format - not retryable, but continue polling
                await self._handle_failed_fetch(f"Invalid response format: {e}")
                logger.warning("Continuing with previous GPU stats due to invalid response format")
                
                # Use normal polling interval for format errors
                await asyncio.sleep(self.config.get_polling_interval())
                
            except Exception as e:
                # Unexpected error
                await self._handle_failed_fetch(f"Unexpected error: {e}")
                logger.error(f"Unexpected error in polling loop: {e}", exc_info=True)
                
                # Calculate retry delay with exponential backoff
                retry_delay = self._calculate_retry_delay()
                logger.info(f"Retrying in {retry_delay:.1f} seconds")
                
                try:
                    await asyncio.sleep(retry_delay)
                except asyncio.CancelledError:
                    break
    
    async def _handle_successful_fetch(self, stats: GPUStats) -> None:
        """Handle successful GPU stats fetch."""
        async with self._stats_lock:
            self._current_stats = stats
            self._last_success_time = datetime.now()
            
            # Reset failure count on success
            if self._consecutive_failures > 0:
                logger.info(f"GPU stats service recovered after {self._consecutive_failures} failures")
                self._consecutive_failures = 0
        
        # Notify callbacks
        await self._notify_callbacks(stats)
        
        logger.debug(f"Updated GPU stats: {stats.gpu_count} GPUs, "
                    f"avg utilization: {stats.average_utilization_percent:.1f}%")
    
    async def _handle_failed_fetch(self, error_message: str) -> None:
        """Handle failed GPU stats fetch."""
        self._consecutive_failures += 1
        self._last_failure_time = datetime.now()
        
        logger.error(f"Failed to fetch GPU stats (attempt {self._consecutive_failures}): {error_message}")
        
        # Log additional context for persistent failures
        if self._consecutive_failures >= 3:
            time_since_success = "never" if self._last_success_time is None else \
                str(datetime.now() - self._last_success_time)
            logger.warning(f"GPU stats service has been failing for {time_since_success}")
    
    async def _handle_service_unavailable(self, error_message: str) -> None:
        """Handle service unavailable with graceful degradation."""
        self._consecutive_failures += 1
        self._last_failure_time = datetime.now()
        
        logger.warning(f"GPU service unavailable (attempt {self._consecutive_failures}): {error_message}")
        
        # Implement graceful degradation - continue with last known stats
        if self._current_stats is not None:
            time_since_success = "never" if self._last_success_time is None else \
                str(datetime.now() - self._last_success_time)
            logger.info(f"Continuing with last known GPU stats from {time_since_success} ago")
        else:
            logger.warning("No previous GPU stats available for graceful degradation")
        
        # Log service health status
        if self._consecutive_failures >= 5:
            logger.error(
                f"GPU service has been unavailable for {self._consecutive_failures} consecutive attempts. "
                f"System is operating in degraded mode."
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
        """Get current monitor status for debugging."""
        return {
            "is_running": self._is_running,
            "consecutive_failures": self._consecutive_failures,
            "last_success_time": self._last_success_time.isoformat() if self._last_success_time else None,
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "has_current_stats": self._current_stats is not None,
            "callback_count": len(self._callbacks),
            "polling_interval": self.config.get_polling_interval()
        }


class MockGPUMonitor(GPUMonitor):
    """Mock GPU monitor for testing purposes."""
    
    def __init__(self, mock_stats: Optional[GPUStats] = None):
        """Initialize mock monitor.
        
        Args:
            mock_stats: Mock GPU stats to return
        """
        self.mock_stats = mock_stats
        self._callbacks: List[Callable[[GPUStats], None]] = []
        self._is_running = False
        self.start_call_count = 0
        self.stop_call_count = 0
    
    async def start(self) -> None:
        """Mock start method."""
        self._is_running = True
        self.start_call_count += 1
        
        # Trigger callbacks if we have mock stats
        if self.mock_stats:
            await self._notify_callbacks(self.mock_stats)
    
    async def stop(self) -> None:
        """Mock stop method."""
        self._is_running = False
        self.stop_call_count += 1
    
    def get_current_stats(self) -> Optional[GPUStats]:
        """Get mock GPU statistics."""
        return self.mock_stats
    
    def on_stats_update(self, callback: Callable[[GPUStats], None]) -> None:
        """Register mock callback."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def remove_stats_callback(self, callback: Callable[[GPUStats], None]) -> None:
        """Remove mock callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def trigger_stats_update(self, stats: GPUStats) -> None:
        """Manually trigger stats update for testing."""
        self.mock_stats = stats
        await self._notify_callbacks(stats)
    
    async def _notify_callbacks(self, stats: GPUStats) -> None:
        """Notify callbacks in mock monitor."""
        for callback in self._callbacks.copy():
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(stats)
                else:
                    callback(stats)
            except Exception:
                pass  # Ignore errors in mock