"""Unit tests for GPU monitor with polling mechanism."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import aiohttp

from gpu_worker_pool.gpu_monitor import AsyncGPUMonitor, MockGPUMonitor
from gpu_worker_pool.http_client import MockGPUStatsHTTPClient
from gpu_worker_pool.config import EnvironmentConfigurationManager
from gpu_worker_pool.models import GPUStats, GPUInfo


class TestAsyncGPUMonitor:
    """Test cases for AsyncGPUMonitor."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock(spec=EnvironmentConfigurationManager)
        config.get_polling_interval.return_value = 1  # 1 second for faster tests
        return config
    
    @pytest.fixture
    def sample_gpu_stats(self):
        """Create sample GPU stats for testing."""
        return GPUStats(
            gpu_count=2,
            total_memory_mb=16384,
            total_used_memory_mb=8192,
            average_utilization_percent=45.5,
            total_memory_usage_percent=50.0,
            timestamp="2024-01-15T10:30:00Z",
            gpus_summary=[
                GPUInfo(gpu_id=0, name="GPU 0", memory_usage_percent=60.0, utilization_percent=75.0),
                GPUInfo(gpu_id=1, name="GPU 1", memory_usage_percent=40.0, utilization_percent=16.0)
            ]
        )
    
    @pytest.fixture
    def mock_http_client(self, sample_gpu_stats):
        """Create mock HTTP client."""
        return MockGPUStatsHTTPClient(mock_response={
            "gpu_count": sample_gpu_stats.gpu_count,
            "total_memory_mb": sample_gpu_stats.total_memory_mb,
            "total_used_memory_mb": sample_gpu_stats.total_used_memory_mb,
            "average_utilization_percent": sample_gpu_stats.average_utilization_percent,
            "total_memory_usage_percent": sample_gpu_stats.total_memory_usage_percent,
            "timestamp": sample_gpu_stats.timestamp,
            "gpus_summary": [
                {
                    "gpu_id": gpu.gpu_id,
                    "name": gpu.name,
                    "memory_usage_percent": gpu.memory_usage_percent,
                    "utilization_percent": gpu.utilization_percent
                }
                for gpu in sample_gpu_stats.gpus_summary
            ]
        })
    
    @pytest.fixture
    def monitor(self, mock_http_client, mock_config):
        """Create GPU monitor for testing."""
        return AsyncGPUMonitor(mock_http_client, mock_config)
    
    @pytest.mark.asyncio
    async def test_start_and_stop(self, monitor):
        """Test basic start and stop functionality."""
        assert not monitor._is_running
        
        await monitor.start()
        assert monitor._is_running
        assert monitor._polling_task is not None
        
        await monitor.stop()
        assert not monitor._is_running
        assert monitor._polling_task is None
    
    @pytest.mark.asyncio
    async def test_successful_stats_fetch(self, monitor, sample_gpu_stats):
        """Test successful GPU stats fetching and callback notification."""
        callback_called = asyncio.Event()
        received_stats = None
        
        def stats_callback(stats: GPUStats):
            nonlocal received_stats
            received_stats = stats
            callback_called.set()
        
        monitor.on_stats_update(stats_callback)
        
        await monitor.start()
        
        # Wait for at least one callback
        try:
            await asyncio.wait_for(callback_called.wait(), timeout=3.0)
        finally:
            await monitor.stop()
        
        assert received_stats is not None
        assert received_stats.gpu_count == sample_gpu_stats.gpu_count
        assert monitor.get_current_stats() is not None
    
    @pytest.mark.asyncio
    async def test_callback_management(self, monitor):
        """Test callback registration and removal."""
        def callback1(stats):
            pass
        
        def callback2(stats):
            pass
        
        # Register callbacks
        monitor.on_stats_update(callback1)
        monitor.on_stats_update(callback2)
        assert len(monitor._callbacks) == 2
        
        # Remove one callback
        monitor.remove_stats_callback(callback1)
        assert len(monitor._callbacks) == 1
        assert callback2 in monitor._callbacks
        
        # Remove non-existent callback (should not error)
        monitor.remove_stats_callback(callback1)
        assert len(monitor._callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_calculation(self, monitor):
        """Test exponential backoff retry delay calculation."""
        # First failure should use normal polling interval
        monitor._consecutive_failures = 1
        delay = monitor._calculate_retry_delay()
        assert delay == 1  # polling interval
        
        # Subsequent failures should use exponential backoff
        monitor._consecutive_failures = 2
        delay = monitor._calculate_retry_delay()
        assert delay == 2  # 1 * 2^1
        
        monitor._consecutive_failures = 3
        delay = monitor._calculate_retry_delay()
        assert delay == 4  # 1 * 2^2
        
        # Should cap at max retry delay
        monitor._consecutive_failures = 10
        delay = monitor._calculate_retry_delay()
        assert delay == monitor.max_retry_delay
    
    @pytest.mark.asyncio
    async def test_http_client_failure_handling(self, mock_config):
        """Test handling of HTTP client failures."""
        # Create failing HTTP client
        failing_client = MockGPUStatsHTTPClient(
            should_fail=True,
            failure_exception=aiohttp.ClientError("Connection failed")
        )
        
        monitor = AsyncGPUMonitor(failing_client, mock_config)
        
        await monitor.start()
        
        # Let it run for a short time to accumulate failures
        await asyncio.sleep(0.5)
        
        await monitor.stop()
        
        # Should have recorded failures
        assert monitor._consecutive_failures > 0
        assert monitor._last_failure_time is not None
        assert monitor.get_current_stats() is None
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, monitor, sample_gpu_stats):
        """Test that callback errors don't crash the monitor."""
        callback_called = asyncio.Event()
        
        def failing_callback(stats: GPUStats):
            raise ValueError("Callback error")
        
        def working_callback(stats: GPUStats):
            callback_called.set()
        
        monitor.on_stats_update(failing_callback)
        monitor.on_stats_update(working_callback)
        
        await monitor.start()
        
        # Working callback should still be called despite failing callback
        try:
            await asyncio.wait_for(callback_called.wait(), timeout=3.0)
        finally:
            await monitor.stop()
        
        assert callback_called.is_set()
    
    @pytest.mark.asyncio
    async def test_async_callback_support(self, monitor):
        """Test support for async callbacks."""
        async_callback_called = asyncio.Event()
        
        async def async_callback(stats: GPUStats):
            async_callback_called.set()
        
        monitor.on_stats_update(async_callback)
        
        await monitor.start()
        
        try:
            await asyncio.wait_for(async_callback_called.wait(), timeout=3.0)
        finally:
            await monitor.stop()
        
        assert async_callback_called.is_set()
    
    @pytest.mark.asyncio
    async def test_monitor_status(self, monitor):
        """Test monitor status reporting."""
        status = monitor.get_monitor_status()
        
        assert status["is_running"] is False
        assert status["consecutive_failures"] == 0
        assert status["callback_count"] == 0
        
        await monitor.start()
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        status = monitor.get_monitor_status()
        assert status["is_running"] is True
        
        await monitor.stop()
    
    @pytest.mark.asyncio
    async def test_double_start_warning(self, monitor):
        """Test that starting an already running monitor logs a warning."""
        await monitor.start()
        
        with patch('gpu_worker_pool.gpu_monitor.logger') as mock_logger:
            await monitor.start()  # Second start
            mock_logger.warning.assert_called_with("GPU monitor is already running")
        
        await monitor.stop()
    
    @pytest.mark.asyncio
    async def test_recovery_after_failures(self, mock_config, sample_gpu_stats):
        """Test recovery after consecutive failures."""
        # Create client that fails first, then succeeds
        call_count = 0
        
        async def mock_fetch():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise aiohttp.ClientError("Temporary failure")
            return sample_gpu_stats
        
        mock_client = AsyncMock()
        mock_client.fetch_gpu_stats = mock_fetch
        mock_client.close = AsyncMock()
        
        monitor = AsyncGPUMonitor(mock_client, mock_config)
        
        await monitor.start()
        
        # Wait for recovery - give more time for the exponential backoff
        for _ in range(10):  # Check multiple times
            await asyncio.sleep(0.5)
            if monitor.get_current_stats() is not None:
                break
        
        await monitor.stop()
        
        # Should have recovered
        assert monitor.get_current_stats() is not None
        assert monitor._consecutive_failures == 0


class TestMockGPUMonitor:
    """Test cases for MockGPUMonitor."""
    
    @pytest.fixture
    def sample_gpu_stats(self):
        """Create sample GPU stats for testing."""
        return GPUStats(
            gpu_count=1,
            total_memory_mb=8192,
            total_used_memory_mb=4096,
            average_utilization_percent=50.0,
            total_memory_usage_percent=50.0,
            timestamp="2024-01-15T10:30:00Z",
            gpus_summary=[
                GPUInfo(gpu_id=0, name="Mock GPU", memory_usage_percent=50.0, utilization_percent=50.0)
            ]
        )
    
    @pytest.mark.asyncio
    async def test_mock_basic_functionality(self, sample_gpu_stats):
        """Test basic mock monitor functionality."""
        monitor = MockGPUMonitor(sample_gpu_stats)
        
        assert monitor.get_current_stats() == sample_gpu_stats
        assert not monitor._is_running
        
        await monitor.start()
        assert monitor._is_running
        assert monitor.start_call_count == 1
        
        await monitor.stop()
        assert not monitor._is_running
        assert monitor.stop_call_count == 1
    
    @pytest.mark.asyncio
    async def test_mock_callback_triggering(self, sample_gpu_stats):
        """Test manual callback triggering in mock."""
        monitor = MockGPUMonitor()
        
        callback_called = asyncio.Event()
        received_stats = None
        
        def callback(stats: GPUStats):
            nonlocal received_stats
            received_stats = stats
            callback_called.set()
        
        monitor.on_stats_update(callback)
        
        await monitor.trigger_stats_update(sample_gpu_stats)
        
        assert callback_called.is_set()
        assert received_stats == sample_gpu_stats
        assert monitor.get_current_stats() == sample_gpu_stats
    
    @pytest.mark.asyncio
    async def test_mock_callback_management(self):
        """Test callback management in mock monitor."""
        monitor = MockGPUMonitor()
        
        callback1 = MagicMock()
        callback2 = MagicMock()
        
        monitor.on_stats_update(callback1)
        monitor.on_stats_update(callback2)
        assert len(monitor._callbacks) == 2
        
        monitor.remove_stats_callback(callback1)
        assert len(monitor._callbacks) == 1
        assert callback2 in monitor._callbacks


if __name__ == "__main__":
    pytest.main([__file__])