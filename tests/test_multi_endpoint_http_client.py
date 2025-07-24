"""Tests for multi-endpoint HTTP client functionality."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from gpu_worker_pool.multi_endpoint_http_client import AsyncMultiEndpointHTTPClientPool, MockMultiEndpointHTTPClientPool
from gpu_worker_pool.endpoint_manager import MockEndpointManager
from gpu_worker_pool.models import EndpointInfo, GPUStats, GPUInfo, GlobalGPUInfo
from gpu_worker_pool.http_client import ServiceUnavailableError, RetryableError


class TestAsyncMultiEndpointHTTPClientPool:
    """Test cases for AsyncMultiEndpointHTTPClientPool class."""
    
    @pytest.fixture
    def mock_endpoints(self):
        """Create mock endpoints for testing."""
        return [
            EndpointInfo(
                endpoint_id="server1",
                url="http://server1:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=2,
                response_time_ms=50.0
            ),
            EndpointInfo(
                endpoint_id="server2",
                url="http://server2:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=2,
                available_gpus=1,
                response_time_ms=75.0
            ),
            EndpointInfo(
                endpoint_id="server3",
                url="http://server3:8000",
                is_healthy=False,
                last_seen=datetime.now(),
                total_gpus=6,
                available_gpus=0,
                response_time_ms=0.0
            )
        ]
    
    @pytest.fixture
    def mock_endpoint_manager(self, mock_endpoints):
        """Create mock endpoint manager."""
        return MockEndpointManager(mock_endpoints)
    
    @pytest.fixture
    def http_client_pool(self, mock_endpoint_manager):
        """Create HTTP client pool with mock endpoint manager."""
        return AsyncMultiEndpointHTTPClientPool(
            endpoint_manager=mock_endpoint_manager
        )
    
    @pytest.fixture
    def sample_gpu_stats(self):
        """Create sample GPU statistics."""
        return GPUStats(
            gpu_count=4,
            total_memory_mb=32768,
            total_used_memory_mb=8192,
            average_utilization_percent=50.0,
            gpus_summary=[
                GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=20.0, utilization_percent=30.0),
                GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=25.0, utilization_percent=40.0),
                GPUInfo(gpu_id=2, name="GPU-2", memory_usage_percent=15.0, utilization_percent=60.0),
                GPUInfo(gpu_id=3, name="GPU-3", memory_usage_percent=30.0, utilization_percent=70.0)
            ],
            total_memory_usage_percent=25.0,
            timestamp=datetime.now().isoformat()
        )
    
    def test_http_client_pool_initialization(self, mock_endpoint_manager):
        """Test HTTP client pool initialization."""
        pool = AsyncMultiEndpointHTTPClientPool(
            endpoint_manager=mock_endpoint_manager
        )
        
        assert pool.endpoint_manager == mock_endpoint_manager
        assert len(pool._request_stats) == 0
    
    @pytest.mark.asyncio
    async def test_fetch_gpu_stats_from_single_endpoint(self, http_client_pool, sample_gpu_stats):
        """Test fetching GPU stats from a single endpoint."""
        # Mock the endpoint manager's get_http_client to return our mock
        mock_client = AsyncMock()
        mock_client.fetch_gpu_stats.return_value = sample_gpu_stats
        
        # Mock the endpoint manager to return our mock client and make endpoint healthy
        http_client_pool.endpoint_manager.get_http_client = lambda endpoint_id: mock_client
        
        # Ensure endpoint exists and is healthy
        server1_endpoint = http_client_pool.endpoint_manager.get_endpoint_by_id("server1")
        if server1_endpoint:
            server1_endpoint.is_healthy = True
        
        result = await http_client_pool.fetch_gpu_stats_from_endpoint("server1")
        
        assert result == sample_gpu_stats
        mock_client.fetch_gpu_stats.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_gpu_stats_from_nonexistent_endpoint(self, http_client_pool):
        """Test fetching GPU stats from non-existent endpoint."""
        result = await http_client_pool.fetch_gpu_stats_from_endpoint("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_fetch_gpu_stats_with_service_unavailable_error(self, http_client_pool):
        """Test handling ServiceUnavailableError."""
        mock_client = AsyncMock()
        mock_client.fetch_gpu_stats.side_effect = ServiceUnavailableError("Service down")
        
        # Mock the endpoint manager to return our mock client and make endpoint healthy
        http_client_pool.endpoint_manager.get_http_client = lambda endpoint_id: mock_client
        
        # Ensure endpoint exists and is healthy
        server1_endpoint = http_client_pool.endpoint_manager.get_endpoint_by_id("server1")
        if server1_endpoint:
            server1_endpoint.is_healthy = True
        
        result = await http_client_pool.fetch_gpu_stats_from_endpoint("server1")
        
        assert result is None
        
        # Check request stats were updated
        stats = http_client_pool._request_stats.get("server1", {})
        assert stats.get("failed_requests", 0) > 0
    
    @pytest.mark.asyncio
    async def test_fetch_gpu_stats_with_retryable_error(self, http_client_pool):
        """Test handling RetryableError."""
        mock_client = AsyncMock()
        mock_client.fetch_gpu_stats.side_effect = RetryableError("Temporary failure")
        
        # Mock the endpoint manager to return our mock client and make endpoint healthy
        http_client_pool.endpoint_manager.get_http_client = lambda endpoint_id: mock_client
        
        # Ensure endpoint exists and is healthy
        server1_endpoint = http_client_pool.endpoint_manager.get_endpoint_by_id("server1")
        if server1_endpoint:
            server1_endpoint.is_healthy = True
        
        result = await http_client_pool.fetch_gpu_stats_from_endpoint("server1")
        
        assert result is None
        
        # Check request stats were updated
        stats = http_client_pool._request_stats.get("server1", {})
        assert stats.get("failed_requests", 0) > 0
    
    @pytest.mark.asyncio
    async def test_fetch_gpu_stats_with_unexpected_error(self, http_client_pool):
        """Test handling unexpected errors."""
        mock_client = AsyncMock()
        mock_client.fetch_gpu_stats.side_effect = Exception("Unexpected error")
        
        # Mock the endpoint manager to return our mock client and make endpoint healthy
        http_client_pool.endpoint_manager.get_http_client = lambda endpoint_id: mock_client
        
        # Ensure endpoint exists and is healthy
        server1_endpoint = http_client_pool.endpoint_manager.get_endpoint_by_id("server1")
        if server1_endpoint:
            server1_endpoint.is_healthy = True
        
        result = await http_client_pool.fetch_gpu_stats_from_endpoint("server1")
        
        assert result is None
        
        # Check request stats were updated
        stats = http_client_pool._request_stats.get("server1", {})
        assert stats.get("failed_requests", 0) > 0
    
    @pytest.mark.asyncio
    async def test_fetch_aggregated_gpu_stats_success(self, http_client_pool, sample_gpu_stats):
        """Test fetching aggregated GPU stats from multiple endpoints."""
        # Create different GPU stats for each endpoint
        stats2 = GPUStats(
            gpu_count=2,
            total_memory_mb=16384,
            total_used_memory_mb=4096,
            average_utilization_percent=30.0,
            gpus_summary=[
                GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=15.0, utilization_percent=20.0),
                GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=35.0, utilization_percent=40.0)
            ],
            total_memory_usage_percent=25.0,
            timestamp=datetime.now().isoformat()
        )
        
        # Mock the fetch method to return data for different endpoints
        async def mock_fetch_with_metadata(endpoint_id, url):
            if endpoint_id == "server1":
                return ("server1", sample_gpu_stats)
            elif endpoint_id == "server2":
                return ("server2", stats2)
            return None
        
        # Mock the internal method
        http_client_pool._fetch_gpu_stats_with_metadata = mock_fetch_with_metadata
        
        result = await http_client_pool.fetch_aggregated_gpu_stats()
        
        assert result is not None
        assert len(result) == 6  # 4 GPUs from server1 + 2 GPUs from server2
        
        # Check that global GPU IDs are created correctly
        global_gpu_ids = [gpu.global_gpu_id for gpu in result]
        assert "server1:0" in global_gpu_ids
        assert "server1:1" in global_gpu_ids
        assert "server2:0" in global_gpu_ids
        assert "server2:1" in global_gpu_ids
    
    @pytest.mark.asyncio
    async def test_fetch_aggregated_gpu_stats_partial_failure(self, http_client_pool, sample_gpu_stats):
        """Test fetching aggregated GPU stats with some endpoints failing."""
        # Mock the fetch method - one succeeds, one fails
        async def mock_fetch_with_metadata(endpoint_id, url):
            if endpoint_id == "server1":
                return ("server1", sample_gpu_stats)
            elif endpoint_id == "server2":
                raise ServiceUnavailableError("Service down")
            return None
        
        # Mock the internal method
        http_client_pool._fetch_gpu_stats_with_metadata = mock_fetch_with_metadata
        
        result = await http_client_pool.fetch_aggregated_gpu_stats()
        
        assert result is not None
        assert len(result) == 4  # Only GPUs from server1 (server2 failed)
        
        # All should be from server1
        assert all(gpu.endpoint_id == "server1" for gpu in result)
    
    @pytest.mark.asyncio
    async def test_fetch_aggregated_gpu_stats_all_endpoints_fail(self, http_client_pool):
        """Test fetching aggregated GPU stats when all endpoints fail."""
        # Mock the fetch method to always fail
        async def mock_fetch_with_metadata(endpoint_id, url):
            raise ServiceUnavailableError("All services down")
        
        # Mock the internal method
        http_client_pool._fetch_gpu_stats_with_metadata = mock_fetch_with_metadata
        
        result = await http_client_pool.fetch_aggregated_gpu_stats()
        
        assert result is None
    
    def test_get_client_pool_statistics(self, http_client_pool):
        """Test getting client pool statistics."""
        # Initialize some stats
        http_client_pool._request_stats["server1"] = {
            "total_requests": 10,
            "successful_requests": 8,
            "failed_requests": 2,
            "average_response_time_ms": 75.5,
            "last_request_time": datetime.now(),  # Should be datetime object, not string
            "last_error": "Connection timeout"
        }
        
        stats = http_client_pool.get_client_pool_statistics()
        
        assert isinstance(stats, dict)
        assert "pool_summary" in stats
        assert "endpoint_statistics" in stats
    
    @pytest.mark.asyncio
    async def test_close_clients(self, http_client_pool):
        """Test closing HTTP clients."""
        # Add some request stats
        http_client_pool._request_stats["server1"] = {"total_requests": 5}
        http_client_pool._last_successful_fetch["server1"] = datetime.now()
        
        await http_client_pool.close()
        
        # Should clear internal state
        assert len(http_client_pool._request_stats) == 0
        assert len(http_client_pool._last_successful_fetch) == 0
    
    def test_update_request_stats_success(self, http_client_pool):
        """Test updating request statistics for successful requests."""
        http_client_pool._update_request_stats("server1", success=True)
        
        stats = http_client_pool._request_stats["server1"]
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["failed_requests"] == 0
    
    def test_update_request_stats_failure(self, http_client_pool):
        """Test updating request statistics for failed requests."""
        http_client_pool._update_request_stats("server1", success=False, error="Connection failed")
        
        stats = http_client_pool._request_stats["server1"]
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 1
        assert stats["last_error"] == "Connection failed"


class TestMockMultiEndpointHTTPClientPool:
    """Test cases for MockMultiEndpointHTTPClientPool class."""
    
    @pytest.fixture
    def mock_endpoints(self):
        """Create mock endpoints for testing."""
        return [
            EndpointInfo(
                endpoint_id="server1",
                url="http://server1:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=2,
                response_time_ms=50.0
            ),
            EndpointInfo(
                endpoint_id="server2",
                url="http://server2:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=2,
                available_gpus=1,
                response_time_ms=75.0
            )
        ]
    
    @pytest.fixture
    def mock_endpoint_manager(self, mock_endpoints):
        """Create mock endpoint manager."""
        return MockEndpointManager(mock_endpoints)
    
    def test_mock_http_client_pool_initialization(self):
        """Test mock HTTP client pool initialization."""
        # Create mock responses
        mock_responses = {
            "server1": [
                GlobalGPUInfo(
                    global_gpu_id="server1:0",
                    endpoint_id="server1",
                    local_gpu_id=0,
                    name="GPU-0",
                    memory_usage_percent=20.0,
                    utilization_percent=30.0,
                    is_available=True
                )
            ]
        }
        
        pool = MockMultiEndpointHTTPClientPool(mock_responses=mock_responses)
        
        assert len(pool.mock_responses) == 1
        assert pool.fetch_call_count == 0
    
    @pytest.mark.asyncio
    async def test_mock_fetch_gpu_stats_from_endpoint(self):
        """Test mock fetching GPU stats from endpoint."""
        mock_responses = {
            "server1": [
                GlobalGPUInfo(
                    global_gpu_id="server1:0",
                    endpoint_id="server1",
                    local_gpu_id=0,
                    name="GPU-0",
                    memory_usage_percent=15.0,
                    utilization_percent=20.0,
                    is_available=True
                )
            ]
        }
        
        pool = MockMultiEndpointHTTPClientPool(mock_responses=mock_responses)
        
        result = await pool.fetch_gpu_stats_from_endpoint("server1")
        
        assert result is not None
        assert result.gpu_count == 1
        assert pool.endpoint_fetch_counts.get("server1", 0) == 1
        
        # Test non-existent endpoint
        result = await pool.fetch_gpu_stats_from_endpoint("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_mock_fetch_aggregated_gpu_stats(self):
        """Test mock fetching aggregated GPU stats."""
        mock_global_gpus = [
            GlobalGPUInfo(
                global_gpu_id="server1:0",
                endpoint_id="server1",
                local_gpu_id=0,
                name="GPU-0",
                memory_usage_percent=20.0,
                utilization_percent=30.0,
                is_available=True
            ),
            GlobalGPUInfo(
                global_gpu_id="server1:1",
                endpoint_id="server1",
                local_gpu_id=1,
                name="GPU-1",
                memory_usage_percent=25.0,
                utilization_percent=40.0,
                is_available=False
            )
        ]
        
        pool = MockMultiEndpointHTTPClientPool()
        # Set mock data using set_mock_response
        pool.set_mock_response("server1", mock_global_gpus)
        
        result = await pool.fetch_aggregated_gpu_stats()
        
        assert result == mock_global_gpus
        assert pool.fetch_call_count == 1
    
    def test_mock_basic_functionality(self):
        """Test basic mock functionality."""
        pool = MockMultiEndpointHTTPClientPool()
        
        # Test initial state
        assert pool.fetch_call_count == 0
        assert not pool.should_fail
    
    @pytest.mark.asyncio
    async def test_mock_close(self):
        """Test mock close method."""
        pool = MockMultiEndpointHTTPClientPool()
        
        # Should not raise exception
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_mock_trigger_endpoint_failure(self):
        """Test triggering endpoint failure in mock."""
        pool = MockMultiEndpointHTTPClientPool()
        
        pool.set_failure_mode(True, ServiceUnavailableError("Service down"))
        
        # Should raise the exception, not return None
        with pytest.raises(ServiceUnavailableError, match="Service down"):
            await pool.fetch_gpu_stats_from_endpoint("server1")
        
        # Clear failure
        pool.set_failure_mode(False)
        
        # Should now return data (None by default since no mock data was set)
        result = await pool.fetch_gpu_stats_from_endpoint("server1")
        assert result is None