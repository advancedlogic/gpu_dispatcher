"""Unit tests for GPU statistics HTTP client."""

import pytest
import asyncio
import aiohttp
from unittest.mock import AsyncMock, patch, MagicMock
from gpu_worker_pool.http_client import AsyncGPUStatsHTTPClient, MockGPUStatsHTTPClient
from gpu_worker_pool.models import GPUStats, GPUInfo


class TestAsyncGPUStatsHTTPClient:
    """Test cases for AsyncGPUStatsHTTPClient."""
    
    @pytest.fixture
    def client(self):
        """Create HTTP client for testing."""
        return AsyncGPUStatsHTTPClient("http://localhost:8000", timeout=5.0)
    
    @pytest.fixture
    def sample_response_data(self):
        """Sample valid GPU stats response."""
        return {
            "gpu_count": 2,
            "total_memory_mb": 16384,
            "total_used_memory_mb": 8192,
            "average_utilization_percent": 45.5,
            "total_memory_usage_percent": 50.0,
            "timestamp": "2024-01-15T10:30:00Z",
            "gpus_summary": [
                {
                    "gpu_id": 0,
                    "name": "NVIDIA RTX 4090",
                    "memory_usage_percent": 60.0,
                    "utilization_percent": 75.0
                },
                {
                    "gpu_id": 1,
                    "name": "NVIDIA RTX 4090",
                    "memory_usage_percent": 40.0,
                    "utilization_percent": 16.0
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_successful_fetch(self, client, sample_response_data):
        """Test successful GPU stats fetch."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=sample_response_data)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            stats = await client.fetch_gpu_stats()
            
            assert stats is not None
            assert isinstance(stats, GPUStats)
            assert stats.gpu_count == 2
            assert stats.total_memory_mb == 16384
            assert len(stats.gpus_summary) == 2
            assert stats.gpus_summary[0].gpu_id == 0
            assert stats.gpus_summary[0].name == "NVIDIA RTX 4090"
    
    @pytest.mark.asyncio
    async def test_http_error_response(self, client):
        """Test handling of HTTP error responses."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.request_info = MagicMock()
        mock_response.history = []
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(aiohttp.ClientResponseError):
                await client.fetch_gpu_stats()
    
    @pytest.mark.asyncio
    async def test_invalid_json_response(self, client):
        """Test handling of invalid JSON response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"invalid": "data"})
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(ValueError, match="Invalid response format"):
                await client.fetch_gpu_stats()
    
    @pytest.mark.asyncio
    async def test_network_error(self, client):
        """Test handling of network errors."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectorError(
                connection_key=MagicMock(), 
                os_error=OSError("Connection refused")
            )
            
            with pytest.raises(aiohttp.ClientError):
                await client.fetch_gpu_stats()
    
    @pytest.mark.asyncio
    async def test_timeout_error(self, client):
        """Test handling of timeout errors."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(aiohttp.ClientError, match="Request timeout"):
                await client.fetch_gpu_stats()
    
    @pytest.mark.asyncio
    async def test_endpoint_url_construction(self, sample_response_data):
        """Test correct URL construction for different endpoints."""
        # Test with trailing slash
        client1 = AsyncGPUStatsHTTPClient("http://localhost:8000/")
        
        # Test without trailing slash
        client2 = AsyncGPUStatsHTTPClient("http://localhost:8000")
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=sample_response_data)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            await client1.fetch_gpu_stats()
            await client2.fetch_gpu_stats()
            
            # Both should call the same URL
            calls = mock_get.call_args_list
            assert len(calls) == 2
            assert calls[0][0][0] == "http://localhost:8000/gpu/summary"
            assert calls[1][0][0] == "http://localhost:8000/gpu/summary"
    
    @pytest.mark.asyncio
    async def test_session_reuse(self, client, sample_response_data):
        """Test that HTTP session is reused across requests."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=sample_response_data)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Make multiple requests
            await client.fetch_gpu_stats()
            await client.fetch_gpu_stats()
            
            # Both requests should use the same session
            assert mock_get.call_count == 2
    
    @pytest.mark.asyncio
    async def test_close_session(self, client):
        """Test proper session cleanup."""
        # Create a session first
        session = await client._get_session()
        
        with patch.object(session, 'close', new_callable=AsyncMock) as mock_close:
            await client.close()
            mock_close.assert_called_once()


class TestMockGPUStatsHTTPClient:
    """Test cases for MockGPUStatsHTTPClient."""
    
    @pytest.fixture
    def sample_response_data(self):
        """Sample valid GPU stats response."""
        return {
            "gpu_count": 1,
            "total_memory_mb": 8192,
            "total_used_memory_mb": 4096,
            "average_utilization_percent": 50.0,
            "total_memory_usage_percent": 50.0,
            "timestamp": "2024-01-15T10:30:00Z",
            "gpus_summary": [
                {
                    "gpu_id": 0,
                    "name": "Mock GPU",
                    "memory_usage_percent": 50.0,
                    "utilization_percent": 50.0
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_successful_mock_fetch(self, sample_response_data):
        """Test successful mock fetch."""
        client = MockGPUStatsHTTPClient(mock_response=sample_response_data)
        
        stats = await client.fetch_gpu_stats()
        
        assert stats is not None
        assert isinstance(stats, GPUStats)
        assert stats.gpu_count == 1
        assert client.call_count == 1
    
    @pytest.mark.asyncio
    async def test_mock_failure(self):
        """Test mock failure simulation."""
        client = MockGPUStatsHTTPClient(
            should_fail=True, 
            failure_exception=aiohttp.ClientError("Mock network error")
        )
        
        with pytest.raises(aiohttp.ClientError, match="Mock network error"):
            await client.fetch_gpu_stats()
        
        assert client.call_count == 1
    
    @pytest.mark.asyncio
    async def test_mock_none_response(self):
        """Test mock with None response."""
        client = MockGPUStatsHTTPClient(mock_response=None)
        
        stats = await client.fetch_gpu_stats()
        
        assert stats is None
        assert client.call_count == 1
    
    @pytest.mark.asyncio
    async def test_mock_call_counting(self, sample_response_data):
        """Test that mock client counts calls correctly."""
        client = MockGPUStatsHTTPClient(mock_response=sample_response_data)
        
        assert client.call_count == 0
        
        await client.fetch_gpu_stats()
        assert client.call_count == 1
        
        await client.fetch_gpu_stats()
        assert client.call_count == 2
    
    @pytest.mark.asyncio
    async def test_mock_close(self):
        """Test mock close method."""
        client = MockGPUStatsHTTPClient()
        
        # Should not raise any exceptions
        await client.close()


if __name__ == "__main__":
    pytest.main([__file__])