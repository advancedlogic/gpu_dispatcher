"""Integration tests for the GPU Worker Pool client interface."""

import asyncio
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from gpu_worker_pool.client import GPUWorkerPoolClient, gpu_worker_pool_client, GPUContextManager
from gpu_worker_pool.models import GPUStats, GPUInfo, GPUAssignment


# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def mock_gpu_stats():
    """Create mock GPU statistics for testing."""
    return GPUStats(
        gpu_count=2,
        total_memory_mb=16384,
        total_used_memory_mb=4096,
        average_utilization_percent=25.0,
        gpus_summary=[
            GPUInfo(gpu_id=0, name="GPU-0", memory_usage_percent=20.0, utilization_percent=15.0),
            GPUInfo(gpu_id=1, name="GPU-1", memory_usage_percent=30.0, utilization_percent=35.0)
        ],
        total_memory_usage_percent=25.0,
        timestamp=datetime.now().isoformat()
    )


@pytest.fixture
def mock_http_response():
    """Create mock HTTP response for GPU statistics."""
    return {
        "gpu_count": 2,
        "total_memory_mb": 16384,
        "total_used_memory_mb": 4096,
        "average_utilization_percent": 25.0,
        "gpus_summary": [
            {"gpu_id": 0, "name": "GPU-0", "memory_usage_percent": 20.0, "utilization_percent": 15.0},
            {"gpu_id": 1, "name": "GPU-1", "memory_usage_percent": 30.0, "utilization_percent": 35.0}
        ],
        "total_memory_usage_percent": 25.0,
        "timestamp": datetime.now().isoformat()
    }


class TestGPUWorkerPoolClient:
    """Test cases for GPUWorkerPoolClient."""
    
    @pytest.mark.asyncio
    async def test_client_lifecycle(self, mock_http_response):
        """Test basic client lifecycle - start and stop."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            # Mock HTTP client responses
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_http_response
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            client = GPUWorkerPoolClient()
            
            # Test start
            await client.start()
            assert client._is_started is True
            assert client._pool_manager is not None
            
            # Test stop
            await client.stop()
            assert client._is_started is False
    
    @pytest.mark.asyncio
    async def test_client_context_manager(self, mock_http_response):
        """Test client as async context manager."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            # Mock HTTP client responses
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_http_response
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            async with GPUWorkerPoolClient() as client:
                assert client._is_started is True
                assert client._pool_manager is not None
            
            # Client should be stopped after context exit
            assert client._is_started is False
    
    @pytest.mark.asyncio
    async def test_gpu_request_and_release(self, mock_http_response):
        """Test GPU request and release workflow."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            # Mock HTTP client responses
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_http_response
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            async with GPUWorkerPoolClient() as client:
                # Wait a moment for GPU monitor to get initial stats
                await asyncio.sleep(0.1)
                
                # Request GPU
                assignment = await client.request_gpu(timeout=5.0)
                assert isinstance(assignment, GPUAssignment)
                assert assignment.gpu_id in [0, 1]  # Should be one of our mock GPUs
                assert assignment.worker_id is not None
                assert assignment.assigned_at is not None
                
                # Check pool status
                status = client.get_pool_status()
                assert status.total_gpus == 2
                assert status.active_workers == 1
                assert status.available_gpus == 1  # One GPU is now assigned
                
                # Release GPU
                await client.release_gpu(assignment)
                
                # Check pool status after release
                status = client.get_pool_status()
                assert status.active_workers == 0
                assert status.available_gpus == 2  # Both GPUs should be available again
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, mock_http_response):
        """Test multiple concurrent GPU requests."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            # Mock HTTP client responses
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_http_response
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            async with GPUWorkerPoolClient() as client:
                # Wait a moment for GPU monitor to get initial stats
                await asyncio.sleep(0.1)
                
                # Request multiple GPUs concurrently
                assignments = await asyncio.gather(
                    client.request_gpu(timeout=5.0),
                    client.request_gpu(timeout=5.0)
                )
                
                assert len(assignments) == 2
                assert assignments[0].gpu_id != assignments[1].gpu_id  # Should get different GPUs
                
                # Check pool status
                status = client.get_pool_status()
                assert status.active_workers == 2
                assert status.available_gpus == 0  # All GPUs should be assigned
                
                # Release all assignments
                await asyncio.gather(
                    client.release_gpu(assignments[0]),
                    client.release_gpu(assignments[1])
                )
                
                # Check pool status after release
                status = client.get_pool_status()
                assert status.active_workers == 0
                assert status.available_gpus == 2
    
    @pytest.mark.asyncio
    async def test_client_with_custom_parameters(self, mock_http_response):
        """Test client initialization with custom parameters."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            # Mock HTTP client responses
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_http_response
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            client = GPUWorkerPoolClient(
                service_endpoint="http://custom-endpoint:8000",
                memory_threshold=70.0,
                utilization_threshold=85.0,
                polling_interval=10,
                worker_timeout=120.0,
                request_timeout=15.0
            )
            
            await client.start()
            
            # Verify custom parameters are applied
            assert client._config.get_service_endpoint() == "http://custom-endpoint:8000"
            assert client._config.get_memory_threshold() == 70.0
            assert client._config.get_utilization_threshold() == 85.0
            assert client._config.get_polling_interval() == 10
            assert client.worker_timeout == 120.0
            assert client.request_timeout == 15.0
            
            await client.stop()
    
    @pytest.mark.asyncio
    async def test_client_error_handling(self):
        """Test client error handling for various scenarios."""
        client = GPUWorkerPoolClient()
        
        # Test operations before start
        with pytest.raises(RuntimeError, match="Client must be started"):
            await client.request_gpu()
        
        with pytest.raises(RuntimeError, match="Client must be started"):
            await client.release_gpu(GPUAssignment(gpu_id=0, worker_id="test", assigned_at=datetime.now()))
        
        with pytest.raises(RuntimeError, match="Client must be started"):
            client.get_pool_status()
        
        # Test double start
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession'):
            await client.start()
            
            with pytest.raises(RuntimeError, match="Client is already started"):
                await client.start()
            
            await client.stop()
    
    @pytest.mark.asyncio
    async def test_detailed_metrics(self, mock_http_response):
        """Test detailed metrics functionality."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            # Mock HTTP client responses
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_http_response
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            async with GPUWorkerPoolClient() as client:
                # Wait a moment for GPU monitor to get initial stats
                await asyncio.sleep(0.1)
                
                # Get detailed metrics
                metrics = client.get_detailed_metrics()
                
                assert "timestamp" in metrics
                assert "is_running" in metrics
                assert "total_gpus" in metrics
                assert "available_gpus" in metrics
                assert "active_workers" in metrics
                assert "blocked_workers" in metrics
                assert "gpu_metrics" in metrics
                assert "assignment_metrics" in metrics
                assert "queue_metrics" in metrics
                assert "thresholds" in metrics
                
                assert metrics["is_running"] is True
                assert metrics["total_gpus"] == 2
                assert len(metrics["gpu_metrics"]) == 2


class TestGPUWorkerPoolClientFactory:
    """Test cases for gpu_worker_pool_client factory function."""
    
    @pytest.mark.asyncio
    async def test_factory_function(self, mock_http_response):
        """Test the gpu_worker_pool_client factory function."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            # Mock HTTP client responses
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_http_response
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            async with gpu_worker_pool_client(memory_threshold=75.0) as client:
                assert isinstance(client, GPUWorkerPoolClient)
                assert client._is_started is True
                assert client._config.get_memory_threshold() == 75.0
            
            # Client should be stopped after context exit
            assert client._is_started is False


class TestGPUContextManager:
    """Test cases for GPUContextManager."""
    
    @pytest.mark.asyncio
    async def test_gpu_context_manager(self, mock_http_response):
        """Test GPU context manager for automatic assignment and release."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            # Mock HTTP client responses
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_http_response
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            async with GPUWorkerPoolClient() as client:
                # Wait a moment for GPU monitor to get initial stats
                await asyncio.sleep(0.1)
                
                # Use GPU context manager
                async with GPUContextManager(client, timeout=5.0) as gpu_id:
                    assert isinstance(gpu_id, int)
                    assert gpu_id in [0, 1]  # Should be one of our mock GPUs
                    
                    # Check that GPU is assigned
                    status = client.get_pool_status()
                    assert status.active_workers == 1
                    assert status.available_gpus == 1
                
                # After context exit, GPU should be released
                status = client.get_pool_status()
                assert status.active_workers == 0
                assert status.available_gpus == 2
    
    @pytest.mark.asyncio
    async def test_nested_gpu_context_managers(self, mock_http_response):
        """Test multiple nested GPU context managers."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            # Mock HTTP client responses
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_http_response
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            async with GPUWorkerPoolClient() as client:
                # Wait a moment for GPU monitor to get initial stats
                await asyncio.sleep(0.1)
                
                # Use nested GPU context managers
                async with GPUContextManager(client) as gpu_id_1:
                    async with GPUContextManager(client) as gpu_id_2:
                        assert gpu_id_1 != gpu_id_2  # Should get different GPUs
                        
                        # Check that both GPUs are assigned
                        status = client.get_pool_status()
                        assert status.active_workers == 2
                        assert status.available_gpus == 0
                    
                    # After inner context exit, one GPU should be released
                    status = client.get_pool_status()
                    assert status.active_workers == 1
                    assert status.available_gpus == 1
                
                # After outer context exit, all GPUs should be released
                status = client.get_pool_status()
                assert status.active_workers == 0
                assert status.available_gpus == 2


class TestIntegrationScenarios:
    """Integration test scenarios demonstrating real-world usage patterns."""
    
    @pytest.mark.asyncio
    async def test_worker_blocking_scenario(self, mock_http_response):
        """Test scenario where workers are blocked due to resource exhaustion."""
        # Modify mock response to have high resource usage
        high_usage_response = mock_http_response.copy()
        high_usage_response["gpus_summary"] = [
            {"gpu_id": 0, "name": "GPU-0", "memory_usage_percent": 95.0, "utilization_percent": 98.0},
            {"gpu_id": 1, "name": "GPU-1", "memory_usage_percent": 92.0, "utilization_percent": 95.0}
        ]
        
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            # Mock HTTP client responses
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = high_usage_response
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            async with GPUWorkerPoolClient(memory_threshold=80.0, utilization_threshold=90.0) as client:
                # Wait a moment for GPU monitor to get initial stats
                await asyncio.sleep(0.1)
                
                # This request should timeout because no GPUs meet thresholds
                with pytest.raises(Exception):  # Should be WorkerTimeoutError
                    await client.request_gpu(timeout=1.0)
                
                # Check that worker was blocked
                status = client.get_pool_status()
                assert status.available_gpus == 0  # No GPUs available due to thresholds
    
    @pytest.mark.asyncio
    async def test_service_recovery_scenario(self, mock_http_response):
        """Test scenario where GPU service becomes available after being down."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            # Mock HTTP client that initially fails then succeeds
            mock_session = AsyncMock()
            
            # First few calls fail, then succeed
            call_count = 0
            async def mock_get(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    # Simulate service down
                    raise Exception("Service unavailable")
                else:
                    # Service is back up
                    mock_response = AsyncMock()
                    mock_response.json.return_value = mock_http_response
                    mock_response.status = 200
                    return mock_response
            
            mock_session.get.side_effect = mock_get
            mock_session_class.return_value = mock_session
            
            async with GPUWorkerPoolClient() as client:
                # Wait for service to recover and get stats
                await asyncio.sleep(0.5)
                
                # Should eventually be able to request GPU after service recovery
                assignment = await client.request_gpu(timeout=10.0)
                assert isinstance(assignment, GPUAssignment)
                
                await client.release_gpu(assignment)
    
    @pytest.mark.asyncio
    async def test_configuration_validation_scenario(self):
        """Test scenario with various configuration validation cases."""
        # Test with invalid thresholds (should use defaults)
        async with gpu_worker_pool_client(
            memory_threshold=150.0,  # Invalid - over 100%
            utilization_threshold=-10.0  # Invalid - negative
        ) as client:
            # Client should start successfully with default values
            assert client._is_started is True
            
            # Should use default thresholds due to invalid values
            config = client._config
            assert 0 <= config.get_memory_threshold() <= 100
            assert 0 <= config.get_utilization_threshold() <= 100


if __name__ == "__main__":
    # Run a simple integration test
    async def main():
        print("Running simple integration test...")
        
        # Mock the HTTP client for this example
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = {
                "gpu_count": 1,
                "total_memory_mb": 8192,
                "total_used_memory_mb": 2048,
                "average_utilization_percent": 25.0,
                "gpus_summary": [
                    {"gpu_id": 0, "name": "Test-GPU", "memory_usage_percent": 25.0, "utilization_percent": 30.0}
                ],
                "total_memory_usage_percent": 25.0,
                "timestamp": datetime.now().isoformat()
            }
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            # Test basic client usage
            async with GPUWorkerPoolClient() as client:
                print("Client started successfully")
                
                # Wait for initial stats
                await asyncio.sleep(0.1)
                
                # Request GPU
                assignment = await client.request_gpu()
                print(f"GPU {assignment.gpu_id} assigned to worker {assignment.worker_id}")
                
                # Get status
                status = client.get_pool_status()
                print(f"Pool status: {status.active_workers} active workers, {status.available_gpus} available GPUs")
                
                # Release GPU
                await client.release_gpu(assignment)
                print("GPU released successfully")
                
                print("Integration test completed successfully!")
    
    asyncio.run(main())