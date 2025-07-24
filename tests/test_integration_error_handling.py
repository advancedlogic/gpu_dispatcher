"""
End-to-end integration tests with realistic scenarios and error handling.

These tests demonstrate the GPU Worker Pool system behavior under various
real-world conditions including service failures, high load, and edge cases.
"""

import asyncio
import pytest
import logging
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import aiohttp

from gpu_worker_pool.client import GPUWorkerPoolClient, gpu_worker_pool_client, GPUContextManager
from gpu_worker_pool.models import GPUStats, GPUInfo
from gpu_worker_pool.worker_pool_manager import WorkerTimeoutError

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockGPUService:
    """Mock GPU service that can simulate various conditions."""
    
    def __init__(self):
        self.call_count = 0
        self.failure_mode = None
        self.gpu_count = 2
        self.high_usage = False
        self.response_delay = 0.0
    
    def set_failure_mode(self, mode: str, duration: int = 3):
        """Set failure mode for testing error handling."""
        self.failure_mode = mode
        self.failure_duration = duration
    
    def set_high_usage(self, enabled: bool):
        """Enable/disable high GPU usage simulation."""
        self.high_usage = enabled
    
    def set_response_delay(self, delay: float):
        """Set response delay for testing timeouts."""
        self.response_delay = delay
    
    async def get_response(self):
        """Generate mock response based on current state."""
        self.call_count += 1
        
        # Simulate response delay
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        # Handle failure modes
        if self.failure_mode == "network_error" and self.call_count <= self.failure_duration:
            raise aiohttp.ClientError("Network connection failed")
        
        if self.failure_mode == "service_unavailable" and self.call_count <= self.failure_duration:
            mock_response = AsyncMock()
            mock_response.status = 503
            mock_response.json.side_effect = aiohttp.ClientError("Service unavailable")
            return mock_response
        
        if self.failure_mode == "invalid_response" and self.call_count <= self.failure_duration:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"invalid": "response"}
            return mock_response
        
        # Generate normal response
        if self.high_usage:
            gpu_data = [
                {"gpu_id": i, "name": f"GPU-{i}", "memory_usage_percent": 90.0 + i, "utilization_percent": 95.0 + i}
                for i in range(self.gpu_count)
            ]
            total_memory_usage = 90.0
            avg_utilization = 95.0
        else:
            gpu_data = [
                {"gpu_id": i, "name": f"GPU-{i}", "memory_usage_percent": 20.0 + (i * 10), "utilization_percent": 15.0 + (i * 20)}
                for i in range(self.gpu_count)
            ]
            total_memory_usage = 25.0
            avg_utilization = 25.0
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "gpu_count": self.gpu_count,
            "total_memory_mb": 16384,
            "total_used_memory_mb": int(16384 * total_memory_usage / 100),
            "average_utilization_percent": avg_utilization,
            "gpus_summary": gpu_data,
            "total_memory_usage_percent": total_memory_usage,
            "timestamp": datetime.now().isoformat()
        }
        
        return mock_response


@pytest.fixture
def mock_gpu_service():
    """Fixture providing a configurable mock GPU service."""
    return MockGPUService()


class TestRealisticScenarios:
    """Test realistic usage scenarios."""
    
    @pytest.mark.asyncio
    async def test_ml_training_pipeline(self, mock_gpu_service):
        """Test a realistic ML training pipeline scenario."""
        mock_gpu_service.gpu_count = 4
        
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(side_effect=mock_gpu_service.get_response)
            mock_session_class.return_value = mock_session
            
            async def training_job(client, job_id: str, duration: float):
                """Simulate a training job."""
                assignment = await client.request_gpu(timeout=10.0)
                logger.info(f"Job {job_id} assigned to GPU {assignment.gpu_id}")
                
                # Simulate training work
                await asyncio.sleep(duration)
                
                await client.release_gpu(assignment)
                logger.info(f"Job {job_id} completed")
                return assignment.gpu_id
            
            # Test with production-like configuration
            config = {
                "memory_threshold": 75.0,
                "utilization_threshold": 85.0,
                "worker_timeout": 30.0,
                "polling_interval": 2
            }
            
            async with gpu_worker_pool_client(**config) as client:
                # Wait for initial stats
                await asyncio.sleep(0.2)
                
                # Start multiple training jobs
                jobs = [
                    training_job(client, f"model-{i}", 0.3)
                    for i in range(3)
                ]
                
                # Run jobs and collect results
                gpu_assignments = await asyncio.gather(*jobs)
                
                # Verify all jobs got different GPUs (when possible)
                unique_gpus = set(gpu_assignments)
                assert len(unique_gpus) <= 4  # Can't exceed available GPUs
                assert len(unique_gpus) >= min(3, 4)  # Should use multiple GPUs when available
    
    @pytest.mark.asyncio
    async def test_high_load_scenario(self, mock_gpu_service):
        """Test system behavior under high load."""
        mock_gpu_service.gpu_count = 2
        
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(side_effect=mock_gpu_service.get_response)
            mock_session_class.return_value = mock_session
            
            async def worker_task(client, worker_id: int):
                """Simulate a worker task."""
                try:
                    assignment = await client.request_gpu(timeout=5.0)
                    await asyncio.sleep(0.2)  # Short work duration
                    await client.release_gpu(assignment)
                    return f"worker-{worker_id}-success"
                except WorkerTimeoutError:
                    return f"worker-{worker_id}-timeout"
                except Exception as e:
                    return f"worker-{worker_id}-error-{type(e).__name__}"
            
            async with gpu_worker_pool_client(worker_timeout=5.0) as client:
                # Wait for initial stats
                await asyncio.sleep(0.2)
                
                # Start many concurrent workers (more than available GPUs)
                workers = [
                    worker_task(client, i)
                    for i in range(10)  # 10 workers, 2 GPUs
                ]
                
                results = await asyncio.gather(*workers)
                
                # Analyze results
                successes = [r for r in results if r.endswith('-success')]
                timeouts = [r for r in results if r.endswith('-timeout')]
                errors = [r for r in results if '-error-' in r]
                
                logger.info(f"Results: {len(successes)} successes, {len(timeouts)} timeouts, {len(errors)} errors")
                
                # Should have some successes and possibly some timeouts due to queuing
                assert len(successes) > 0
                assert len(errors) == 0  # No unexpected errors
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_scenario(self, mock_gpu_service):
        """Test behavior when all GPUs exceed thresholds."""
        mock_gpu_service.gpu_count = 2
        mock_gpu_service.set_high_usage(True)
        
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(side_effect=mock_gpu_service.get_response)
            mock_session_class.return_value = mock_session
            
            # Use conservative thresholds
            config = {
                "memory_threshold": 80.0,
                "utilization_threshold": 90.0,
                "worker_timeout": 2.0  # Short timeout for quick test
            }
            
            async with gpu_worker_pool_client(**config) as client:
                # Wait for initial stats
                await asyncio.sleep(0.2)
                
                # Check that no GPUs are available due to high usage
                status = client.get_pool_status()
                assert status.total_gpus == 2
                assert status.available_gpus == 0  # All GPUs should exceed thresholds
                
                # Try to request GPU - should timeout
                with pytest.raises(WorkerTimeoutError):
                    await client.request_gpu(timeout=2.0)


class TestErrorHandling:
    """Test error handling and recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_service_recovery_scenario(self, mock_gpu_service):
        """Test recovery when GPU service becomes available after being down."""
        mock_gpu_service.set_failure_mode("network_error", duration=3)
        
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(side_effect=mock_gpu_service.get_response)
            mock_session_class.return_value = mock_session
            
            async with gpu_worker_pool_client(worker_timeout=10.0) as client:
                # Wait for service to recover (after 3 failed attempts)
                await asyncio.sleep(1.0)
                
                # Should eventually be able to request GPU after service recovery
                assignment = await client.request_gpu(timeout=10.0)
                assert assignment is not None
                assert assignment.gpu_id >= 0
                
                await client.release_gpu(assignment)
    
    @pytest.mark.asyncio
    async def test_invalid_response_handling(self, mock_gpu_service):
        """Test handling of invalid responses from GPU service."""
        mock_gpu_service.set_failure_mode("invalid_response", duration=2)
        
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(side_effect=mock_gpu_service.get_response)
            mock_session_class.return_value = mock_session
            
            async with gpu_worker_pool_client(worker_timeout=8.0) as client:
                # Wait for service to start returning valid responses
                await asyncio.sleep(0.5)
                
                # Should eventually get valid response and be able to request GPU
                assignment = await client.request_gpu(timeout=8.0)
                assert assignment is not None
                
                await client.release_gpu(assignment)
    
    @pytest.mark.asyncio
    async def test_service_unavailable_handling(self, mock_gpu_service):
        """Test handling when service returns 503 Service Unavailable."""
        mock_gpu_service.set_failure_mode("service_unavailable", duration=2)
        
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(side_effect=mock_gpu_service.get_response)
            mock_session_class.return_value = mock_session
            
            async with gpu_worker_pool_client(worker_timeout=8.0) as client:
                # Wait for service to become available
                await asyncio.sleep(0.5)
                
                # Should eventually be able to request GPU
                assignment = await client.request_gpu(timeout=8.0)
                assert assignment is not None
                
                await client.release_gpu(assignment)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_gpu_service):
        """Test handling of request timeouts."""
        mock_gpu_service.set_response_delay(2.0)  # 2 second delay
        
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(side_effect=mock_gpu_service.get_response)
            mock_session_class.return_value = mock_session
            
            # Use short timeout that will be exceeded
            config = {
                "request_timeout": 1.0,  # 1 second timeout
                "worker_timeout": 5.0
            }
            
            async with gpu_worker_pool_client(**config) as client:
                # The HTTP client should handle timeouts and retry
                # Eventually should succeed when retries work
                await asyncio.sleep(0.5)
                
                # This might timeout due to slow responses, but system should handle it gracefully
                try:
                    assignment = await client.request_gpu(timeout=5.0)
                    if assignment:
                        await client.release_gpu(assignment)
                except WorkerTimeoutError:
                    # Acceptable outcome due to slow service
                    pass


class TestConfigurationValidation:
    """Test configuration validation and error handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_threshold_handling(self, mock_gpu_service):
        """Test that invalid thresholds are handled gracefully."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(side_effect=mock_gpu_service.get_response)
            mock_session_class.return_value = mock_session
            
            # Test with invalid thresholds - should use defaults
            config = {
                "memory_threshold": 150.0,  # Invalid - over 100%
                "utilization_threshold": -10.0  # Invalid - negative
            }
            
            # Client should start successfully with default values
            async with gpu_worker_pool_client(**config) as client:
                await asyncio.sleep(0.2)
                
                # Should use reasonable default thresholds
                metrics = client.get_detailed_metrics()
                thresholds = metrics["thresholds"]
                
                # Should have valid threshold values (defaults)
                assert 0 <= thresholds["memory_threshold_percent"] <= 100
                assert 0 <= thresholds["utilization_threshold_percent"] <= 100
    
    @pytest.mark.asyncio
    async def test_configuration_edge_cases(self, mock_gpu_service):
        """Test edge cases in configuration."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(side_effect=mock_gpu_service.get_response)
            mock_session_class.return_value = mock_session
            
            # Test with extreme but valid values
            config = {
                "memory_threshold": 0.1,    # Very low
                "utilization_threshold": 99.9,  # Very high
                "worker_timeout": 1.0,      # Very short
                "polling_interval": 1       # Very frequent
            }
            
            async with gpu_worker_pool_client(**config) as client:
                await asyncio.sleep(0.2)
                
                # Should work with extreme values
                status = client.get_pool_status()
                assert status.total_gpus >= 0


class TestConcurrencyAndRaceConditions:
    """Test concurrent access and race condition handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_and_releases(self, mock_gpu_service):
        """Test concurrent GPU requests and releases."""
        mock_gpu_service.gpu_count = 3
        
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(side_effect=mock_gpu_service.get_response)
            mock_session_class.return_value = mock_session
            
            async def request_release_cycle(client, cycle_id: int):
                """Perform multiple request/release cycles."""
                results = []
                for i in range(3):
                    try:
                        assignment = await client.request_gpu(timeout=5.0)
                        await asyncio.sleep(0.1)  # Hold GPU briefly
                        await client.release_gpu(assignment)
                        results.append(f"cycle-{cycle_id}-{i}-success")
                    except Exception as e:
                        results.append(f"cycle-{cycle_id}-{i}-{type(e).__name__}")
                return results
            
            async with gpu_worker_pool_client() as client:
                await asyncio.sleep(0.2)
                
                # Run multiple concurrent request/release cycles
                cycles = [
                    request_release_cycle(client, i)
                    for i in range(5)
                ]
                
                all_results = await asyncio.gather(*cycles)
                
                # Flatten results
                flat_results = [item for sublist in all_results for item in sublist]
                
                # Should have mostly successful cycles
                successes = [r for r in flat_results if r.endswith('-success')]
                assert len(successes) > len(flat_results) * 0.8  # At least 80% success rate
    
    @pytest.mark.asyncio
    async def test_rapid_client_lifecycle(self, mock_gpu_service):
        """Test rapid client start/stop cycles."""
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(side_effect=mock_gpu_service.get_response)
            mock_session_class.return_value = mock_session
            
            # Rapidly start and stop clients
            for i in range(5):
                async with gpu_worker_pool_client() as client:
                    await asyncio.sleep(0.1)  # Brief usage
                    status = client.get_pool_status()
                    assert status is not None


if __name__ == "__main__":
    # Run a simple integration test
    async def main():
        print("Running integration error handling tests...")
        
        mock_service = MockGPUService()
        
        # Test basic functionality
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(side_effect=mock_service.get_response)
            mock_session_class.return_value = mock_session
            
            async with gpu_worker_pool_client() as client:
                await asyncio.sleep(0.2)
                
                assignment = await client.request_gpu(timeout=5.0)
                print(f"GPU {assignment.gpu_id} assigned successfully")
                
                await client.release_gpu(assignment)
                print("GPU released successfully")
                
                print("Integration test completed!")
    
    asyncio.run(main())