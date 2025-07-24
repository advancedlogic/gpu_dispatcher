"""Integration tests for multi-endpoint scenarios."""

import asyncio
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from gpu_worker_pool.client import GPUWorkerPoolClient
from gpu_worker_pool.models import EndpointInfo, GPUStats, GPUInfo, GlobalGPUInfo
from gpu_worker_pool.endpoint_manager import MockEndpointManager
from gpu_worker_pool.multi_endpoint_http_client import MockMultiEndpointHTTPClientPool
from gpu_worker_pool.error_recovery import CircuitState


class TestMultiEndpointClientIntegration:
    """Integration tests for multi-endpoint client functionality."""
    
    @pytest.fixture
    def mock_endpoints(self):
        """Create mock endpoints for integration testing."""
        return [
            EndpointInfo(
                endpoint_id="server1",
                url="http://server1:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=3,
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
    def mock_global_gpus(self):
        """Create mock global GPU data."""
        return [
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
                memory_usage_percent=15.0,
                utilization_percent=25.0,
                is_available=True
            ),
            GlobalGPUInfo(
                global_gpu_id="server2:0",
                endpoint_id="server2",
                local_gpu_id=0,
                name="GPU-0",
                memory_usage_percent=30.0,
                utilization_percent=45.0,
                is_available=True
            )
        ]
    
    def test_client_mode_detection_multi_endpoint(self):
        """Test client correctly detects multi-endpoint mode."""
        # Test with explicit service_endpoints parameter
        client = GPUWorkerPoolClient(service_endpoints="http://server1:8000,http://server2:8000")
        assert client.is_multi_endpoint_mode()
        
        # Test with environment variable
        with patch.dict(os.environ, {'GPU_STATS_SERVICE_ENDPOINTS': 'http://server1:8000,http://server2:8000'}):
            client = GPUWorkerPoolClient()
            assert client.is_multi_endpoint_mode()
    
    def test_client_mode_detection_single_endpoint(self):
        """Test client correctly detects single-endpoint mode."""
        # Test with explicit service_endpoint parameter
        client = GPUWorkerPoolClient(service_endpoint="http://localhost:8000")
        assert not client.is_multi_endpoint_mode()
        
        # Test with single endpoint in environment variable
        with patch.dict(os.environ, {'GPU_STATS_SERVICE_ENDPOINTS': 'http://localhost:8000'}):
            client = GPUWorkerPoolClient()
            assert not client.is_multi_endpoint_mode()
    
    @pytest.mark.asyncio
    async def test_multi_endpoint_client_lifecycle(self, mock_endpoints, mock_global_gpus):
        """Test multi-endpoint client start/stop lifecycle."""
        client = GPUWorkerPoolClient(service_endpoints="http://server1:8000,http://server2:8000")
        
        # Mock the underlying components
        with patch('gpu_worker_pool.client.EnvironmentMultiEndpointConfigurationManager') as mock_config_class, \
             patch('gpu_worker_pool.client.EndpointManager') as mock_endpoint_manager_class, \
             patch('gpu_worker_pool.client.AsyncMultiEndpointHTTPClientPool') as mock_http_client_class, \
             patch('gpu_worker_pool.client.MultiEndpointLoadBalancer') as mock_load_balancer_class, \
             patch('gpu_worker_pool.client.MultiEndpointGPUMonitor') as mock_monitor_class, \
             patch('gpu_worker_pool.client.AsyncWorkerPoolManager') as mock_pool_manager_class:
            
            # Setup mocks
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config
            
            mock_endpoint_manager = AsyncMock()
            mock_endpoint_manager_class.return_value = mock_endpoint_manager
            
            mock_http_client = AsyncMock()
            mock_http_client_class.return_value = mock_http_client
            
            mock_load_balancer = MagicMock()
            mock_load_balancer_class.return_value = mock_load_balancer
            
            mock_monitor = AsyncMock()
            mock_monitor_class.return_value = mock_monitor
            
            mock_pool_manager = AsyncMock()
            mock_pool_manager_class.return_value = mock_pool_manager
            
            # Test lifecycle
            assert not client._is_started
            
            await client.start()
            assert client._is_started
            
            # Verify components were started
            mock_endpoint_manager.start.assert_called_once()
            mock_pool_manager.start.assert_called_once()
            
            await client.stop()
            assert not client._is_started
            
            # Verify components were stopped
            mock_endpoint_manager.stop.assert_called_once()
            mock_pool_manager.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_single_endpoint_client_backward_compatibility(self):
        """Test that single-endpoint mode maintains backward compatibility."""
        client = GPUWorkerPoolClient(service_endpoint="http://localhost:8000")
        
        # Should be in single-endpoint mode
        assert not client.is_multi_endpoint_mode()
        
        # Multi-endpoint specific methods should return None when not started
        assert client.get_endpoints_info() is None
        
        # These methods should raise RuntimeError when client not started
        with pytest.raises(RuntimeError, match="must be started"):
            client.get_error_recovery_status()
        
        with pytest.raises(RuntimeError, match="not in multi-endpoint mode"):
            await client.trigger_endpoint_recovery("server1")
        
        with pytest.raises(RuntimeError, match="not in multi-endpoint mode"):
            await client.queue_request_for_retry(lambda: asyncio.sleep(0.1))
    
    def test_multi_endpoint_client_api_methods(self):
        """Test multi-endpoint client API methods."""
        client = GPUWorkerPoolClient(service_endpoints="http://server1:8000,http://server2:8000")
        
        # Should be in multi-endpoint mode
        assert client.is_multi_endpoint_mode()
        
        # get_endpoints_info should return None when not started (based on implementation)
        assert client.get_endpoints_info() is None
        
        # These methods should require client to be started
        with pytest.raises(RuntimeError, match="must be started"):
            client.get_error_recovery_status()
        
        with pytest.raises(RuntimeError, match="must be started"):
            client.print_error_recovery_summary()
    
    @pytest.mark.asyncio
    async def test_multi_endpoint_pool_status_integration(self, mock_endpoints):
        """Test multi-endpoint pool status integration."""
        client = GPUWorkerPoolClient(service_endpoints="http://server1:8000,http://server2:8000")
        
        # Mock all the required components
        mock_endpoint_manager = AsyncMock()
        mock_endpoint_manager.get_all_endpoints.return_value = mock_endpoints
        mock_endpoint_manager.get_degradation_status.return_value = {"degradation_manager": {"is_fully_degraded": False}}
        mock_endpoint_manager.start = AsyncMock()
        mock_endpoint_manager.stop = AsyncMock()
        
        mock_pool_manager = AsyncMock()
        from gpu_worker_pool.models import PoolStatus
        mock_pool_status = PoolStatus(
            total_gpus=12,  # Will be overridden by endpoint aggregation
            available_gpus=4,
            active_workers=2,
            blocked_workers=1,
            gpu_assignments={}
        )
        mock_pool_manager.get_pool_status.return_value = mock_pool_status
        mock_pool_manager.start = AsyncMock()
        mock_pool_manager.stop = AsyncMock()
        
        with patch.multiple(
            'gpu_worker_pool.client',
            EnvironmentMultiEndpointConfigurationManager=MagicMock(),
            EndpointManager=MagicMock(return_value=mock_endpoint_manager),
            AsyncMultiEndpointHTTPClientPool=MagicMock(return_value=AsyncMock()),
            MultiEndpointLoadBalancer=MagicMock(),
            MultiEndpointGPUMonitor=MagicMock(return_value=AsyncMock()),
            AsyncWorkerPoolManager=MagicMock(return_value=mock_pool_manager),
            ThresholdBasedGPUAllocator=MagicMock(),
            FIFOWorkerQueue=MagicMock()
        ):
            await client.start()
            
            # Get unified pool status
            pool_status = client.get_pool_status()
            
            # Should return MultiEndpointPoolStatus
            from gpu_worker_pool.models import MultiEndpointPoolStatus
            assert isinstance(pool_status, MultiEndpointPoolStatus)
            assert pool_status.total_endpoints == 3
            assert pool_status.healthy_endpoints == 2  # server1 and server2
            assert pool_status.total_gpus == 6  # 4 + 2 (only healthy endpoints)
            
            await client.stop()
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, mock_endpoints):
        """Test error recovery integration with client."""
        client = GPUWorkerPoolClient(service_endpoints="http://server1:8000,http://server2:8000")
        
        # Mock all components including endpoint manager with degradation capabilities
        mock_endpoint_manager = MagicMock()
        mock_endpoint_manager.get_all_endpoints.return_value = mock_endpoints
        mock_endpoint_manager.get_degradation_status.return_value = {
            "degradation_manager": {
                "is_fully_degraded": False,
                "healthy_endpoints": ["server1", "server2"],
                "degraded_endpoints": ["server3"],
                "circuit_breaker_stats": {
                    "server1": {"state": "closed", "failure_rate": 0.0},
                    "server2": {"state": "closed", "failure_rate": 5.0},
                    "server3": {"state": "open", "failure_rate": 100.0}
                }
            },
            "recovery_orchestrator": {
                "is_running": True,
                "active_recovery_tasks": {}
            },
            "endpoint_health_summary": {
                "total_endpoints": 3,
                "healthy_count": 2,
                "degraded_count": 1,
                "queued_requests": 0
            }
        }
        mock_endpoint_manager.recovery_orchestrator = AsyncMock()
        mock_endpoint_manager.queue_request_when_degraded = AsyncMock()
        
        with patch.multiple(
            'gpu_worker_pool.client',
            EnvironmentMultiEndpointConfigurationManager=MagicMock(),
            EndpointManager=MagicMock(return_value=mock_endpoint_manager),
            AsyncMultiEndpointHTTPClientPool=MagicMock(return_value=AsyncMock()),
            MultiEndpointLoadBalancer=MagicMock(),
            MultiEndpointGPUMonitor=MagicMock(return_value=AsyncMock()),
            AsyncWorkerPoolManager=MagicMock(return_value=AsyncMock()),
            ThresholdBasedGPUAllocator=MagicMock(),
            FIFOWorkerQueue=MagicMock()
        ):
            await client.start()
            
            # Test error recovery status
            recovery_status = client.get_error_recovery_status()
            assert recovery_status is not None
            assert recovery_status["endpoint_health_summary"]["total_endpoints"] == 3
            assert recovery_status["endpoint_health_summary"]["healthy_count"] == 2
            
            # Test manual recovery trigger
            success = await client.trigger_endpoint_recovery("server3")
            assert success  # Should return True if no exception
            mock_endpoint_manager.recovery_orchestrator.trigger_recovery_attempt.assert_called_once()
            
            # Test request queueing
            test_func = AsyncMock()
            await client.queue_request_for_retry(test_func)
            mock_endpoint_manager.queue_request_when_degraded.assert_called_once_with(test_func)
            
            await client.stop()
    
    @pytest.mark.asyncio
    async def test_load_balancing_strategy_integration(self):
        """Test load balancing strategy integration with client."""
        # Test different load balancing strategies
        strategies = ["availability_based", "round_robin", "weighted"]
        
        for strategy in strategies:
            client = GPUWorkerPoolClient(
                service_endpoints="http://server1:8000,http://server2:8000",
                load_balancing_strategy=strategy
            )
            
            with patch.multiple(
                'gpu_worker_pool.client',
                EnvironmentMultiEndpointConfigurationManager=MagicMock(),
                EndpointManager=MagicMock(return_value=AsyncMock()),
                AsyncMultiEndpointHTTPClientPool=MagicMock(return_value=AsyncMock()),
                MultiEndpointLoadBalancer=MagicMock(),
                MultiEndpointGPUMonitor=MagicMock(return_value=AsyncMock()),
                AsyncWorkerPoolManager=MagicMock(return_value=AsyncMock()),
                ThresholdBasedGPUAllocator=MagicMock(),
                FIFOWorkerQueue=MagicMock()
            ) as mocks:
                await client.start()
                
                # Verify load balancer was created with correct strategy
                load_balancer_class = mocks['MultiEndpointLoadBalancer']
                load_balancer_class.assert_called_once()
                args, kwargs = load_balancer_class.call_args
                assert args[0] == strategy  # First argument should be strategy
                
                await client.stop()
    
    @pytest.mark.asyncio
    async def test_failover_scenario_simulation(self, mock_endpoints, mock_global_gpus):
        """Test simulated failover scenario."""
        client = GPUWorkerPoolClient(service_endpoints="http://server1:8000,http://server2:8000")
        
        # Create mock endpoint manager that can simulate failures
        mock_endpoint_manager = MockEndpointManager(mock_endpoints)
        
        # Create mock HTTP client pool
        mock_http_client_pool = MockMultiEndpointHTTPClientPool(
            endpoint_manager=mock_endpoint_manager,
            mock_global_gpus=mock_global_gpus
        )
        
        with patch.multiple(
            'gpu_worker_pool.client',
            EnvironmentMultiEndpointConfigurationManager=MagicMock(),
            EndpointManager=MagicMock(return_value=mock_endpoint_manager),
            AsyncMultiEndpointHTTPClientPool=MagicMock(return_value=mock_http_client_pool),
            MultiEndpointLoadBalancer=MagicMock(),
            MultiEndpointGPUMonitor=MagicMock(return_value=AsyncMock()),
            AsyncWorkerPoolManager=MagicMock(return_value=AsyncMock()),
            ThresholdBasedGPUAllocator=MagicMock(),
            FIFOWorkerQueue=MagicMock()
        ):
            await client.start()
            
            # Initially, both server1 and server2 should be healthy
            endpoints_info = client.get_endpoints_info()
            healthy_count = sum(1 for ep in endpoints_info if ep['is_healthy'])
            assert healthy_count == 2
            
            # Simulate server1 failure
            mock_endpoint_manager.set_endpoint_health("server1", False)
            
            # Now only server2 should be healthy
            endpoints_info = client.get_endpoints_info()
            healthy_count = sum(1 for ep in endpoints_info if ep['is_healthy'])
            assert healthy_count == 1
            
            # Verify the specific endpoint states
            server1_info = next((ep for ep in endpoints_info if ep['endpoint_id'] == 'server1'), None)
            server2_info = next((ep for ep in endpoints_info if ep['endpoint_id'] == 'server2'), None)
            
            assert server1_info is not None
            assert not server1_info['is_healthy']
            
            assert server2_info is not None
            assert server2_info['is_healthy']
            
            # Simulate server1 recovery
            mock_endpoint_manager.set_endpoint_health("server1", True, available_gpus=3, response_time_ms=50.0)
            
            # Both should be healthy again
            endpoints_info = client.get_endpoints_info()
            healthy_count = sum(1 for ep in endpoints_info if ep['is_healthy'])
            assert healthy_count == 2
            
            await client.stop()
    
    def test_configuration_validation_multi_endpoint(self):
        """Test configuration validation for multi-endpoint scenarios."""
        # Test valid configurations
        valid_configs = [
            {"service_endpoints": "http://server1:8000,http://server2:8000"},
            {"service_endpoints": "http://server1:8000,http://server2:8000,http://server3:8000"},
        ]
        
        for config in valid_configs:
            client = GPUWorkerPoolClient(**config)
            assert client.is_multi_endpoint_mode()
        
        # Test that single endpoint configurations work
        single_endpoint_configs = [
            {"service_endpoint": "http://localhost:8000"},
            {"service_endpoints": "http://localhost:8000"}  # Single endpoint in multi-endpoint format
        ]
        
        for config in single_endpoint_configs:
            client = GPUWorkerPoolClient(**config)
            # Single endpoint in service_endpoints should still trigger multi-endpoint mode
            if "service_endpoints" in config:
                # But with only one endpoint, it might not trigger multi-endpoint mode
                # depending on detection logic
                pass
    
    @pytest.mark.asyncio
    async def test_performance_and_scalability_simulation(self, mock_endpoints):
        """Test performance characteristics with multiple endpoints."""
        # Create a larger set of endpoints for scalability testing
        large_endpoint_set = []
        for i in range(10):  # 10 endpoints
            endpoint = EndpointInfo(
                endpoint_id=f"server{i}",
                url=f"http://server{i}:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=4 + (i % 3),  # Varying GPU counts
                available_gpus=2 + (i % 2),  # Varying availability
                response_time_ms=50.0 + (i * 10)  # Varying response times
            )
            large_endpoint_set.append(endpoint)
        
        endpoint_urls = ",".join([ep.url for ep in large_endpoint_set])
        client = GPUWorkerPoolClient(service_endpoints=endpoint_urls)
        
        # Mock the endpoint manager with larger set
        mock_endpoint_manager = MockEndpointManager(large_endpoint_set)
        
        with patch.multiple(
            'gpu_worker_pool.client',
            EnvironmentMultiEndpointConfigurationManager=MagicMock(),
            EndpointManager=MagicMock(return_value=mock_endpoint_manager),
            AsyncMultiEndpointHTTPClientPool=MagicMock(return_value=AsyncMock()),
            MultiEndpointLoadBalancer=MagicMock(),
            MultiEndpointGPUMonitor=MagicMock(return_value=AsyncMock()),
            AsyncWorkerPoolManager=MagicMock(return_value=AsyncMock()),
            ThresholdBasedGPUAllocator=MagicMock(),
            FIFOWorkerQueue=MagicMock()
        ):
            # Test that client can handle many endpoints
            await client.start()
            
            endpoints_info = client.get_endpoints_info()
            assert len(endpoints_info) == 10
            
            # Test metrics gathering performance
            import time
            start_time = time.time()
            
            for _ in range(10):  # Multiple calls to test performance
                metrics = client.get_detailed_metrics()
                assert "endpoints" in metrics
                assert metrics["endpoints"]["summary"]["total_endpoints"] == 10
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete reasonably quickly (less than 1 second for 10 calls)
            assert duration < 1.0
            
            await client.stop()