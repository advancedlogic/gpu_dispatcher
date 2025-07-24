"""Tests for endpoint manager functionality."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from gpu_worker_pool.endpoint_manager import AsyncEndpointManager, MockEndpointManager
from gpu_worker_pool.config import EnvironmentMultiEndpointConfigurationManager
from gpu_worker_pool.models import EndpointInfo, GPUStats, GPUInfo
from gpu_worker_pool.error_recovery import CircuitState


class TestAsyncEndpointManager:
    """Test cases for AsyncEndpointManager class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock multi-endpoint configuration."""
        config = MagicMock(spec=EnvironmentMultiEndpointConfigurationManager)
        config.get_service_endpoints.return_value = [
            "http://server1:8000",
            "http://server2:8000",
            "http://server3:8000"
        ]
        config.get_endpoint_timeout.return_value = 10.0
        config.get_endpoint_max_retries.return_value = 3
        config.get_utilization_threshold.return_value = 80.0
        config.get_memory_threshold.return_value = 90.0
        return config
    
    @pytest.fixture
    def endpoint_manager(self, mock_config):
        """Create endpoint manager with mock configuration."""
        return AsyncEndpointManager(mock_config)
    
    def test_endpoint_manager_initialization(self, mock_config):
        """Test endpoint manager initialization."""
        manager = AsyncEndpointManager(mock_config)
        
        assert manager.config == mock_config
        assert not manager._is_running
        assert len(manager._endpoints) == 3
        assert len(manager._http_clients) == 0  # Created during start()
        
        # Check that endpoints were initialized correctly
        endpoint_ids = list(manager._endpoints.keys())
        assert "server1_8000" in endpoint_ids
        assert "server2_8000" in endpoint_ids
        assert "server3_8000" in endpoint_ids
    
    def test_get_all_endpoints(self, endpoint_manager):
        """Test getting all endpoints."""
        endpoints = endpoint_manager.get_all_endpoints()
        
        assert len(endpoints) == 3
        assert all(isinstance(ep, EndpointInfo) for ep in endpoints)
        assert all(not ep.is_healthy for ep in endpoints)  # Initially unhealthy
    
    def test_get_healthy_endpoints_initially_empty(self, endpoint_manager):
        """Test getting healthy endpoints when none are healthy."""
        healthy = endpoint_manager.get_healthy_endpoints()
        assert len(healthy) == 0
    
    def test_get_endpoint_by_id(self, endpoint_manager):
        """Test getting endpoint by ID."""
        endpoint_ids = list(endpoint_manager._endpoints.keys())
        first_endpoint_id = endpoint_ids[0]
        
        endpoint = endpoint_manager.get_endpoint_by_id(first_endpoint_id)
        assert endpoint is not None
        assert endpoint.endpoint_id == first_endpoint_id
        
        # Test non-existent endpoint
        non_existent = endpoint_manager.get_endpoint_by_id("non-existent")
        assert non_existent is None
    
    def test_is_endpoint_healthy(self, endpoint_manager):
        """Test checking endpoint health status."""
        endpoint_ids = list(endpoint_manager._endpoints.keys())
        first_endpoint_id = endpoint_ids[0]
        
        # Initially unhealthy
        assert not endpoint_manager.is_endpoint_healthy(first_endpoint_id)
        
        # Mark as healthy
        endpoint_manager._endpoints[first_endpoint_id].is_healthy = True
        assert endpoint_manager.is_endpoint_healthy(first_endpoint_id)
        
        # Test non-existent endpoint
        assert not endpoint_manager.is_endpoint_healthy("non-existent")
    
    @pytest.mark.asyncio
    async def test_start_and_stop_lifecycle(self, endpoint_manager):
        """Test endpoint manager start and stop lifecycle."""
        assert not endpoint_manager._is_running
        
        with patch('gpu_worker_pool.endpoint_manager.AsyncGPUStatsHTTPClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock the health check to avoid actual network calls
            with patch.object(endpoint_manager, '_perform_health_checks', new_callable=AsyncMock):
                await endpoint_manager.start()
                
                assert endpoint_manager._is_running
                assert len(endpoint_manager._http_clients) == 3
                assert endpoint_manager._health_check_task is not None
                
                await endpoint_manager.stop()
                
                assert not endpoint_manager._is_running
                assert len(endpoint_manager._http_clients) == 0
                assert endpoint_manager._health_check_task is None
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, endpoint_manager):
        """Test successful health check."""
        endpoint_ids = list(endpoint_manager._endpoints.keys())
        first_endpoint_id = endpoint_ids[0]
        
        # Create mock client and GPU stats
        mock_client = AsyncMock()
        mock_gpu_stats = GPUStats(
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
        mock_client.fetch_gpu_stats.return_value = mock_gpu_stats
        
        endpoint_manager._http_clients[first_endpoint_id] = mock_client
        
        # Perform health check
        result = await endpoint_manager._perform_raw_health_check(first_endpoint_id, mock_client)
        
        assert result is True
        
        # Check that endpoint was marked as healthy
        endpoint = endpoint_manager._endpoints[first_endpoint_id]
        assert endpoint.is_healthy
        assert endpoint.total_gpus == 4
        assert endpoint.available_gpus == 4  # All GPUs are below utilization/memory thresholds
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, endpoint_manager):
        """Test failed health check."""
        endpoint_ids = list(endpoint_manager._endpoints.keys())
        first_endpoint_id = endpoint_ids[0]
        
        # Create mock client that raises exception
        mock_client = AsyncMock()
        mock_client.fetch_gpu_stats.side_effect = Exception("Connection failed")
        
        endpoint_manager._http_clients[first_endpoint_id] = mock_client
        
        # Health check should not raise but return False through circuit breaker
        result = await endpoint_manager._check_endpoint_health(first_endpoint_id)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, endpoint_manager):
        """Test circuit breaker integration in health checks."""
        endpoint_ids = list(endpoint_manager._endpoints.keys())
        first_endpoint_id = endpoint_ids[0]
        
        # Register endpoint with degradation manager
        endpoint_manager.degradation_manager.register_endpoint(first_endpoint_id)
        
        # Create failing mock client
        mock_client = AsyncMock()
        mock_client.fetch_gpu_stats.side_effect = Exception("Service unavailable")
        endpoint_manager._http_clients[first_endpoint_id] = mock_client
        
        # Perform multiple failed health checks to open circuit
        for _ in range(endpoint_manager.degradation_manager.config.failure_threshold):
            result = await endpoint_manager._check_endpoint_health(first_endpoint_id)
            assert result is False
        
        # Check circuit breaker state
        circuit_breaker = endpoint_manager.degradation_manager.circuit_breakers[first_endpoint_id]
        assert circuit_breaker.state == CircuitState.OPEN
    
    def test_degradation_status(self, endpoint_manager):
        """Test getting degradation status."""
        # Register some endpoints
        endpoint_manager.degradation_manager.register_endpoint("endpoint1")
        endpoint_manager.degradation_manager.register_endpoint("endpoint2")
        
        status = endpoint_manager.get_degradation_status()
        
        assert "degradation_manager" in status
        assert "recovery_orchestrator" in status
        assert "endpoint_health_summary" in status
        
        health_summary = status["endpoint_health_summary"]
        assert "total_endpoints" in health_summary
        assert "healthy_count" in health_summary
        assert "degraded_count" in health_summary
    
    @pytest.mark.asyncio
    async def test_endpoint_recovery_attempt(self, endpoint_manager):
        """Test endpoint recovery attempt."""
        endpoint_ids = list(endpoint_manager._endpoints.keys())
        first_endpoint_id = endpoint_ids[0]
        
        # Mock successful recovery
        with patch.object(endpoint_manager, '_check_endpoint_health', new_callable=AsyncMock) as mock_health_check:
            mock_health_check.return_value = True
            
            # Attempt recovery
            await endpoint_manager._attempt_endpoint_recovery(first_endpoint_id)
            
            mock_health_check.assert_called_once_with(first_endpoint_id)
    
    @pytest.mark.asyncio
    async def test_endpoint_recovery_failure(self, endpoint_manager):
        """Test endpoint recovery failure."""
        endpoint_ids = list(endpoint_manager._endpoints.keys())
        first_endpoint_id = endpoint_ids[0]
        
        # Mock failed recovery
        with patch.object(endpoint_manager, '_check_endpoint_health', new_callable=AsyncMock) as mock_health_check:
            mock_health_check.return_value = False
            
            # Recovery should raise exception
            with pytest.raises(Exception):
                await endpoint_manager._attempt_endpoint_recovery(first_endpoint_id)
    
    @pytest.mark.asyncio
    async def test_degradation_callback(self, endpoint_manager):
        """Test degradation callback handling."""
        endpoint_ids = list(endpoint_manager._endpoints.keys())
        first_endpoint_id = endpoint_ids[0]
        
        # Initially healthy
        endpoint_manager._endpoints[first_endpoint_id].is_healthy = True
        
        # Mock recovery orchestrator
        with patch.object(endpoint_manager.recovery_orchestrator, 'trigger_recovery_attempt', new_callable=AsyncMock) as mock_trigger:
            await endpoint_manager._on_endpoint_degradation(first_endpoint_id, "Test error")
            
            # Should mark endpoint as unhealthy
            assert not endpoint_manager._endpoints[first_endpoint_id].is_healthy
            
            # Should trigger recovery attempt
            mock_trigger.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_recovery_callback(self, endpoint_manager):
        """Test recovery callback handling."""
        endpoint_ids = list(endpoint_manager._endpoints.keys())
        first_endpoint_id = endpoint_ids[0]
        
        # Mock health check method
        with patch.object(endpoint_manager, '_check_endpoint_health', new_callable=AsyncMock) as mock_health_check:
            mock_health_check.return_value = True
            await endpoint_manager._on_endpoint_recovery(first_endpoint_id)
            
            mock_health_check.assert_called_once_with(first_endpoint_id)


class TestMockEndpointManager:
    """Test cases for MockEndpointManager class."""
    
    def test_mock_endpoint_manager_initialization(self):
        """Test mock endpoint manager initialization."""
        mock_endpoints = [
            EndpointInfo(
                endpoint_id="test1",
                url="http://test1:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=2,
                response_time_ms=50.0
            ),
            EndpointInfo(
                endpoint_id="test2",
                url="http://test2:8000",
                is_healthy=False,
                last_seen=datetime.now() - timedelta(minutes=5),
                total_gpus=2,
                available_gpus=0,
                response_time_ms=0.0
            )
        ]
        
        manager = MockEndpointManager(mock_endpoints)
        
        assert len(manager.mock_endpoints) == 2
        assert not manager._is_running
        assert manager.start_call_count == 0
        assert manager.stop_call_count == 0
    
    @pytest.mark.asyncio
    async def test_mock_lifecycle(self):
        """Test mock endpoint manager lifecycle."""
        manager = MockEndpointManager()
        
        assert not manager._is_running
        
        await manager.start()
        assert manager._is_running
        assert manager.start_call_count == 1
        
        await manager.stop()
        assert not manager._is_running
        assert manager.stop_call_count == 1
    
    def test_mock_get_healthy_endpoints(self):
        """Test getting healthy endpoints from mock."""
        mock_endpoints = [
            EndpointInfo(
                endpoint_id="healthy1",
                url="http://test1:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=2,
                response_time_ms=50.0
            ),
            EndpointInfo(
                endpoint_id="unhealthy1",
                url="http://test2:8000",
                is_healthy=False,
                last_seen=datetime.now(),
                total_gpus=2,
                available_gpus=0,
                response_time_ms=0.0
            ),
            EndpointInfo(
                endpoint_id="healthy2",
                url="http://test3:8000",
                is_healthy=True,  
                last_seen=datetime.now(),
                total_gpus=8,
                available_gpus=4,
                response_time_ms=75.0
            )
        ]
        
        manager = MockEndpointManager(mock_endpoints)
        
        healthy = manager.get_healthy_endpoints()
        assert len(healthy) == 2
        assert all(ep.is_healthy for ep in healthy)
        assert {ep.endpoint_id for ep in healthy} == {"healthy1", "healthy2"}
    
    def test_mock_get_all_endpoints(self):
        """Test getting all endpoints from mock."""
        mock_endpoints = [
            EndpointInfo(
                endpoint_id="test1",
                url="http://test1:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=2,
                response_time_ms=50.0
            )
        ]
        
        manager = MockEndpointManager(mock_endpoints)
        
        all_endpoints = manager.get_all_endpoints()
        assert len(all_endpoints) == 1
        assert all_endpoints[0].endpoint_id == "test1"
    
    def test_mock_get_endpoint_by_id(self):
        """Test getting endpoint by ID from mock."""
        mock_endpoints = [
            EndpointInfo(
                endpoint_id="test1",
                url="http://test1:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=2,
                response_time_ms=50.0
            )
        ]
        
        manager = MockEndpointManager(mock_endpoints)
        
        endpoint = manager.get_endpoint_by_id("test1")
        assert endpoint is not None
        assert endpoint.endpoint_id == "test1"
        
        non_existent = manager.get_endpoint_by_id("non-existent")
        assert non_existent is None
    
    def test_mock_is_endpoint_healthy(self):
        """Test checking endpoint health in mock."""
        mock_endpoints = [
            EndpointInfo(
                endpoint_id="healthy",
                url="http://test1:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=2,
                response_time_ms=50.0
            ),
            EndpointInfo(
                endpoint_id="unhealthy",
                url="http://test2:8000",
                is_healthy=False,
                last_seen=datetime.now(),
                total_gpus=2,
                available_gpus=0,
                response_time_ms=0.0
            )
        ]
        
        manager = MockEndpointManager(mock_endpoints)
        
        assert manager.is_endpoint_healthy("healthy")
        assert not manager.is_endpoint_healthy("unhealthy")
        assert not manager.is_endpoint_healthy("non-existent")
    
    def test_mock_set_endpoint_health(self):
        """Test setting endpoint health in mock."""
        mock_endpoints = [
            EndpointInfo(
                endpoint_id="test1",
                url="http://test1:8000",
                is_healthy=False,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=0,
                response_time_ms=0.0
            )
        ]
        
        manager = MockEndpointManager(mock_endpoints)
        
        assert not manager.is_endpoint_healthy("test1")
        
        manager.set_endpoint_health("test1", True, available_gpus=2, response_time_ms=50.0)
        
        endpoint = manager.get_endpoint_by_id("test1")
        assert endpoint.is_healthy
        assert endpoint.available_gpus == 2
        assert endpoint.response_time_ms == 50.0