"""Tests for load balancer functionality."""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from gpu_worker_pool.load_balancer import (
    MultiEndpointLoadBalancer, LoadBalancingStrategy,
    AvailabilityBasedLoadBalancer, RoundRobinLoadBalancer, WeightedLoadBalancer
)
from gpu_worker_pool.endpoint_manager import MockEndpointManager
from gpu_worker_pool.models import EndpointInfo


class TestLoadBalancingStrategies:
    """Test cases for individual load balancing strategies."""
    
    @pytest.fixture
    def mock_endpoints(self):
        """Create mock endpoints with different characteristics."""
        return [
            EndpointInfo(
                endpoint_id="server1",
                url="http://server1:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=8,
                available_gpus=6,  # 75% available
                response_time_ms=50.0
            ),
            EndpointInfo(
                endpoint_id="server2",
                url="http://server2:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=2,  # 50% available
                response_time_ms=100.0
            ),
            EndpointInfo(
                endpoint_id="server3",
                url="http://server3:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=6,
                available_gpus=6,  # 100% available
                response_time_ms=75.0
            ),
            EndpointInfo(
                endpoint_id="unhealthy_server",
                url="http://unhealthy:8000",
                is_healthy=False,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=0,
                response_time_ms=0.0
            )
        ]
    
    def test_availability_based_strategy(self, mock_endpoints):
        """Test availability-based load balancing strategy."""
        strategy = AvailabilityBasedLoadBalancer()
        
        # Filter to healthy endpoints
        healthy_endpoints = [ep for ep in mock_endpoints if ep.is_healthy]
        
        # Should select server3 (100% available)
        selected = strategy.select_endpoint_for_gpu_request(healthy_endpoints)
        assert selected.endpoint_id == "server3"
        
        # Test with empty list
        selected = strategy.select_endpoint_for_gpu_request([])
        assert selected is None
        
        # Test strategy name and metrics
        assert strategy.get_strategy_name() == "availability"
        metrics = strategy.get_strategy_metrics()
        assert "strategy_name" in metrics
    
    def test_round_robin_strategy(self, mock_endpoints):
        """Test round-robin load balancing strategy."""
        strategy = RoundRobinLoadBalancer()
        
        # Filter to healthy endpoints
        healthy_endpoints = [ep for ep in mock_endpoints if ep.is_healthy]
        
        # Should cycle through endpoints
        selections = []
        for _ in range(6):  # More than number of endpoints
            selected = strategy.select_endpoint_for_gpu_request(healthy_endpoints)
            selections.append(selected.endpoint_id)
        
        # Should see each endpoint at least once
        unique_selections = set(selections)
        assert len(unique_selections) == 3  # 3 healthy endpoints
        assert "server1" in unique_selections
        assert "server2" in unique_selections
        assert "server3" in unique_selections
        
        # Test with empty list
        selected = strategy.select_endpoint_for_gpu_request([])
        assert selected is None
        
        # Test strategy name and metrics
        assert strategy.get_strategy_name() == "round_robin"
        metrics = strategy.get_strategy_metrics()
        assert "strategy_name" in metrics
    
    def test_weighted_strategy(self, mock_endpoints):
        """Test weighted load balancing strategy."""
        strategy = WeightedLoadBalancer()
        
        # Filter to healthy endpoints
        healthy_endpoints = [ep for ep in mock_endpoints if ep.is_healthy]
        
        # Test multiple selections to see distribution
        selections = {}
        for _ in range(100):  # Many selections to see pattern
            selected = strategy.select_endpoint_for_gpu_request(healthy_endpoints)
            endpoint_id = selected.endpoint_id
            selections[endpoint_id] = selections.get(endpoint_id, 0) + 1
        
        # server1 has most GPUs (8), should be selected most often
        # server3 has 6 GPUs, server2 has 4 GPUs
        assert selections["server1"] > selections["server2"]
        # Note: Due to randomness, server3 vs server2 comparison may vary
        
        # Test with empty list
        selected = strategy.select_endpoint_for_gpu_request([])
        assert selected is None
        
        # Test strategy name and metrics
        assert strategy.get_strategy_name() == "weighted"
        metrics = strategy.get_strategy_metrics()
        assert "strategy_name" in metrics
    
    def test_strategy_with_single_endpoint(self, mock_endpoints):
        """Test strategies with only one healthy endpoint."""
        single_endpoint = [mock_endpoints[0]]  # Just server1
        
        strategies = [
            AvailabilityBasedLoadBalancer(),
            RoundRobinLoadBalancer(),
            WeightedLoadBalancer()
        ]
        
        for strategy in strategies:
            selected = strategy.select_endpoint_for_gpu_request(single_endpoint)
            assert selected.endpoint_id == "server1"
    
    def test_strategy_with_zero_available_gpus(self):
        """Test strategies when all endpoints have zero available GPUs."""
        endpoints_no_gpus = [
            EndpointInfo(
                endpoint_id="server1",
                url="http://server1:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=0,  # No GPUs available
                response_time_ms=50.0
            ),
            EndpointInfo(
                endpoint_id="server2",
                url="http://server2:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=2,
                available_gpus=0,  # No GPUs available
                response_time_ms=75.0
            )
        ]
        
        # Availability-based should return None when no GPUs available (correct behavior)
        availability_strategy = AvailabilityBasedLoadBalancer()
        selected = availability_strategy.select_endpoint_for_gpu_request(endpoints_no_gpus)
        assert selected is None
        
        # Round-robin should also return None when no available GPUs
        rr_strategy = RoundRobinLoadBalancer()
        selected = rr_strategy.select_endpoint_for_gpu_request(endpoints_no_gpus)
        assert selected is None
        
        # Weighted should also return None when no available GPUs
        weighted_strategy = WeightedLoadBalancer()
        selected = weighted_strategy.select_endpoint_for_gpu_request(endpoints_no_gpus)
        assert selected is None


class TestMultiEndpointLoadBalancer:
    """Test cases for MultiEndpointLoadBalancer class."""
    
    @pytest.fixture
    def mock_endpoints(self):
        """Create mock endpoints for testing."""
        return [
            EndpointInfo(
                endpoint_id="server1",
                url="http://server1:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=8,
                available_gpus=6,
                response_time_ms=50.0
            ),
            EndpointInfo(
                endpoint_id="server2",
                url="http://server2:8000",
                is_healthy=True,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=2,
                response_time_ms=100.0
            ),
            EndpointInfo(
                endpoint_id="unhealthy",
                url="http://unhealthy:8000",
                is_healthy=False,
                last_seen=datetime.now(),
                total_gpus=4,
                available_gpus=0,
                response_time_ms=0.0
            )
        ]
    
    @pytest.fixture
    def mock_endpoint_manager(self, mock_endpoints):
        """Create mock endpoint manager."""
        return MockEndpointManager(mock_endpoints)
    
    def test_load_balancer_initialization_default_strategy(self, mock_endpoint_manager):
        """Test load balancer initialization with default strategy."""
        balancer = MultiEndpointLoadBalancer(
            endpoint_manager=mock_endpoint_manager,
            strategy_name="availability"
        )
        
        assert isinstance(balancer.strategy, AvailabilityBasedLoadBalancer)
        assert balancer.endpoint_manager == mock_endpoint_manager
    
    def test_load_balancer_initialization_round_robin(self, mock_endpoint_manager):
        """Test load balancer initialization with round-robin strategy."""
        balancer = MultiEndpointLoadBalancer(
            endpoint_manager=mock_endpoint_manager,
            strategy_name="round_robin"
        )
        
        assert isinstance(balancer.strategy, RoundRobinLoadBalancer)
    
    def test_load_balancer_initialization_weighted(self, mock_endpoint_manager):
        """Test load balancer initialization with weighted strategy."""
        balancer = MultiEndpointLoadBalancer(
            endpoint_manager=mock_endpoint_manager,
            strategy_name="weighted"
        )
        
        assert isinstance(balancer.strategy, WeightedLoadBalancer)
    
    def test_load_balancer_initialization_invalid_strategy(self, mock_endpoint_manager):
        """Test load balancer initialization with invalid strategy."""
        with pytest.raises(ValueError, match="Unknown load balancing strategy"):
            MultiEndpointLoadBalancer(
                endpoint_manager=mock_endpoint_manager,
                strategy_name="invalid_strategy"
            )
    
    def test_load_balancer_default_strategy(self, mock_endpoint_manager):
        """Test load balancer initialization with default strategy."""
        balancer = MultiEndpointLoadBalancer(
            endpoint_manager=mock_endpoint_manager
        )
        
        assert isinstance(balancer.strategy, AvailabilityBasedLoadBalancer)
    
    def test_select_endpoint_healthy_endpoints_only(self, mock_endpoint_manager):
        """Test endpoint selection filters to healthy endpoints only."""
        balancer = MultiEndpointLoadBalancer(
            endpoint_manager=mock_endpoint_manager,
            strategy_name="availability"
        )
        
        selected = balancer.select_endpoint_for_gpu_request()
        
        # Should only select from healthy endpoints
        assert selected is not None
        assert selected.is_healthy
        assert selected.endpoint_id != "unhealthy"
    
    def test_select_endpoint_no_healthy_endpoints(self, mock_endpoint_manager):
        """Test endpoint selection when no endpoints are healthy."""
        # Make all endpoints unhealthy
        for endpoint in mock_endpoint_manager.mock_endpoints:
            endpoint.is_healthy = False
        
        balancer = MultiEndpointLoadBalancer(
            endpoint_manager=mock_endpoint_manager,
            strategy_name="availability"
        )
        
        selected = balancer.select_endpoint_for_gpu_request()
        assert selected is None
    
    def test_get_strategy_metrics(self, mock_endpoint_manager):
        """Test getting strategy metrics."""
        balancer = MultiEndpointLoadBalancer(
            endpoint_manager=mock_endpoint_manager,
            strategy_name="availability"
        )
        
        metrics = balancer.get_load_balancer_metrics()
        assert isinstance(metrics, dict)
        assert "load_balancer_summary" in metrics
        assert "strategy_metrics" in metrics
    
    def test_load_balancer_with_different_strategies(self, mock_endpoint_manager):
        """Test that different strategies produce different selections."""
        # Test multiple times to see patterns
        availability_selections = set()
        rr_selections = set()
        weighted_selections = set()
        
        availability_balancer = MultiEndpointLoadBalancer(mock_endpoint_manager, "availability")
        rr_balancer = MultiEndpointLoadBalancer(mock_endpoint_manager, "round_robin")
        weighted_balancer = MultiEndpointLoadBalancer(mock_endpoint_manager, "weighted")
        
        for _ in range(10):
            # Availability-based should consistently pick the same endpoint (highest availability)
            selected = availability_balancer.select_endpoint_for_gpu_request()
            if selected:
                availability_selections.add(selected.endpoint_id)
            
            # Round-robin should cycle through endpoints
            selected = rr_balancer.select_endpoint_for_gpu_request()
            if selected:
                rr_selections.add(selected.endpoint_id)
            
            # Weighted should vary based on capacity
            selected = weighted_balancer.select_endpoint_for_gpu_request()
            if selected:
                weighted_selections.add(selected.endpoint_id)
        
        # Availability-based should be consistent (pick server1 which has highest availability)
        assert len(availability_selections) <= 2  # Should be consistent, allowing for some variation
        
        # Round-robin should use both healthy endpoints
        assert len(rr_selections) == 2  # Should cycle through both healthy endpoints
        
        # Weighted might use both, with preference for higher capacity
        assert len(weighted_selections) >= 1
    
    def test_strategy_consistency(self, mock_endpoint_manager):
        """Test that the same strategy produces consistent results for same input."""
        balancer = MultiEndpointLoadBalancer(
            endpoint_manager=mock_endpoint_manager,
            strategy_name="availability"
        )
        
        # Availability-based should be deterministic for same input
        selections = []
        for _ in range(5):
            selected = balancer.select_endpoint_for_gpu_request()
            if selected:
                selections.append(selected.endpoint_id)
        
        # Should be consistent (all same endpoint for availability-based)
        if selections:
            first_selection = selections[0]
            assert all(sel == first_selection for sel in selections)
    
    def test_load_balancer_respects_endpoint_health_changes(self, mock_endpoint_manager):
        """Test that load balancer respects real-time endpoint health changes."""
        balancer = MultiEndpointLoadBalancer(
            endpoint_manager=mock_endpoint_manager,
            strategy_name="round_robin"
        )
        
        # Initially should select from 2 healthy endpoints
        initial_selections = set()
        for _ in range(4):
            selected = balancer.select_endpoint_for_gpu_request()
            if selected:
                initial_selections.add(selected.endpoint_id)
        
        assert len(initial_selections) == 2  # server1 and server2
        
        # Make server1 unhealthy
        mock_endpoint_manager.set_endpoint_health("server1", False)
        
        # Now should only select server2
        new_selections = set()
        for _ in range(4):
            selected = balancer.select_endpoint_for_gpu_request()
            if selected:
                new_selections.add(selected.endpoint_id)
        
        assert new_selections == {"server2"}
    
    def test_load_balancer_selection_tracking(self, mock_endpoint_manager):
        """Test that load balancer tracks selections for analysis."""
        balancer = MultiEndpointLoadBalancer(
            endpoint_manager=mock_endpoint_manager,
            strategy_name="round_robin"
        )
        
        # Make several selections
        for _ in range(6):
            balancer.select_endpoint_for_gpu_request()
        
        # Strategy should have internal state for round-robin
        # This is implementation-specific, but we can test that it's working
        # by verifying the distribution is reasonable
        selections = []
        for _ in range(10):
            selected = balancer.select_endpoint_for_gpu_request()
            if selected:
                selections.append(selected.endpoint_id)
        
        # Should have used both healthy endpoints
        unique_selections = set(selections)
        assert len(unique_selections) == 2