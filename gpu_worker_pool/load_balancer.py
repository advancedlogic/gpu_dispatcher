"""Load balancing strategies for multi-endpoint GPU Worker Pool."""

import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import EndpointInfo, GlobalGPUInfo
from .endpoint_manager import EndpointManager

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(ABC):
    """Abstract base class for load balancing strategies."""
    
    @abstractmethod
    def select_endpoint_for_gpu_request(self, healthy_endpoints: List[EndpointInfo]) -> Optional[EndpointInfo]:
        """Select the best endpoint for a GPU request.
        
        Args:
            healthy_endpoints: List of healthy endpoints to choose from
            
        Returns:
            Selected endpoint or None if no suitable endpoint found
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this load balancing strategy."""
        pass
    
    @abstractmethod
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get metrics specific to this load balancing strategy."""
        pass


class AvailabilityBasedLoadBalancer(LoadBalancingStrategy):
    """Load balancer that prioritizes endpoints with higher GPU availability."""
    
    def __init__(self):
        """Initialize availability-based load balancer."""
        self._selection_history: List[str] = []
        self._selection_counts: Dict[str, int] = {}
        
    def select_endpoint_for_gpu_request(self, healthy_endpoints: List[EndpointInfo]) -> Optional[EndpointInfo]:
        """Select endpoint with highest GPU availability.
        
        Prioritizes endpoints with:
        1. Higher percentage of available GPUs
        2. Higher absolute number of available GPUs
        3. Lower response time as tiebreaker
        
        Args:
            healthy_endpoints: List of healthy endpoints to choose from
            
        Returns:
            Selected endpoint or None if no suitable endpoint found
        """
        if not healthy_endpoints:
            logger.warning("No healthy endpoints available for selection")
            return None
        
        # Filter endpoints that have available GPUs
        available_endpoints = [ep for ep in healthy_endpoints if ep.available_gpus > 0]
        
        if not available_endpoints:
            logger.warning("No endpoints with available GPUs")
            return None
        
        # Sort by availability criteria
        def availability_score(endpoint: EndpointInfo) -> tuple:
            # Calculate availability percentage
            availability_percent = (endpoint.available_gpus / endpoint.total_gpus * 100) if endpoint.total_gpus > 0 else 0
            
            # Return tuple for sorting (higher availability percent, higher absolute count, lower response time)
            # Negative values for descending sort, positive for ascending
            return (-availability_percent, -endpoint.available_gpus, endpoint.response_time_ms)
        
        # Sort endpoints by availability score
        sorted_endpoints = sorted(available_endpoints, key=availability_score)
        selected_endpoint = sorted_endpoints[0]
        
        # Update selection tracking
        self._selection_history.append(selected_endpoint.endpoint_id)
        self._selection_counts[selected_endpoint.endpoint_id] = self._selection_counts.get(selected_endpoint.endpoint_id, 0) + 1
        
        # Keep history limited
        if len(self._selection_history) > 100:
            self._selection_history = self._selection_history[-100:]
        
        logger.debug(f"Selected endpoint {selected_endpoint.endpoint_id} with {selected_endpoint.available_gpus}/{selected_endpoint.total_gpus} available GPUs")
        return selected_endpoint
    
    def get_strategy_name(self) -> str:
        """Get the name of this load balancing strategy."""
        return "availability"
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get metrics specific to availability-based load balancing."""
        total_selections = len(self._selection_history)
        
        return {
            "strategy_name": self.get_strategy_name(),
            "total_selections": total_selections,
            "selection_counts": self._selection_counts.copy(),
            "recent_selections": self._selection_history[-10:] if self._selection_history else []
        }


class RoundRobinLoadBalancer(LoadBalancingStrategy):
    """Load balancer that distributes requests evenly across healthy endpoints."""
    
    def __init__(self):
        """Initialize round-robin load balancer."""
        self._current_index = 0
        self._selection_history: List[str] = []
        self._selection_counts: Dict[str, int] = {}
        self._endpoint_order: List[str] = []
        
    def select_endpoint_for_gpu_request(self, healthy_endpoints: List[EndpointInfo]) -> Optional[EndpointInfo]:
        """Select next endpoint in round-robin order.
        
        Only considers endpoints with available GPUs. If the current round-robin
        endpoint has no available GPUs, continues to the next one.
        
        Args:
            healthy_endpoints: List of healthy endpoints to choose from
            
        Returns:
            Selected endpoint or None if no suitable endpoint found
        """
        if not healthy_endpoints:
            logger.warning("No healthy endpoints available for selection")
            return None
        
        # Filter endpoints that have available GPUs
        available_endpoints = [ep for ep in healthy_endpoints if ep.available_gpus > 0]
        
        if not available_endpoints:
            logger.warning("No endpoints with available GPUs")
            return None
        
        # Update endpoint order if it has changed
        current_endpoint_ids = [ep.endpoint_id for ep in available_endpoints]
        if self._endpoint_order != current_endpoint_ids:
            self._endpoint_order = current_endpoint_ids
            self._current_index = 0  # Reset index when endpoints change
        
        # Select endpoint using round-robin
        if self._current_index >= len(available_endpoints):
            self._current_index = 0
        
        selected_endpoint = available_endpoints[self._current_index]
        self._current_index = (self._current_index + 1) % len(available_endpoints)
        
        # Update selection tracking
        self._selection_history.append(selected_endpoint.endpoint_id)
        self._selection_counts[selected_endpoint.endpoint_id] = self._selection_counts.get(selected_endpoint.endpoint_id, 0) + 1
        
        # Keep history limited
        if len(self._selection_history) > 100:
            self._selection_history = self._selection_history[-100:]
        
        logger.debug(f"Round-robin selected endpoint {selected_endpoint.endpoint_id} (index {self._current_index - 1})")
        return selected_endpoint
    
    def get_strategy_name(self) -> str:
        """Get the name of this load balancing strategy."""
        return "round_robin"
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get metrics specific to round-robin load balancing."""
        total_selections = len(self._selection_history)
        
        return {
            "strategy_name": self.get_strategy_name(),
            "total_selections": total_selections,
            "current_index": self._current_index,
            "endpoint_order": self._endpoint_order.copy(),
            "selection_counts": self._selection_counts.copy(),
            "recent_selections": self._selection_history[-10:] if self._selection_history else []
        }


class WeightedLoadBalancer(LoadBalancingStrategy):
    """Load balancer that considers total GPU capacity of each endpoint."""
    
    def __init__(self):
        """Initialize weighted load balancer."""
        self._selection_history: List[str] = []
        self._selection_counts: Dict[str, int] = {}
        self._endpoint_weights: Dict[str, float] = {}
        
    def select_endpoint_for_gpu_request(self, healthy_endpoints: List[EndpointInfo]) -> Optional[EndpointInfo]:
        """Select endpoint based on weighted random selection.
        
        Endpoints with more total GPUs have higher probability of being selected.
        Only considers endpoints with available GPUs.
        
        Args:
            healthy_endpoints: List of healthy endpoints to choose from
            
        Returns:
            Selected endpoint or None if no suitable endpoint found
        """
        if not healthy_endpoints:
            logger.warning("No healthy endpoints available for selection")
            return None
        
        # Filter endpoints that have available GPUs
        available_endpoints = [ep for ep in healthy_endpoints if ep.available_gpus > 0]
        
        if not available_endpoints:
            logger.warning("No endpoints with available GPUs")
            return None
        
        # Calculate weights based on total GPU capacity
        total_capacity = sum(ep.total_gpus for ep in available_endpoints)
        if total_capacity == 0:
            # Fallback to equal weights if no capacity information
            weights = [1.0] * len(available_endpoints)
        else:
            weights = [ep.total_gpus / total_capacity for ep in available_endpoints]
        
        # Update endpoint weights for metrics
        for i, endpoint in enumerate(available_endpoints):
            self._endpoint_weights[endpoint.endpoint_id] = weights[i]
        
        # Weighted random selection
        selected_endpoint = self._weighted_random_choice(available_endpoints, weights)
        
        # Update selection tracking
        self._selection_history.append(selected_endpoint.endpoint_id)
        self._selection_counts[selected_endpoint.endpoint_id] = self._selection_counts.get(selected_endpoint.endpoint_id, 0) + 1
        
        # Keep history limited
        if len(self._selection_history) > 100:
            self._selection_history = self._selection_history[-100:]
        
        logger.debug(f"Weighted selection chose endpoint {selected_endpoint.endpoint_id} "
                    f"(weight: {self._endpoint_weights.get(selected_endpoint.endpoint_id, 0):.3f})")
        return selected_endpoint
    
    def _weighted_random_choice(self, endpoints: List[EndpointInfo], weights: List[float]) -> EndpointInfo:
        """Perform weighted random selection.
        
        Args:
            endpoints: List of endpoints to choose from
            weights: List of weights corresponding to endpoints
            
        Returns:
            Selected endpoint
        """
        if len(endpoints) != len(weights):
            raise ValueError("Endpoints and weights lists must have the same length")
        
        # Cumulative weights for selection
        cumulative_weights = []
        cumulative_sum = 0.0
        for weight in weights:
            cumulative_sum += weight
            cumulative_weights.append(cumulative_sum)
        
        # Random selection
        random_value = random.random() * cumulative_sum
        
        for i, cumulative_weight in enumerate(cumulative_weights):
            if random_value <= cumulative_weight:
                return endpoints[i]
        
        # Fallback to last endpoint (shouldn't happen with proper weights)
        return endpoints[-1]
    
    def get_strategy_name(self) -> str:
        """Get the name of this load balancing strategy."""
        return "weighted"
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get metrics specific to weighted load balancing."""
        total_selections = len(self._selection_history)
        
        return {
            "strategy_name": self.get_strategy_name(),
            "total_selections": total_selections,
            "endpoint_weights": self._endpoint_weights.copy(),
            "selection_counts": self._selection_counts.copy(),
            "recent_selections": self._selection_history[-10:] if self._selection_history else []
        }


class LoadBalancerFactory:
    """Factory for creating load balancing strategy instances."""
    
    _strategies = {
        "availability": AvailabilityBasedLoadBalancer,
        "round_robin": RoundRobinLoadBalancer,
        "weighted": WeightedLoadBalancer
    }
    
    @classmethod
    def create_load_balancer(cls, strategy_name: str) -> LoadBalancingStrategy:
        """Create a load balancing strategy instance.
        
        Args:
            strategy_name: Name of the strategy to create
            
        Returns:
            LoadBalancingStrategy instance
            
        Raises:
            ValueError: If strategy name is not supported
        """
        strategy_name = strategy_name.lower()
        
        if strategy_name not in cls._strategies:
            available_strategies = ", ".join(cls._strategies.keys())
            raise ValueError(f"Unknown load balancing strategy: {strategy_name}. "
                           f"Available strategies: {available_strategies}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class()
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available load balancing strategy names."""
        return list(cls._strategies.keys())


class MultiEndpointLoadBalancer:
    """Main load balancer that manages endpoint selection for GPU requests."""
    
    def __init__(self, endpoint_manager: EndpointManager, strategy_name: str = "availability"):
        """Initialize multi-endpoint load balancer.
        
        Args:
            endpoint_manager: Endpoint manager for accessing endpoint information
            strategy_name: Name of the load balancing strategy to use
        """
        self.endpoint_manager = endpoint_manager
        self.strategy = LoadBalancerFactory.create_load_balancer(strategy_name)
        self._total_requests = 0
        self._successful_selections = 0
        self._failed_selections = 0
        
        logger.info(f"Initialized multi-endpoint load balancer with {strategy_name} strategy")
    
    def select_endpoint_for_gpu_request(self) -> Optional[EndpointInfo]:
        """Select the best endpoint for a GPU request.
        
        Returns:
            Selected endpoint or None if no suitable endpoint available
        """
        self._total_requests += 1
        
        healthy_endpoints = self.endpoint_manager.get_healthy_endpoints()
        
        try:
            selected_endpoint = self.strategy.select_endpoint_for_gpu_request(healthy_endpoints)
            
            if selected_endpoint:
                self._successful_selections += 1
                logger.debug(f"Load balancer selected endpoint {selected_endpoint.endpoint_id}")
            else:
                self._failed_selections += 1
                logger.warning("Load balancer could not select any endpoint")
            
            return selected_endpoint
            
        except Exception as e:
            self._failed_selections += 1
            logger.error(f"Error in load balancer endpoint selection: {e}")
            return None
    
    def get_load_balancer_metrics(self) -> Dict[str, Any]:
        """Get comprehensive load balancer metrics.
        
        Returns:
            Dictionary containing load balancer statistics and strategy metrics
        """
        strategy_metrics = self.strategy.get_strategy_metrics()
        
        success_rate = (self._successful_selections / self._total_requests * 100) if self._total_requests > 0 else 0.0
        
        return {
            "load_balancer_summary": {
                "strategy_name": self.strategy.get_strategy_name(),
                "total_requests": self._total_requests,
                "successful_selections": self._successful_selections,
                "failed_selections": self._failed_selections,
                "success_rate": success_rate
            },
            "strategy_metrics": strategy_metrics,
            "endpoint_summary": {
                "total_endpoints": len(self.endpoint_manager.get_all_endpoints()),
                "healthy_endpoints": len(self.endpoint_manager.get_healthy_endpoints()),
                "endpoints_with_available_gpus": len([
                    ep for ep in self.endpoint_manager.get_healthy_endpoints() 
                    if ep.available_gpus > 0
                ])
            }
        }
    
    def get_strategy_name(self) -> str:
        """Get the name of the current load balancing strategy.
        
        Returns:
            Name of the current strategy
        """
        return self.strategy.get_strategy_name()
    
    def change_strategy(self, strategy_name: str) -> None:
        """Change the load balancing strategy.
        
        Args:
            strategy_name: Name of the new strategy to use
            
        Raises:
            ValueError: If strategy name is not supported
        """
        old_strategy_name = self.strategy.get_strategy_name()
        self.strategy = LoadBalancerFactory.create_load_balancer(strategy_name)
        
        logger.info(f"Changed load balancing strategy from {old_strategy_name} to {strategy_name}")


class MockLoadBalancer(LoadBalancingStrategy):
    """Mock load balancer for testing purposes."""
    
    def __init__(self, mock_selections: Optional[List[str]] = None):
        """Initialize mock load balancer.
        
        Args:
            mock_selections: List of endpoint IDs to return in order
        """
        self.mock_selections = mock_selections or []
        self._selection_index = 0
        self.selection_call_count = 0
        
    def select_endpoint_for_gpu_request(self, healthy_endpoints: List[EndpointInfo]) -> Optional[EndpointInfo]:
        """Mock endpoint selection."""
        self.selection_call_count += 1
        
        if not self.mock_selections or not healthy_endpoints:
            return None
        
        # Get the next mock selection
        if self._selection_index >= len(self.mock_selections):
            self._selection_index = 0
        
        target_endpoint_id = self.mock_selections[self._selection_index]
        self._selection_index += 1
        
        # Find the endpoint with matching ID
        for endpoint in healthy_endpoints:
            if endpoint.endpoint_id == target_endpoint_id:
                return endpoint
        
        # If not found, return first healthy endpoint
        return healthy_endpoints[0] if healthy_endpoints else None
    
    def get_strategy_name(self) -> str:
        """Get mock strategy name."""
        return "mock"
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get mock strategy metrics."""
        return {
            "strategy_name": "mock",
            "selection_call_count": self.selection_call_count,
            "mock_selections": self.mock_selections.copy(),
            "current_index": self._selection_index
        }