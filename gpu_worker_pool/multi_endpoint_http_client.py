"""Multi-endpoint HTTP client pool for GPU Worker Pool."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from .models import (
    GPUStats, GlobalGPUInfo, EndpointInfo, 
    create_global_gpu_id, create_global_gpu_info_from_gpu_info,
    create_endpoint_id_from_url
)
from .http_client import AsyncGPUStatsHTTPClient, ServiceUnavailableError, RetryableError
from .endpoint_manager import EndpointManager

logger = logging.getLogger(__name__)


class MultiEndpointHTTPClientPool(ABC):
    """Abstract base class for multi-endpoint HTTP client pool."""
    
    @abstractmethod
    async def fetch_aggregated_gpu_stats(self) -> Optional[List[GlobalGPUInfo]]:
        """Fetch and aggregate GPU statistics from all healthy endpoints."""
        pass
    
    @abstractmethod
    async def fetch_gpu_stats_from_endpoint(self, endpoint_id: str) -> Optional[GPUStats]:
        """Fetch GPU statistics from a specific endpoint."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close all HTTP clients and cleanup resources."""
        pass


class AsyncMultiEndpointHTTPClientPool(MultiEndpointHTTPClientPool):
    """Async HTTP client pool for managing multiple GPU statistics endpoints."""
    
    def __init__(self, endpoint_manager: EndpointManager):
        """Initialize the multi-endpoint HTTP client pool.
        
        Args:
            endpoint_manager: Endpoint manager for health monitoring and client access
        """
        self.endpoint_manager = endpoint_manager
        self._request_stats: Dict[str, Dict[str, Any]] = {}
        self._last_successful_fetch: Dict[str, datetime] = {}
        
    async def fetch_aggregated_gpu_stats(self) -> Optional[List[GlobalGPUInfo]]:
        """Fetch and aggregate GPU statistics from all healthy endpoints.
        
        Returns:
            List of GlobalGPUInfo objects from all healthy endpoints, or None if no data available
        """
        healthy_endpoints = self.endpoint_manager.get_healthy_endpoints()
        
        if not healthy_endpoints:
            logger.warning("No healthy endpoints available for GPU stats aggregation")
            return None
        
        logger.debug(f"Fetching GPU stats from {len(healthy_endpoints)} healthy endpoints")
        
        # Create tasks for concurrent fetching from all healthy endpoints
        fetch_tasks = []
        for endpoint in healthy_endpoints:
            task = asyncio.create_task(
                self._fetch_gpu_stats_with_metadata(endpoint.endpoint_id, endpoint.url)
            )
            fetch_tasks.append((endpoint.endpoint_id, task))
        
        # Wait for all fetch operations to complete
        aggregated_gpus = []
        successful_fetches = 0
        
        for endpoint_id, task in fetch_tasks:
            try:
                result = await task
                if result is not None:
                    endpoint_id_from_result, gpu_stats = result
                    
                    # Convert local GPU info to global GPU info
                    for gpu_info in gpu_stats.gpus_summary:
                        global_gpu_info = create_global_gpu_info_from_gpu_info(
                            gpu_info=gpu_info,
                            endpoint_id=endpoint_id_from_result,
                            is_available=True  # Will be determined by allocation logic
                        )
                        aggregated_gpus.append(global_gpu_info)
                    
                    successful_fetches += 1
                    self._last_successful_fetch[endpoint_id] = datetime.now()
                    logger.debug(f"Successfully fetched {len(gpu_stats.gpus_summary)} GPUs from endpoint {endpoint_id}")
                else:
                    logger.warning(f"Received None response from endpoint {endpoint_id}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch GPU stats from endpoint {endpoint_id}: {e}")
                self._update_request_stats(endpoint_id, success=False, error=str(e))
                continue
        
        if successful_fetches == 0:
            logger.error("Failed to fetch GPU stats from any endpoint")
            return None
        
        logger.info(f"Aggregated {len(aggregated_gpus)} GPUs from {successful_fetches}/{len(healthy_endpoints)} endpoints")
        return aggregated_gpus
    
    async def fetch_gpu_stats_from_endpoint(self, endpoint_id: str) -> Optional[GPUStats]:
        """Fetch GPU statistics from a specific endpoint.
        
        Args:
            endpoint_id: ID of the endpoint to fetch from
            
        Returns:
            GPUStats object if successful, None otherwise
        """
        endpoint = self.endpoint_manager.get_endpoint_by_id(endpoint_id)
        if not endpoint:
            logger.error(f"Endpoint {endpoint_id} not found")
            return None
        
        if not endpoint.is_healthy:
            logger.warning(f"Endpoint {endpoint_id} is not healthy, skipping fetch")
            return None
        
        client = self.endpoint_manager.get_http_client(endpoint_id)
        if not client:
            logger.error(f"HTTP client not found for endpoint {endpoint_id}")
            return None
        
        try:
            logger.debug(f"Fetching GPU stats from endpoint {endpoint_id}")
            gpu_stats = await client.fetch_gpu_stats()
            
            if gpu_stats is not None:
                self._update_request_stats(endpoint_id, success=True)
                self._last_successful_fetch[endpoint_id] = datetime.now()
                logger.debug(f"Successfully fetched {gpu_stats.gpu_count} GPUs from endpoint {endpoint_id}")
                return gpu_stats
            else:
                logger.warning(f"Received None response from endpoint {endpoint_id}")
                self._update_request_stats(endpoint_id, success=False, error="None response")
                return None
                
        except ServiceUnavailableError as e:
            logger.warning(f"Service unavailable for endpoint {endpoint_id}: {e}")
            self._update_request_stats(endpoint_id, success=False, error=f"Service unavailable: {e}")
            return None
            
        except RetryableError as e:
            logger.warning(f"Retryable error for endpoint {endpoint_id}: {e}")
            self._update_request_stats(endpoint_id, success=False, error=f"Retryable error: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error fetching from endpoint {endpoint_id}: {e}")
            self._update_request_stats(endpoint_id, success=False, error=f"Unexpected error: {e}")
            return None
    
    async def _fetch_gpu_stats_with_metadata(self, endpoint_id: str, endpoint_url: str) -> Optional[Tuple[str, GPUStats]]:
        """Fetch GPU stats with endpoint metadata.
        
        Args:
            endpoint_id: ID of the endpoint
            endpoint_url: URL of the endpoint
            
        Returns:
            Tuple of (endpoint_id, gpu_stats) if successful, None otherwise
        """
        gpu_stats = await self.fetch_gpu_stats_from_endpoint(endpoint_id)
        if gpu_stats is not None:
            return (endpoint_id, gpu_stats)
        return None
    
    def _update_request_stats(self, endpoint_id: str, success: bool, error: Optional[str] = None) -> None:
        """Update request statistics for an endpoint.
        
        Args:
            endpoint_id: ID of the endpoint
            success: Whether the request was successful
            error: Error message if request failed
        """
        if endpoint_id not in self._request_stats:
            self._request_stats[endpoint_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "last_error": None,
                "last_request_time": None
            }
        
        stats = self._request_stats[endpoint_id]
        stats["total_requests"] += 1
        stats["last_request_time"] = datetime.now()
        
        if success:
            stats["successful_requests"] += 1
            stats["last_error"] = None
        else:
            stats["failed_requests"] += 1
            stats["last_error"] = error
    
    def get_client_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive client pool statistics for monitoring.
        
        Returns:
            Dictionary containing pool statistics and per-endpoint metrics
        """
        healthy_endpoints = self.endpoint_manager.get_healthy_endpoints()
        all_endpoints = self.endpoint_manager.get_all_endpoints()
        
        total_requests = sum(stats["total_requests"] for stats in self._request_stats.values())
        total_successful = sum(stats["successful_requests"] for stats in self._request_stats.values())
        total_failed = sum(stats["failed_requests"] for stats in self._request_stats.values())
        
        success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0.0
        
        endpoint_stats = []
        for endpoint in all_endpoints:
            endpoint_id = endpoint.endpoint_id
            request_stats = self._request_stats.get(endpoint_id, {})
            last_successful = self._last_successful_fetch.get(endpoint_id)
            
            endpoint_stats.append({
                "endpoint_id": endpoint_id,
                "url": endpoint.url,
                "is_healthy": endpoint.is_healthy,
                "total_requests": request_stats.get("total_requests", 0),
                "successful_requests": request_stats.get("successful_requests", 0),
                "failed_requests": request_stats.get("failed_requests", 0),
                "success_rate": (
                    request_stats.get("successful_requests", 0) / request_stats.get("total_requests", 1) * 100
                    if request_stats.get("total_requests", 0) > 0 else 0.0
                ),
                "last_error": request_stats.get("last_error"),
                "last_request_time": (
                    request_stats.get("last_request_time").isoformat() 
                    if request_stats.get("last_request_time") else None
                ),
                "last_successful_fetch": last_successful.isoformat() if last_successful else None
            })
        
        return {
            "pool_summary": {
                "total_endpoints": len(all_endpoints),
                "healthy_endpoints": len(healthy_endpoints),
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "failed_requests": total_failed,
                "overall_success_rate": success_rate
            },
            "endpoint_statistics": endpoint_stats
        }
    
    async def close(self) -> None:
        """Close all HTTP clients and cleanup resources."""
        # HTTP clients are managed by the endpoint manager
        # Just clear our internal state
        self._request_stats.clear()
        self._last_successful_fetch.clear()
        logger.debug("Multi-endpoint HTTP client pool closed")


class MockMultiEndpointHTTPClientPool(MultiEndpointHTTPClientPool):
    """Mock multi-endpoint HTTP client pool for testing purposes."""
    
    def __init__(self, mock_responses: Optional[Dict[str, List[GlobalGPUInfo]]] = None):
        """Initialize mock client pool.
        
        Args:
            mock_responses: Dictionary mapping endpoint IDs to mock GPU info lists
        """
        self.mock_responses = mock_responses or {}
        self.fetch_call_count = 0
        self.endpoint_fetch_counts: Dict[str, int] = {}
        self.should_fail = False
        self.failure_exception = Exception("Mock failure")
    
    async def fetch_aggregated_gpu_stats(self) -> Optional[List[GlobalGPUInfo]]:
        """Mock fetch aggregated GPU statistics."""
        self.fetch_call_count += 1
        
        if self.should_fail:
            raise self.failure_exception
        
        # Aggregate all mock responses
        aggregated = []
        for endpoint_id, gpu_infos in self.mock_responses.items():
            aggregated.extend(gpu_infos)
        
        return aggregated if aggregated else None
    
    async def fetch_gpu_stats_from_endpoint(self, endpoint_id: str) -> Optional[GPUStats]:
        """Mock fetch GPU statistics from specific endpoint."""
        self.endpoint_fetch_counts[endpoint_id] = self.endpoint_fetch_counts.get(endpoint_id, 0) + 1
        
        if self.should_fail:
            raise self.failure_exception
        
        # Return mock GPUStats if we have mock data for this endpoint
        if endpoint_id in self.mock_responses:
            gpu_infos = self.mock_responses[endpoint_id]
            # Convert GlobalGPUInfo back to GPUInfo for GPUStats
            from .models import GPUInfo
            
            gpus_summary = []
            for global_gpu_info in gpu_infos:
                gpu_info = GPUInfo(
                    gpu_id=global_gpu_info.local_gpu_id,
                    name=global_gpu_info.name,
                    memory_usage_percent=global_gpu_info.memory_usage_percent,
                    utilization_percent=global_gpu_info.utilization_percent
                )
                gpus_summary.append(gpu_info)
            
            # Create mock GPUStats
            total_memory = len(gpus_summary) * 8192  # Mock 8GB per GPU
            used_memory = sum(int(gpu.memory_usage_percent / 100 * 8192) for gpu in gpus_summary)
            avg_util = sum(gpu.utilization_percent for gpu in gpus_summary) / len(gpus_summary) if gpus_summary else 0
            
            from .models import GPUStats
            return GPUStats(
                gpu_count=len(gpus_summary),
                total_memory_mb=total_memory,
                total_used_memory_mb=used_memory,
                average_utilization_percent=avg_util,
                gpus_summary=gpus_summary,
                total_memory_usage_percent=used_memory / total_memory * 100 if total_memory > 0 else 0,
                timestamp=datetime.now().isoformat()
            )
        
        return None
    
    async def close(self) -> None:
        """Mock close method."""
        pass
    
    def set_mock_response(self, endpoint_id: str, gpu_infos: List[GlobalGPUInfo]) -> None:
        """Set mock response for an endpoint."""
        self.mock_responses[endpoint_id] = gpu_infos
    
    def set_failure_mode(self, should_fail: bool, exception: Optional[Exception] = None) -> None:
        """Set failure mode for testing."""
        self.should_fail = should_fail
        if exception:
            self.failure_exception = exception