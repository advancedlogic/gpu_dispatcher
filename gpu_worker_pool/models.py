"""Core data models for the GPU Worker Pool system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Callable, Optional, Any
import json


@dataclass
class GPUInfo:
    """Information about a single GPU's current state."""
    gpu_id: int
    name: str
    memory_usage_percent: float
    utilization_percent: float
    
    def __post_init__(self):
        """Validate GPU information after initialization."""
        if not isinstance(self.gpu_id, int) or self.gpu_id < 0:
            raise ValueError(f"gpu_id must be a non-negative integer, got {self.gpu_id}")
        
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"name must be a non-empty string, got {self.name}")
        
        if not isinstance(self.memory_usage_percent, (int, float)) or not (0 <= self.memory_usage_percent <= 100):
            raise ValueError(f"memory_usage_percent must be between 0 and 100, got {self.memory_usage_percent}")
        
        if not isinstance(self.utilization_percent, (int, float)) or not (0 <= self.utilization_percent <= 100):
            raise ValueError(f"utilization_percent must be between 0 and 100, got {self.utilization_percent}")


@dataclass
class GPUStats:
    """Complete GPU statistics from the monitoring service."""
    gpu_count: int
    total_memory_mb: int
    total_used_memory_mb: int
    average_utilization_percent: float
    gpus_summary: List[GPUInfo]
    total_memory_usage_percent: float
    timestamp: str
    
    def __post_init__(self):
        """Validate GPU statistics after initialization."""
        if not isinstance(self.gpu_count, int) or self.gpu_count < 0:
            raise ValueError(f"gpu_count must be a non-negative integer, got {self.gpu_count}")
        
        if not isinstance(self.total_memory_mb, int) or self.total_memory_mb < 0:
            raise ValueError(f"total_memory_mb must be a non-negative integer, got {self.total_memory_mb}")
        
        if not isinstance(self.total_used_memory_mb, int) or self.total_used_memory_mb < 0:
            raise ValueError(f"total_used_memory_mb must be a non-negative integer, got {self.total_used_memory_mb}")
        
        if self.total_used_memory_mb > self.total_memory_mb:
            raise ValueError(f"total_used_memory_mb ({self.total_used_memory_mb}) cannot exceed total_memory_mb ({self.total_memory_mb})")
        
        if not isinstance(self.average_utilization_percent, (int, float)) or not (0 <= self.average_utilization_percent <= 100):
            raise ValueError(f"average_utilization_percent must be between 0 and 100, got {self.average_utilization_percent}")
        
        if not isinstance(self.total_memory_usage_percent, (int, float)) or not (0 <= self.total_memory_usage_percent <= 100):
            raise ValueError(f"total_memory_usage_percent must be between 0 and 100, got {self.total_memory_usage_percent}")
        
        if not isinstance(self.gpus_summary, list):
            raise ValueError(f"gpus_summary must be a list, got {type(self.gpus_summary)}")
        
        if len(self.gpus_summary) != self.gpu_count:
            raise ValueError(f"gpus_summary length ({len(self.gpus_summary)}) must match gpu_count ({self.gpu_count})")
        
        if not isinstance(self.timestamp, str) or not self.timestamp.strip():
            raise ValueError(f"timestamp must be a non-empty string, got {self.timestamp}")
        
        # Validate all GPUInfo objects in the summary
        for i, gpu_info in enumerate(self.gpus_summary):
            if not isinstance(gpu_info, GPUInfo):
                raise ValueError(f"gpus_summary[{i}] must be a GPUInfo instance, got {type(gpu_info)}")


@dataclass
class WorkerInfo:
    """Information about a worker process requesting GPU resources."""
    id: str
    enqueued_at: datetime
    callback: Callable[[int], None]
    on_error: Callable[[Exception], None]
    
    def __post_init__(self):
        """Validate worker information after initialization."""
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError(f"id must be a non-empty string, got {self.id}")
        
        if not isinstance(self.enqueued_at, datetime):
            raise ValueError(f"enqueued_at must be a datetime instance, got {type(self.enqueued_at)}")
        
        if not callable(self.callback):
            raise ValueError(f"callback must be callable, got {type(self.callback)}")
        
        if not callable(self.on_error):
            raise ValueError(f"on_error must be callable, got {type(self.on_error)}")


@dataclass
class GPUAssignment:
    """Represents an assignment of a worker to a specific GPU.
    
    This class supports both local GPU IDs (for backward compatibility) and global GPU IDs
    (for multi-endpoint support). When using multi-endpoint mode, the gpu_id field contains
    the local GPU ID and additional fields provide global context.
    """
    gpu_id: int
    worker_id: str
    assigned_at: datetime
    # Multi-endpoint support fields (optional for backward compatibility)
    global_gpu_id: Optional[str] = None
    endpoint_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate GPU assignment after initialization."""
        if not isinstance(self.gpu_id, int) or self.gpu_id < 0:
            raise ValueError(f"gpu_id must be a non-negative integer, got {self.gpu_id}")
        
        if not isinstance(self.worker_id, str) or not self.worker_id.strip():
            raise ValueError(f"worker_id must be a non-empty string, got {self.worker_id}")
        
        if not isinstance(self.assigned_at, datetime):
            raise ValueError(f"assigned_at must be a datetime instance, got {type(self.assigned_at)}")
        
        # Validate multi-endpoint fields if provided
        if self.global_gpu_id is not None:
            if not isinstance(self.global_gpu_id, str) or not self.global_gpu_id.strip():
                raise ValueError(f"global_gpu_id must be a non-empty string if provided, got {self.global_gpu_id}")
            
            if ':' not in self.global_gpu_id:
                raise ValueError(f"global_gpu_id must follow format 'endpoint_id:local_gpu_id', got {self.global_gpu_id}")
        
        if self.endpoint_id is not None:
            if not isinstance(self.endpoint_id, str) or not self.endpoint_id.strip():
                raise ValueError(f"endpoint_id must be a non-empty string if provided, got {self.endpoint_id}")
        
        # If global_gpu_id is provided, endpoint_id should also be provided and vice versa
        if (self.global_gpu_id is None) != (self.endpoint_id is None):
            raise ValueError("global_gpu_id and endpoint_id must both be provided or both be None")
        
        # If both are provided, validate consistency
        if self.global_gpu_id is not None and self.endpoint_id is not None:
            expected_global_id = f"{self.endpoint_id}:{self.gpu_id}"
            if self.global_gpu_id != expected_global_id:
                raise ValueError(f"global_gpu_id '{self.global_gpu_id}' does not match expected '{expected_global_id}'")
    
    @property
    def is_multi_endpoint(self) -> bool:
        """Check if this assignment uses multi-endpoint global IDs."""
        return self.global_gpu_id is not None and self.endpoint_id is not None
    
    def get_display_id(self) -> str:
        """Get the appropriate GPU ID for display purposes."""
        return self.global_gpu_id if self.is_multi_endpoint else str(self.gpu_id)
    
    @classmethod
    def create_local_assignment(cls, gpu_id: int, worker_id: str, assigned_at: Optional[datetime] = None) -> 'GPUAssignment':
        """Create a local GPU assignment (single-endpoint mode).
        
        Args:
            gpu_id: Local GPU ID
            worker_id: Worker identifier
            assigned_at: Assignment timestamp (defaults to now)
            
        Returns:
            GPUAssignment instance for single-endpoint use
        """
        return cls(
            gpu_id=gpu_id,
            worker_id=worker_id,
            assigned_at=assigned_at or datetime.now()
        )
    
    @classmethod
    def create_global_assignment(cls, global_gpu_id: str, worker_id: str, assigned_at: Optional[datetime] = None) -> 'GPUAssignment':
        """Create a global GPU assignment (multi-endpoint mode).
        
        Args:
            global_gpu_id: Global GPU identifier in format 'endpoint_id:local_gpu_id'
            worker_id: Worker identifier
            assigned_at: Assignment timestamp (defaults to now)
            
        Returns:
            GPUAssignment instance for multi-endpoint use
            
        Raises:
            ValueError: If global_gpu_id format is invalid
        """
        endpoint_id, local_gpu_id = parse_global_gpu_id(global_gpu_id)
        
        return cls(
            gpu_id=local_gpu_id,
            worker_id=worker_id,
            assigned_at=assigned_at or datetime.now(),
            global_gpu_id=global_gpu_id,
            endpoint_id=endpoint_id
        )


@dataclass
class PoolStatus:
    """Current status of the GPU worker pool."""
    total_gpus: int
    available_gpus: int
    active_workers: int
    blocked_workers: int
    gpu_assignments: Dict[int, List[WorkerInfo]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate pool status after initialization."""
        if not isinstance(self.total_gpus, int) or self.total_gpus < 0:
            raise ValueError(f"total_gpus must be a non-negative integer, got {self.total_gpus}")
        
        if not isinstance(self.available_gpus, int) or self.available_gpus < 0:
            raise ValueError(f"available_gpus must be a non-negative integer, got {self.available_gpus}")
        
        if self.available_gpus > self.total_gpus:
            raise ValueError(f"available_gpus ({self.available_gpus}) cannot exceed total_gpus ({self.total_gpus})")
        
        if not isinstance(self.active_workers, int) or self.active_workers < 0:
            raise ValueError(f"active_workers must be a non-negative integer, got {self.active_workers}")
        
        if not isinstance(self.blocked_workers, int) or self.blocked_workers < 0:
            raise ValueError(f"blocked_workers must be a non-negative integer, got {self.blocked_workers}")
        
        if not isinstance(self.gpu_assignments, dict):
            raise ValueError(f"gpu_assignments must be a dictionary, got {type(self.gpu_assignments)}")
        
        # Validate gpu_assignments structure
        for gpu_id, workers in self.gpu_assignments.items():
            if not isinstance(gpu_id, int) or gpu_id < 0:
                raise ValueError(f"GPU ID in assignments must be a non-negative integer, got {gpu_id}")
            
            if not isinstance(workers, list):
                raise ValueError(f"Workers list for GPU {gpu_id} must be a list, got {type(workers)}")
            
            for i, worker in enumerate(workers):
                if not isinstance(worker, WorkerInfo):
                    raise ValueError(f"Worker {i} for GPU {gpu_id} must be a WorkerInfo instance, got {type(worker)}")


@dataclass
class EndpointInfo:
    """Information about a GPU statistics server endpoint."""
    endpoint_id: str
    url: str
    is_healthy: bool
    last_seen: datetime
    total_gpus: int
    available_gpus: int
    response_time_ms: float
    
    def __post_init__(self):
        """Validate endpoint information after initialization."""
        if not isinstance(self.endpoint_id, str) or not self.endpoint_id.strip():
            raise ValueError(f"endpoint_id must be a non-empty string, got {self.endpoint_id}")
        
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError(f"url must be a non-empty string, got {self.url}")
        
        if not isinstance(self.is_healthy, bool):
            raise ValueError(f"is_healthy must be a boolean, got {type(self.is_healthy)}")
        
        if not isinstance(self.last_seen, datetime):
            raise ValueError(f"last_seen must be a datetime instance, got {type(self.last_seen)}")
        
        if not isinstance(self.total_gpus, int) or self.total_gpus < 0:
            raise ValueError(f"total_gpus must be a non-negative integer, got {self.total_gpus}")
        
        if not isinstance(self.available_gpus, int) or self.available_gpus < 0:
            raise ValueError(f"available_gpus must be a non-negative integer, got {self.available_gpus}")
        
        if self.available_gpus > self.total_gpus:
            raise ValueError(f"available_gpus ({self.available_gpus}) cannot exceed total_gpus ({self.total_gpus})")
        
        if not isinstance(self.response_time_ms, (int, float)) or self.response_time_ms < 0:
            raise ValueError(f"response_time_ms must be a non-negative number, got {self.response_time_ms}")


@dataclass
class GlobalGPUInfo:
    """Information about a GPU with global identifier across multiple endpoints."""
    global_gpu_id: str
    endpoint_id: str
    local_gpu_id: int
    name: str
    memory_usage_percent: float
    utilization_percent: float
    is_available: bool
    
    def __post_init__(self):
        """Validate global GPU information after initialization."""
        if not isinstance(self.global_gpu_id, str) or not self.global_gpu_id.strip():
            raise ValueError(f"global_gpu_id must be a non-empty string, got {self.global_gpu_id}")
        
        if not isinstance(self.endpoint_id, str) or not self.endpoint_id.strip():
            raise ValueError(f"endpoint_id must be a non-empty string, got {self.endpoint_id}")
        
        if not isinstance(self.local_gpu_id, int) or self.local_gpu_id < 0:
            raise ValueError(f"local_gpu_id must be a non-negative integer, got {self.local_gpu_id}")
        
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"name must be a non-empty string, got {self.name}")
        
        if not isinstance(self.memory_usage_percent, (int, float)) or not (0 <= self.memory_usage_percent <= 100):
            raise ValueError(f"memory_usage_percent must be between 0 and 100, got {self.memory_usage_percent}")
        
        if not isinstance(self.utilization_percent, (int, float)) or not (0 <= self.utilization_percent <= 100):
            raise ValueError(f"utilization_percent must be between 0 and 100, got {self.utilization_percent}")
        
        if not isinstance(self.is_available, bool):
            raise ValueError(f"is_available must be a boolean, got {type(self.is_available)}")
        
        # Validate that global_gpu_id follows the expected format
        if ':' not in self.global_gpu_id:
            raise ValueError(f"global_gpu_id must follow format 'endpoint_id:local_gpu_id', got {self.global_gpu_id}")
        
        # Validate that global_gpu_id matches endpoint_id and local_gpu_id
        expected_global_id = f"{self.endpoint_id}:{self.local_gpu_id}"
        if self.global_gpu_id != expected_global_id:
            raise ValueError(f"global_gpu_id '{self.global_gpu_id}' does not match expected '{expected_global_id}'")


@dataclass
class MultiEndpointPoolStatus:
    """Current status of the GPU worker pool across multiple endpoints."""
    total_endpoints: int
    healthy_endpoints: int
    total_gpus: int
    available_gpus: int
    active_workers: int
    blocked_workers: int
    endpoints: List[EndpointInfo]
    gpu_assignments: Dict[str, List[WorkerInfo]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate multi-endpoint pool status after initialization."""
        if not isinstance(self.total_endpoints, int) or self.total_endpoints < 0:
            raise ValueError(f"total_endpoints must be a non-negative integer, got {self.total_endpoints}")
        
        if not isinstance(self.healthy_endpoints, int) or self.healthy_endpoints < 0:
            raise ValueError(f"healthy_endpoints must be a non-negative integer, got {self.healthy_endpoints}")
        
        if self.healthy_endpoints > self.total_endpoints:
            raise ValueError(f"healthy_endpoints ({self.healthy_endpoints}) cannot exceed total_endpoints ({self.total_endpoints})")
        
        if not isinstance(self.total_gpus, int) or self.total_gpus < 0:
            raise ValueError(f"total_gpus must be a non-negative integer, got {self.total_gpus}")
        
        if not isinstance(self.available_gpus, int) or self.available_gpus < 0:
            raise ValueError(f"available_gpus must be a non-negative integer, got {self.available_gpus}")
        
        if self.available_gpus > self.total_gpus:
            raise ValueError(f"available_gpus ({self.available_gpus}) cannot exceed total_gpus ({self.total_gpus})")
        
        if not isinstance(self.active_workers, int) or self.active_workers < 0:
            raise ValueError(f"active_workers must be a non-negative integer, got {self.active_workers}")
        
        if not isinstance(self.blocked_workers, int) or self.blocked_workers < 0:
            raise ValueError(f"blocked_workers must be a non-negative integer, got {self.blocked_workers}")
        
        if not isinstance(self.endpoints, list):
            raise ValueError(f"endpoints must be a list, got {type(self.endpoints)}")
        
        if len(self.endpoints) != self.total_endpoints:
            raise ValueError(f"endpoints length ({len(self.endpoints)}) must match total_endpoints ({self.total_endpoints})")
        
        # Validate all EndpointInfo objects
        for i, endpoint in enumerate(self.endpoints):
            if not isinstance(endpoint, EndpointInfo):
                raise ValueError(f"endpoints[{i}] must be an EndpointInfo instance, got {type(endpoint)}")
        
        if not isinstance(self.gpu_assignments, dict):
            raise ValueError(f"gpu_assignments must be a dictionary, got {type(self.gpu_assignments)}")
        
        # Validate gpu_assignments structure (now uses global GPU IDs as keys)
        for global_gpu_id, workers in self.gpu_assignments.items():
            if not isinstance(global_gpu_id, str) or not global_gpu_id.strip():
                raise ValueError(f"Global GPU ID in assignments must be a non-empty string, got {global_gpu_id}")
            
            if ':' not in global_gpu_id:
                raise ValueError(f"Global GPU ID must follow format 'endpoint_id:local_gpu_id', got {global_gpu_id}")
            
            if not isinstance(workers, list):
                raise ValueError(f"Workers list for GPU {global_gpu_id} must be a list, got {type(workers)}")
            
            for i, worker in enumerate(workers):
                if not isinstance(worker, WorkerInfo):
                    raise ValueError(f"Worker {i} for GPU {global_gpu_id} must be a WorkerInfo instance, got {type(worker)}")


@dataclass
class GlobalGPUAssignment:
    """Represents an assignment of a worker to a specific GPU using global identifiers."""
    global_gpu_id: str
    endpoint_id: str
    local_gpu_id: int
    worker_id: str
    assigned_at: datetime
    
    def __post_init__(self):
        """Validate global GPU assignment after initialization."""
        if not isinstance(self.global_gpu_id, str) or not self.global_gpu_id.strip():
            raise ValueError(f"global_gpu_id must be a non-empty string, got {self.global_gpu_id}")
        
        if not isinstance(self.endpoint_id, str) or not self.endpoint_id.strip():
            raise ValueError(f"endpoint_id must be a non-empty string, got {self.endpoint_id}")
        
        if not isinstance(self.local_gpu_id, int) or self.local_gpu_id < 0:
            raise ValueError(f"local_gpu_id must be a non-negative integer, got {self.local_gpu_id}")
        
        if not isinstance(self.worker_id, str) or not self.worker_id.strip():
            raise ValueError(f"worker_id must be a non-empty string, got {self.worker_id}")
        
        if not isinstance(self.assigned_at, datetime):
            raise ValueError(f"assigned_at must be a datetime instance, got {type(self.assigned_at)}")
        
        # Validate that global_gpu_id follows the expected format
        if ':' not in self.global_gpu_id:
            raise ValueError(f"global_gpu_id must follow format 'endpoint_id:local_gpu_id', got {self.global_gpu_id}")
        
        # Validate that global_gpu_id matches endpoint_id and local_gpu_id
        expected_global_id = f"{self.endpoint_id}:{self.local_gpu_id}"
        if self.global_gpu_id != expected_global_id:
            raise ValueError(f"global_gpu_id '{self.global_gpu_id}' does not match expected '{expected_global_id}'")


# Global GPU ID Utilities

def create_global_gpu_id(endpoint_id: str, local_gpu_id: int) -> str:
    """Create a global GPU identifier from endpoint ID and local GPU ID.
    
    Args:
        endpoint_id: Unique identifier for the endpoint
        local_gpu_id: GPU ID on the local endpoint
        
    Returns:
        Global GPU identifier in format 'endpoint_id:local_gpu_id'
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(endpoint_id, str) or not endpoint_id.strip():
        raise ValueError(f"endpoint_id must be a non-empty string, got {endpoint_id}")
    
    if not isinstance(local_gpu_id, int) or local_gpu_id < 0:
        raise ValueError(f"local_gpu_id must be a non-negative integer, got {local_gpu_id}")
    
    # Ensure endpoint_id doesn't contain colons to avoid parsing ambiguity
    if ':' in endpoint_id:
        raise ValueError(f"endpoint_id cannot contain colons, got {endpoint_id}")
    
    return f"{endpoint_id}:{local_gpu_id}"


def parse_global_gpu_id(global_gpu_id: str) -> tuple[str, int]:
    """Parse a global GPU identifier into endpoint ID and local GPU ID.
    
    Args:
        global_gpu_id: Global GPU identifier in format 'endpoint_id:local_gpu_id'
        
    Returns:
        Tuple of (endpoint_id, local_gpu_id)
        
    Raises:
        ValueError: If the global GPU ID format is invalid
    """
    if not isinstance(global_gpu_id, str) or not global_gpu_id.strip():
        raise ValueError(f"global_gpu_id must be a non-empty string, got {global_gpu_id}")
    
    if ':' not in global_gpu_id:
        raise ValueError(f"global_gpu_id must contain a colon separator, got {global_gpu_id}")
    
    # Split on the last colon to handle endpoint IDs that might contain colons
    # (though we validate against this in create_global_gpu_id)
    parts = global_gpu_id.rsplit(':', 1)
    
    if len(parts) != 2:
        raise ValueError(f"global_gpu_id must have exactly one colon separator, got {global_gpu_id}")
    
    endpoint_id, local_gpu_id_str = parts
    
    if not endpoint_id.strip():
        raise ValueError(f"endpoint_id part cannot be empty in global_gpu_id {global_gpu_id}")
    
    try:
        local_gpu_id = int(local_gpu_id_str)
        if local_gpu_id < 0:
            raise ValueError(f"local_gpu_id must be non-negative, got {local_gpu_id}")
    except ValueError as e:
        raise ValueError(f"Invalid local_gpu_id in global_gpu_id {global_gpu_id}: {e}")
    
    return endpoint_id, local_gpu_id


def create_endpoint_id_from_url(url: str) -> str:
    """Create a unique endpoint identifier from a URL.
    
    Args:
        url: The endpoint URL
        
    Returns:
        Unique endpoint identifier safe for use in global GPU IDs
        
    Raises:
        ValueError: If URL is invalid
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"url must be a non-empty string, got {url}")
    
    # Parse URL to extract host and port
    import urllib.parse
    
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc:
            raise ValueError(f"URL must include host, got {url}")
        
        # Create endpoint ID from host:port, replacing colons with underscores
        # to avoid conflicts with global GPU ID format
        endpoint_id = parsed.netloc.replace(':', '_')
        
        # Remove any characters that might cause issues
        endpoint_id = endpoint_id.replace('/', '_').replace('\\', '_')
        
        return endpoint_id
        
    except Exception as e:
        raise ValueError(f"Invalid URL format: {url}, error: {e}")


def create_gpu_stats_from_json(data: Dict[str, Any]) -> GPUStats:
    """Create GPUStats from JSON response data with validation."""
    try:
        # Parse GPU summaries
        gpus_summary = []
        for gpu_data in data.get('gpus_summary', []):
            gpu_info = GPUInfo(
                gpu_id=gpu_data['gpu_id'],
                name=gpu_data['name'],
                memory_usage_percent=gpu_data['memory_usage_percent'],
                utilization_percent=gpu_data['utilization_percent']
            )
            gpus_summary.append(gpu_info)
        
        # Create GPUStats instance
        gpu_stats = GPUStats(
            gpu_count=data['gpu_count'],
            total_memory_mb=data['total_memory_mb'],
            total_used_memory_mb=data['total_used_memory_mb'],
            average_utilization_percent=data['average_utilization_percent'],
            gpus_summary=gpus_summary,
            total_memory_usage_percent=data['total_memory_usage_percent'],
            timestamp=data['timestamp']
        )
        
        return gpu_stats
        
    except KeyError as e:
        raise ValueError(f"Missing required field in GPU stats JSON: {e}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid data format in GPU stats JSON: {e}")


def create_global_gpu_info_from_gpu_info(gpu_info: GPUInfo, endpoint_id: str, is_available: bool = True) -> GlobalGPUInfo:
    """Create GlobalGPUInfo from GPUInfo and endpoint information.
    
    Args:
        gpu_info: Local GPU information
        endpoint_id: Endpoint identifier
        is_available: Whether the GPU is available for assignment
        
    Returns:
        GlobalGPUInfo instance with global identifier
    """
    global_gpu_id = create_global_gpu_id(endpoint_id, gpu_info.gpu_id)
    
    return GlobalGPUInfo(
        global_gpu_id=global_gpu_id,
        endpoint_id=endpoint_id,
        local_gpu_id=gpu_info.gpu_id,
        name=gpu_info.name,
        memory_usage_percent=gpu_info.memory_usage_percent,
        utilization_percent=gpu_info.utilization_percent,
        is_available=is_available
    )