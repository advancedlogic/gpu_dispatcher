"""GPU Worker Pool - Intelligent GPU resource allocation and management."""

__version__ = "1.0.0"
__author__ = "GPU Worker Pool Contributors"
__email__ = "contributors@gpu-worker-pool.dev"
__license__ = "MIT"

# Public API exports
from .client import (
    GPUWorkerPoolClient,
    GPUContextManager,
    gpu_worker_pool_client,
)
from .models import (
    GPUAssignment,
    GPUInfo,
    GPUStats,
    PoolStatus,
    WorkerInfo,
)
from .config import (
    EnvironmentConfigurationManager,
)
from .worker_pool_manager import (
    WorkerTimeoutError,
    StaleAssignmentError,
)
from .http_client import (
    ServiceUnavailableError,
    RetryableError,
    NetworkRetryableError,
    ServiceRetryableError,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Main client interface
    "GPUWorkerPoolClient",
    "GPUContextManager", 
    "gpu_worker_pool_client",
    
    # Data models
    "GPUAssignment",
    "GPUInfo",
    "GPUStats",
    "PoolStatus",
    "WorkerInfo",
    
    # Configuration
    "EnvironmentConfigurationManager",
    
    # Exceptions
    "WorkerTimeoutError",
    "StaleAssignmentError",
    "ServiceUnavailableError",
    "RetryableError",
    "NetworkRetryableError",
    "ServiceRetryableError",
]