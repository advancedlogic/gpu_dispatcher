# GPU Worker Pool API Documentation

This document provides detailed API documentation for the GPU Worker Pool system.

## Table of Contents

- [Client Interface](#client-interface)
- [Data Models](#data-models)
- [Configuration](#configuration)
- [Exceptions](#exceptions)
- [Utilities](#utilities)

## Client Interface

### GPUWorkerPoolClient

The main client interface for the GPU Worker Pool system.

```python
from gpu_worker_pool.client import GPUWorkerPoolClient
```

#### Constructor

```python
def __init__(self,
             service_endpoint: Optional[str] = None,
             service_endpoints: Optional[str] = None,
             load_balancing_strategy: Optional[str] = None,
             memory_threshold: Optional[float] = None,
             utilization_threshold: Optional[float] = None,
             polling_interval: Optional[int] = None,
             worker_timeout: float = 300.0,
             request_timeout: float = 30.0)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `service_endpoint` | `Optional[str]` | `None` | Single GPU statistics service URL. If None, uses `GPU_SERVICE_ENDPOINT` environment variable or default `http://localhost:8080` |
| `service_endpoints` | `Optional[str]` | `None` | Comma-separated list of GPU statistics service URLs for multi-endpoint mode. If None, uses `GPU_STATS_SERVICE_ENDPOINTS` environment variable |
| `load_balancing_strategy` | `Optional[str]` | `None` | Load balancing strategy: `"availability"`, `"round_robin"`, or `"weighted"`. If None, uses `GPU_LOAD_BALANCING_STRATEGY` environment variable or default `"availability"` |
| `memory_threshold` | `Optional[float]` | `None` | Memory usage threshold percentage (0-100). If None, uses `GPU_MEMORY_THRESHOLD_PERCENT` environment variable or default 80.0 |
| `utilization_threshold` | `Optional[float]` | `None` | GPU utilization threshold percentage (0-100). If None, uses `GPU_UTILIZATION_THRESHOLD_PERCENT` environment variable or default 90.0 |
| `polling_interval` | `Optional[int]` | `None` | Statistics polling interval in seconds. If None, uses `GPU_POLLING_INTERVAL` environment variable or default 5 |
| `worker_timeout` | `float` | `300.0` | Timeout for GPU requests in seconds |
| `request_timeout` | `float` | `30.0` | HTTP request timeout in seconds |

**Mode Selection:**
- If `service_endpoints` is provided: **Multi-endpoint mode**
- If only `service_endpoint` is provided: **Single-endpoint mode**
- If both are provided: `service_endpoints` takes precedence
- If neither is provided: Uses environment variables with same precedence rules

**Example (Single-endpoint):**
```python
client = GPUWorkerPoolClient(
    service_endpoint="https://gpu-service.company.com",
    memory_threshold=75.0,
    utilization_threshold=85.0,
    worker_timeout=600.0
)
```

**Example (Multi-endpoint):**
```python
client = GPUWorkerPoolClient(
    service_endpoints="http://gpu1:8000,http://gpu2:8000,http://gpu3:8000",
    load_balancing_strategy="availability",
    memory_threshold=75.0,
    utilization_threshold=85.0,
    worker_timeout=600.0
)
```

#### Methods

##### `async start() -> None`

Start the GPU worker pool client and all underlying components.

**Raises:**
- `RuntimeError`: If the client is already started
- `Exception`: If initialization fails

**Example:**
```python
client = GPUWorkerPoolClient()
await client.start()
```

##### `async stop() -> None`

Stop the GPU worker pool client and cleanup all resources.

**Example:**
```python
await client.stop()
```

##### `async request_gpu(timeout: Optional[float] = None) -> GPUAssignment`

Request a GPU assignment for the current worker.

**Parameters:**
- `timeout` (`Optional[float]`): Optional timeout in seconds. If None, uses the client's `worker_timeout`

**Returns:**
- `GPUAssignment`: Contains the assigned GPU ID, worker ID, and assignment timestamp

**Raises:**
- `RuntimeError`: If the client is not started
- `WorkerTimeoutError`: If the request times out
- `Exception`: If GPU assignment fails

**Example:**
```python
# Request with default timeout
assignment = await client.request_gpu()

# Request with custom timeout
assignment = await client.request_gpu(timeout=120.0)
```

##### `async release_gpu(assignment: GPUAssignment) -> None`

Release a GPU assignment.

**Parameters:**
- `assignment` (`GPUAssignment`): The GPU assignment to release

**Raises:**
- `RuntimeError`: If the client is not started
- `ValueError`: If the assignment is invalid

**Example:**
```python
await client.release_gpu(assignment)
```

##### `get_pool_status() -> PoolStatus`

Get the current status of the GPU worker pool.

**Returns:**
- `PoolStatus`: Current pool metrics including GPU counts and worker counts

**Raises:**
- `RuntimeError`: If the client is not started

**Example:**
```python
status = client.get_pool_status()
print(f"Available GPUs: {status.available_gpus}")
print(f"Active workers: {status.active_workers}")
```

##### `get_detailed_metrics() -> Dict[str, Any]`

Get detailed pool metrics for monitoring and debugging.

**Returns:**
- `Dict[str, Any]`: Comprehensive pool metrics

**Raises:**
- `RuntimeError`: If the client is not started

**Example:**
```python
metrics = client.get_detailed_metrics()
print(f"GPU metrics: {metrics['gpu_metrics']}")
print(f"Thresholds: {metrics['thresholds']}")
```

##### `get_endpoints_info() -> Optional[List[Dict[str, Any]]]`

Get information about all configured endpoints (multi-endpoint mode only).

**Returns:**
- `List[Dict[str, Any]]`: List of endpoint information dictionaries containing:
  - `endpoint_id`: Unique endpoint identifier
  - `url`: Endpoint URL
  - `is_healthy`: Current health status
  - `total_gpus`: Total GPUs at endpoint
  - `available_gpus`: Available GPUs at endpoint
  - `response_time_ms`: Average response time
- `None`: If client is not started or in single-endpoint mode

**Example:**
```python
endpoints = client.get_endpoints_info()
if endpoints:
    for endpoint in endpoints:
        print(f"{endpoint['endpoint_id']}: {endpoint['available_gpus']}/{endpoint['total_gpus']} GPUs")
```

##### `get_error_recovery_status() -> Dict[str, Any]`

Get detailed error recovery and circuit breaker status (multi-endpoint mode only).

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `degradation_manager`: Degradation status and endpoint health
  - `recovery_orchestrator`: Recovery task information
  - `circuit_breaker_stats`: Per-endpoint circuit breaker states
  - `endpoint_health_summary`: Summary statistics

**Raises:**
- `RuntimeError`: If client is not started

**Example:**
```python
status = client.get_error_recovery_status()
degradation = status['degradation_manager']
print(f"Healthy endpoints: {len(degradation['healthy_endpoints'])}")
print(f"Degraded endpoints: {len(degradation['degraded_endpoints'])}")
```

##### `async trigger_endpoint_recovery(endpoint_id: str) -> bool`

Manually trigger recovery attempt for a specific endpoint (multi-endpoint mode only).

**Parameters:**
- `endpoint_id` (`str`): The endpoint identifier to recover

**Returns:**
- `bool`: True if recovery was triggered successfully

**Raises:**
- `RuntimeError`: If client is not started or not in multi-endpoint mode

**Example:**
```python
success = await client.trigger_endpoint_recovery("server1")
if success:
    print("Recovery triggered for server1")
```

##### `async queue_request_for_retry(request_func: Callable) -> Any`

Queue a request for retry when endpoints become available (multi-endpoint mode only).

**Parameters:**
- `request_func` (`Callable`): Async callable to execute when endpoints are available

**Returns:**
- Result of the request function when executed

**Raises:**
- `RuntimeError`: If client is not started or not in multi-endpoint mode

**Example:**
```python
# Queue a GPU request for when endpoints become available
async def request_with_custom_timeout():
    return await client.request_gpu(timeout=300.0)

result = await client.queue_request_for_retry(request_with_custom_timeout)
```

##### `print_error_recovery_summary() -> None`

Print a formatted summary of error recovery status to console (multi-endpoint mode only).

**Raises:**
- `RuntimeError`: If client is not started

**Example:**
```python
client.print_error_recovery_summary()
# Output:
# === Error Recovery Status ===
# Degradation Status: Partial (2/3 endpoints healthy)
# ...
```

##### `is_multi_endpoint_mode() -> bool`

Check if the client is configured for multi-endpoint mode.

**Returns:**
- `bool`: True if in multi-endpoint mode, False otherwise

**Example:**
```python
if client.is_multi_endpoint_mode():
    print("Client configured for multi-endpoint operation")
else:
    print("Client in single-endpoint mode")
```

#### Context Manager Support

The `GPUWorkerPoolClient` supports async context manager protocol for automatic resource management.

**Example:**
```python
async with GPUWorkerPoolClient() as client:
    assignment = await client.request_gpu()
    # Use the GPU
    await client.release_gpu(assignment)
# Client is automatically stopped
```

### GPUContextManager

Context manager for automatic GPU assignment and release.

```python
from gpu_worker_pool.client import GPUContextManager
```

#### Constructor

```python
def __init__(self, client: GPUWorkerPoolClient, timeout: Optional[float] = None)
```

**Parameters:**
- `client` (`GPUWorkerPoolClient`): Started GPUWorkerPoolClient instance
- `timeout` (`Optional[float]`): Optional timeout for GPU request

#### Usage

```python
async with GPUWorkerPoolClient() as client:
    async with GPUContextManager(client) as gpu_id:
        print(f"Using GPU {gpu_id}")
        # GPU is automatically released when exiting context
```

### Factory Functions

#### `gpu_worker_pool_client(**kwargs) -> AsyncContextManager[GPUWorkerPoolClient]`

Async context manager factory for convenient client creation.

**Parameters:**
- `**kwargs`: Arguments passed to `GPUWorkerPoolClient` constructor

**Returns:**
- `AsyncContextManager[GPUWorkerPoolClient]`: Context manager that yields a started client

**Example:**
```python
from gpu_worker_pool.client import gpu_worker_pool_client

async with gpu_worker_pool_client(memory_threshold=75.0) as client:
    assignment = await client.request_gpu()
    await client.release_gpu(assignment)
```

## Data Models

### GPUAssignment

Represents an assignment of a worker to a specific GPU.

```python
from gpu_worker_pool.models import GPUAssignment
```

**Attributes:**
- `gpu_id` (`int`): The assigned GPU ID
- `worker_id` (`str`): The worker identifier
- `assigned_at` (`datetime`): Timestamp when the assignment was made

**Example:**
```python
assignment = GPUAssignment(
    gpu_id=0,
    worker_id="worker-123",
    assigned_at=datetime.now()
)
```

### PoolStatus

Current status of the GPU worker pool.

```python
from gpu_worker_pool.models import PoolStatus
```

**Attributes:**
- `total_gpus` (`int`): Total number of GPUs in the pool
- `available_gpus` (`int`): Number of GPUs available for assignment
- `active_workers` (`int`): Number of currently active workers
- `blocked_workers` (`int`): Number of workers waiting in queue
- `gpu_assignments` (`Dict[int, List[WorkerInfo]]`): Current assignments per GPU

**Example:**
```python
status = client.get_pool_status()
utilization = (status.total_gpus - status.available_gpus) / status.total_gpus
print(f"Pool utilization: {utilization:.1%}")
```

### GPUInfo

Information about a single GPU's current state.

```python
from gpu_worker_pool.models import GPUInfo
```

**Attributes:**
- `gpu_id` (`int`): GPU identifier
- `name` (`str`): GPU name/model
- `memory_usage_percent` (`float`): Memory usage percentage (0-100)
- `utilization_percent` (`float`): GPU utilization percentage (0-100)

### GPUStats

Complete GPU statistics from the monitoring service.

```python
from gpu_worker_pool.models import GPUStats
```

**Attributes:**
- `gpu_count` (`int`): Number of GPUs
- `total_memory_mb` (`int`): Total memory across all GPUs in MB
- `total_used_memory_mb` (`int`): Total used memory across all GPUs in MB
- `average_utilization_percent` (`float`): Average utilization across all GPUs
- `gpus_summary` (`List[GPUInfo]`): Detailed information for each GPU
- `total_memory_usage_percent` (`float`): Total memory usage percentage
- `timestamp` (`str`): Timestamp of the statistics

### WorkerInfo

Information about a worker process requesting GPU resources.

```python
from gpu_worker_pool.models import WorkerInfo
```

**Attributes:**
- `id` (`str`): Worker identifier
- `enqueued_at` (`datetime`): When the worker was enqueued
- `callback` (`Callable[[int], None]`): Callback function for GPU assignment
- `on_error` (`Callable[[Exception], None]`): Callback function for errors

### Multi-Endpoint Data Models

#### GlobalGPUAssignment

GPU assignment with global GPU ID for multi-endpoint mode.

```python
from gpu_worker_pool.models import GlobalGPUAssignment
```

**Attributes:**
- `global_gpu_id` (`str`): Global GPU ID in format "endpoint_id:local_gpu_id"
- `endpoint_id` (`str`): Endpoint identifier hosting the GPU
- `local_gpu_id` (`int`): Local GPU ID on the endpoint
- `worker_id` (`str`): Worker identifier
- `assigned_at` (`datetime`): Assignment timestamp

**Example:**
```python
# In multi-endpoint mode, assignments use global IDs
assignment = GlobalGPUAssignment(
    global_gpu_id="server1:2",
    endpoint_id="server1",
    local_gpu_id=2,
    worker_id="worker-123",
    assigned_at=datetime.now()
)
```

#### MultiEndpointPoolStatus

Extended pool status for multi-endpoint configurations.

```python
from gpu_worker_pool.models import MultiEndpointPoolStatus
```

**Attributes:**
- Inherits all attributes from `PoolStatus`
- `total_endpoints` (`int`): Total number of configured endpoints
- `healthy_endpoints` (`int`): Number of currently healthy endpoints
- `endpoint_details` (`List[Dict[str, Any]]`): Per-endpoint statistics

**Example:**
```python
status = client.get_pool_status()  # Returns MultiEndpointPoolStatus in multi-endpoint mode
print(f"Total endpoints: {status.total_endpoints}")
print(f"Healthy endpoints: {status.healthy_endpoints}")
for endpoint in status.endpoint_details:
    print(f"{endpoint['endpoint_id']}: {endpoint['available_gpus']} GPUs available")
```

#### EndpointInfo

Information about a single endpoint in the multi-endpoint system.

```python
from gpu_worker_pool.models import EndpointInfo
```

**Attributes:**
- `endpoint_id` (`str`): Unique endpoint identifier
- `url` (`str`): Endpoint URL
- `is_healthy` (`bool`): Current health status
- `last_seen` (`datetime`): Last successful communication timestamp
- `total_gpus` (`int`): Total GPUs at this endpoint
- `available_gpus` (`int`): Available GPUs at this endpoint
- `response_time_ms` (`float`): Average response time in milliseconds

**Example:**
```python
endpoint = EndpointInfo(
    endpoint_id="server1",
    url="http://gpu-server-1:8000",
    is_healthy=True,
    last_seen=datetime.now(),
    total_gpus=4,
    available_gpus=3,
    response_time_ms=50.0
)
```

#### GlobalGPUInfo

GPU information with global identification for multi-endpoint systems.

```python
from gpu_worker_pool.models import GlobalGPUInfo
```

**Attributes:**
- `global_gpu_id` (`str`): Global GPU ID
- `endpoint_id` (`str`): Endpoint hosting this GPU
- `local_gpu_id` (`int`): Local GPU ID at endpoint
- `name` (`str`): GPU name/model
- `memory_usage_percent` (`float`): Memory usage percentage
- `utilization_percent` (`float`): GPU utilization percentage
- `is_available` (`bool`): Availability status

**Example:**
```python
gpu_info = GlobalGPUInfo(
    global_gpu_id="server1:0",
    endpoint_id="server1",
    local_gpu_id=0,
    name="NVIDIA Tesla V100",
    memory_usage_percent=25.0,
    utilization_percent=30.0,
    is_available=True
)
```

## Configuration

### EnvironmentConfigurationManager

Manages configuration from environment variables.

```python
from gpu_worker_pool.config import EnvironmentConfigurationManager
```

#### Methods

##### `get_service_endpoint() -> str`
Get the GPU service endpoint URL.

**Environment Variable:** `GPU_SERVICE_ENDPOINT`  
**Default:** `"http://localhost:8080"`

##### `get_memory_threshold() -> float`
Get the memory usage threshold percentage.

**Environment Variable:** `GPU_MEMORY_THRESHOLD_PERCENT`  
**Default:** `80.0`

##### `get_utilization_threshold() -> float`
Get the GPU utilization threshold percentage.

**Environment Variable:** `GPU_UTILIZATION_THRESHOLD_PERCENT`  
**Default:** `90.0`

##### `get_polling_interval() -> int`
Get the statistics polling interval in seconds.

**Environment Variable:** `GPU_POLLING_INTERVAL`  
**Default:** `5`

**Example:**
```python
config = EnvironmentConfigurationManager()
print(f"Service endpoint: {config.get_service_endpoint()}")
print(f"Memory threshold: {config.get_memory_threshold()}%")
```

### Environment Variables

#### Single-Endpoint Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GPU_SERVICE_ENDPOINT` | `str` | `"http://localhost:8080"` | Single GPU statistics service URL |
| `GPU_MEMORY_THRESHOLD_PERCENT` | `float` | `80.0` | Memory usage threshold (0-100%) |
| `GPU_UTILIZATION_THRESHOLD_PERCENT` | `float` | `90.0` | GPU utilization threshold (0-100%) |
| `GPU_POLLING_INTERVAL` | `int` | `5` | Statistics polling interval in seconds |

**Example:**
```bash
export GPU_SERVICE_ENDPOINT="https://gpu-service.company.com"
export GPU_MEMORY_THRESHOLD_PERCENT="75.0"
export GPU_UTILIZATION_THRESHOLD_PERCENT="85.0"
export GPU_POLLING_INTERVAL="3"
```

#### Multi-Endpoint Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GPU_STATS_SERVICE_ENDPOINTS` | `str` | `None` | Comma-separated list of GPU statistics service URLs |
| `GPU_LOAD_BALANCING_STRATEGY` | `str` | `"availability"` | Load balancing strategy: `availability`, `round_robin`, or `weighted` |
| `GPU_ENDPOINT_TIMEOUT` | `float` | `10.0` | Timeout for individual endpoint requests (seconds) |
| `GPU_ENDPOINT_MAX_RETRIES` | `int` | `3` | Maximum retries per endpoint before circuit breaker opens |
| `GPU_MEMORY_THRESHOLD_PERCENT` | `float` | `80.0` | Memory usage threshold (0-100%) |
| `GPU_UTILIZATION_THRESHOLD_PERCENT` | `float` | `90.0` | GPU utilization threshold (0-100%) |
| `GPU_POLLING_INTERVAL` | `int` | `5` | Statistics polling interval in seconds |

**Example:**
```bash
# Multi-endpoint configuration
export GPU_STATS_SERVICE_ENDPOINTS="http://gpu1:8000,http://gpu2:8000,http://gpu3:8000"
export GPU_LOAD_BALANCING_STRATEGY="availability"
export GPU_ENDPOINT_TIMEOUT="15.0"
export GPU_ENDPOINT_MAX_RETRIES="5"
export GPU_MEMORY_THRESHOLD_PERCENT="75.0"
export GPU_UTILIZATION_THRESHOLD_PERCENT="85.0"
export GPU_POLLING_INTERVAL="3"
```

**Priority Order:**
- `GPU_STATS_SERVICE_ENDPOINTS` takes precedence over `GPU_SERVICE_ENDPOINT`
- If both are set, multi-endpoint mode is used
- All threshold and polling interval settings apply to both modes

## Exceptions

### WorkerTimeoutError

Raised when a worker request times out.

```python
from gpu_worker_pool.worker_pool_manager import WorkerTimeoutError
```

**Example:**
```python
try:
    assignment = await client.request_gpu(timeout=60.0)
except WorkerTimeoutError as e:
    print(f"GPU request timed out: {e}")
```

### StaleAssignmentError

Raised when a worker assignment has become stale.

```python
from gpu_worker_pool.worker_pool_manager import StaleAssignmentError
```

### ServiceUnavailableError

Raised when the GPU service is unavailable after retries.

```python
from gpu_worker_pool.http_client import ServiceUnavailableError
```

### RetryableError

Base class for errors that should trigger retry logic.

```python
from gpu_worker_pool.http_client import RetryableError
```

#### Subclasses:
- `NetworkRetryableError`: Network-related error that should be retried
- `ServiceRetryableError`: Service-related error that should be retried

## Utilities

### Configuration Validation

```python
from examples.config_examples import ConfigurationValidator

validator = ConfigurationValidator()

# Validate thresholds
result = validator.validate_thresholds(75.0, 85.0)
if result["valid"]:
    print("Configuration is valid")
else:
    print(f"Errors: {result['errors']}")

# Validate endpoint
result = validator.validate_service_endpoint("https://gpu-service.com")
if result["valid"]:
    print("Endpoint is valid")
```

### Metrics Collection

```python
# Get detailed metrics
metrics = client.get_detailed_metrics()

# Access GPU-level metrics
for gpu in metrics["gpu_metrics"]:
    print(f"GPU {gpu['gpu_id']}: {gpu['memory_usage_percent']}% memory")

# Access thresholds
thresholds = metrics["thresholds"]
print(f"Memory threshold: {thresholds['memory_threshold_percent']}%")
```

### Health Monitoring

```python
# Check pool health
status = client.get_pool_status()

# Calculate utilization
if status.total_gpus > 0:
    utilization = (status.total_gpus - status.available_gpus) / status.total_gpus
    if utilization > 0.9:
        print("Warning: High GPU utilization")

# Check for blocked workers
if status.blocked_workers > 0:
    print(f"Warning: {status.blocked_workers} workers are waiting for GPUs")
```

## Error Handling Patterns

### Timeout Handling

```python
import asyncio

try:
    assignment = await client.request_gpu(timeout=300.0)
    # Use GPU
    await client.release_gpu(assignment)
except WorkerTimeoutError:
    print("No GPU available within timeout")
except asyncio.TimeoutError:
    print("Operation timed out")
```

### Service Failure Handling

```python
try:
    async with GPUWorkerPoolClient() as client:
        assignment = await client.request_gpu()
        await client.release_gpu(assignment)
except ServiceUnavailableError:
    print("GPU service is currently unavailable")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Graceful Degradation

```python
# Check service health before making requests
try:
    status = client.get_pool_status()
    if status.available_gpus == 0:
        print("No GPUs available, deferring work")
        return
    
    assignment = await client.request_gpu(timeout=60.0)
    # Use GPU
    await client.release_gpu(assignment)
    
except WorkerTimeoutError:
    print("GPU request timed out, will retry later")
except Exception as e:
    print(f"GPU allocation failed: {e}")
```

## Performance Considerations

### Optimal Configuration

- **Memory Threshold**: Set based on your application's memory requirements
  - Conservative: 70-75% for stability
  - Balanced: 80-85% for good utilization
  - Aggressive: 85-90% for maximum utilization

- **Utilization Threshold**: Set based on acceptable performance degradation
  - Conservative: 75-80% to avoid contention
  - Balanced: 85-90% for good utilization
  - Aggressive: 90-95% for maximum utilization

- **Polling Interval**: Balance between responsiveness and overhead
  - Fast: 1-2 seconds for interactive workloads
  - Normal: 3-5 seconds for most applications
  - Slow: 10+ seconds for batch processing

### Resource Management

```python
# Use context managers for automatic cleanup
async with GPUWorkerPoolClient() as client:
    async with GPUContextManager(client) as gpu_id:
        # GPU is automatically managed
        pass

# Monitor pool utilization
status = client.get_pool_status()
if status.blocked_workers > status.total_gpus:
    print("Consider scaling GPU resources")
```

### Concurrent Usage

```python
import asyncio

async def worker_task(client, task_id):
    async with GPUContextManager(client) as gpu_id:
        print(f"Task {task_id} using GPU {gpu_id}")
        await asyncio.sleep(1.0)  # Simulate work

async def main():
    async with GPUWorkerPoolClient() as client:
        # Run multiple workers concurrently
        tasks = [worker_task(client, i) for i in range(5)]
        await asyncio.gather(*tasks)

asyncio.run(main())
```