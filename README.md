# GPU Worker Pool

A comprehensive Python library for intelligent GPU resource allocation and management in multi-worker environments. The GPU Worker Pool automatically manages GPU assignments based on real-time resource usage, ensuring optimal utilization while preventing resource conflicts.

## Features

### Client Library
- **Intelligent Resource Allocation**: Automatically assigns GPUs based on memory usage and utilization thresholds
- **Multi-Endpoint Support**: Connect to multiple GPU statistics servers with automatic load balancing
- **Worker Queue Management**: Handles worker blocking and unblocking when resources become available
- **Real-time Monitoring**: Continuously monitors GPU statistics from remote services
- **Fault Tolerance**: Robust error handling with retry logic and circuit breaker patterns
- **Load Balancing**: Multiple strategies including availability-based, round-robin, and weighted distribution
- **Error Recovery**: Automatic failover and graceful degradation when endpoints become unavailable
- **Flexible Configuration**: Environment variables and programmatic configuration support
- **Context Manager Support**: Automatic resource cleanup using Python's `with` statement
- **Production Ready**: Comprehensive logging, metrics, and health checking
- **Backward Compatibility**: Seamless compatibility with existing single-endpoint configurations

### GPU Statistics Server
- **Real-time GPU Monitoring**: Live GPU statistics using nvidia-smi
- **GPU Filtering**: Support for CUDA_VISIBLE_DEVICES environment variable to filter visible GPUs
- **REST API**: FastAPI-based server with comprehensive endpoints
- **Interactive Documentation**: Built-in Swagger UI and ReDoc documentation
- **Configurable Caching**: Adjustable refresh intervals to optimize performance
- **Error Handling**: Graceful handling of missing nvidia-smi or unsupported features
- **CORS Support**: Configurable Cross-Origin Resource Sharing
- **Environment Configuration**: Full configuration via environment variables

## Quick Start

### Installation

#### From Wheel Package

```bash
pip install gpu_worker_pool-1.0.0-py3-none-any.whl
```

#### From Source

```bash
git clone <repository-url>
cd gpu-worker-pool
pip install -e .
```

#### Verify Installation

```bash
python -c "import gpu_worker_pool; print(f'v{gpu_worker_pool.__version__} installed')"
```

### GPU Statistics Server

The GPU Worker Pool includes a built-in GPU statistics server that provides real-time GPU monitoring via a REST API.

#### Starting the Server

```bash
# Method 1: Direct Python execution (recommended)
python -m gpu_worker_pool.gpu_server

# Method 2: Using uvicorn
uvicorn gpu_worker_pool.gpu_server:app --host 0.0.0.0 --port 8000 --reload

# Method 3: Standalone script
python gpu_worker_pool/gpu_server.py
```

#### Server Configuration

Configure the server using environment variables:

```bash
# Server settings
export HOST=0.0.0.0
export PORT=8000
export LOG_LEVEL=info
export REFRESH_INTERVAL=1.0

# GPU filtering (optional)
export CUDA_VISIBLE_DEVICES="0,1,2"  # Show only GPUs 0, 1, and 2

# Start the server
python -m gpu_worker_pool.gpu_server
```

#### GPU Filtering

The server supports GPU filtering using the `CUDA_VISIBLE_DEVICES` environment variable, allowing you to control which GPUs are visible to the monitoring system:

```bash
# Show only specific GPUs
export CUDA_VISIBLE_DEVICES="0,2"
python -m gpu_worker_pool.gpu_server

# Hide all GPUs (useful for testing)
export CUDA_VISIBLE_DEVICES=""
python -m gpu_worker_pool.gpu_server

# Docker example with GPU filtering
docker run -e CUDA_VISIBLE_DEVICES="0,1" your-gpu-server-image
```

**Supported formats:**
- Comma-separated: `"0,1,2"`
- Space-separated: `"0 1 2"`
- Mixed: `"0,1 2"`
- Single GPU: `"1"`
- No GPUs: `""` (empty string)

#### API Endpoints

Once running, the server provides several endpoints:

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **GPU Statistics**: http://localhost:8000/gpu/stats
- **GPU Summary**: http://localhost:8000/gpu/summary
- **Health Check**: http://localhost:8000/health

#### Example API Usage

```bash
# Get GPU summary
curl http://localhost:8000/gpu/summary

# Check server health
curl http://localhost:8000/health

# Get detailed GPU statistics
curl http://localhost:8000/gpu/stats
```

### Basic Client Usage

```python
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient

async def main():
    # Using context manager (recommended)
    async with GPUWorkerPoolClient() as client:
        # Request a GPU
        assignment = await client.request_gpu()
        print(f"Assigned GPU {assignment.gpu_id}")
        
        # Use the GPU for your work
        # ... your GPU computation here ...
        
        # Release the GPU
        await client.release_gpu(assignment)

asyncio.run(main())
```

### Advanced Usage with Configuration

```python
import asyncio
from gpu_worker_pool.client import gpu_worker_pool_client, GPUContextManager

async def main():
    # Configure for production environment
    config = {
        "memory_threshold": 75.0,           # Don't assign if GPU memory > 75%
        "utilization_threshold": 85.0,      # Don't assign if GPU util > 85%
        "worker_timeout": 600.0,            # 10 minute timeout for workers
        "service_endpoint": "https://gpu-service.company.com"
    }
    
    async with gpu_worker_pool_client(**config) as client:
        # Automatic GPU assignment and release
        async with GPUContextManager(client) as gpu_id:
            print(f"Automatically assigned GPU {gpu_id}")
            # GPU is automatically released when exiting context

asyncio.run(main())
```

## Multi-Endpoint Support

The GPU Worker Pool supports connecting to multiple GPU statistics servers simultaneously, providing automatic load balancing, failover, and increased resource capacity. This is ideal for distributed GPU clusters and high-availability deployments.

### Multi-Endpoint Configuration

Configure multiple endpoints using comma-separated URLs:

```python
import asyncio
from gpu_worker_pool.client import GPUWorkerPoolClient

async def main():
    # Multi-endpoint configuration
    client = GPUWorkerPoolClient(
        service_endpoints="http://gpu-server-1:8000,http://gpu-server-2:8000,http://gpu-server-3:8000",
        load_balancing_strategy="availability"  # Choose best available endpoint
    )
    
    async with client:
        # Request GPU from the best available endpoint
        assignment = await client.request_gpu()
        print(f"Assigned GPU {assignment.global_gpu_id} from endpoint {assignment.endpoint_id}")
        
        # GPU is automatically released when done
        await client.release_gpu(assignment)

asyncio.run(main())
```

### Environment Variable Configuration

```bash
# Multiple endpoints
export GPU_STATS_SERVICE_ENDPOINTS="http://gpu-node-1:8000,http://gpu-node-2:8000,http://gpu-node-3:8000"

# Load balancing strategy
export GPU_LOAD_BALANCING_STRATEGY="availability"

# Endpoint-specific settings
export GPU_ENDPOINT_TIMEOUT="10.0"
export GPU_ENDPOINT_MAX_RETRIES="3"
```

### Load Balancing Strategies

Choose the optimal load balancing strategy for your use case:

#### 1. Availability-Based (Recommended)
Selects endpoints with the highest percentage of available GPUs:

```python
client = GPUWorkerPoolClient(
    service_endpoints="http://server1:8000,http://server2:8000",
    load_balancing_strategy="availability"
)
```

**Best for:** Production environments where you want optimal resource utilization.

#### 2. Round-Robin
Distributes requests evenly across all healthy endpoints:

```python
client = GPUWorkerPoolClient(
    service_endpoints="http://server1:8000,http://server2:8000",
    load_balancing_strategy="round_robin"
)
```

**Best for:** Development environments or when endpoints have similar capabilities.

#### 3. Weighted
Distributes requests proportionally based on total GPU capacity:

```python
client = GPUWorkerPoolClient(
    service_endpoints="http://server1:8000,http://server2:8000",
    load_balancing_strategy="weighted"
)
```

**Best for:** Heterogeneous clusters with different GPU counts per server.

### Multi-Endpoint Features

#### Global GPU IDs
In multi-endpoint mode, GPUs are identified using global IDs in the format `endpoint_id:local_gpu_id`:

```python
async with GPUWorkerPoolClient(service_endpoints="...") as client:
    assignment = await client.request_gpu()
    print(f"Global GPU ID: {assignment.global_gpu_id}")  # e.g., "server1:2"
    print(f"Endpoint ID: {assignment.endpoint_id}")      # e.g., "server1"
    print(f"Local GPU ID: {assignment.local_gpu_id}")    # e.g., 2
```

#### Unified Pool Status
Get aggregated status across all endpoints:

```python
async with GPUWorkerPoolClient(service_endpoints="...") as client:
    status = client.get_pool_status()
    
    # Multi-endpoint status includes additional information
    print(f"Total endpoints: {status.total_endpoints}")
    print(f"Healthy endpoints: {status.healthy_endpoints}")
    print(f"Total GPUs across all endpoints: {status.total_gpus}")
    print(f"Available GPUs: {status.available_gpus}")
    
    # Per-endpoint breakdown
    for endpoint in status.endpoint_details:
        print(f"Endpoint {endpoint['endpoint_id']}: {endpoint['available_gpus']}/{endpoint['total_gpus']} GPUs")
```

#### Error Recovery and Failover
Automatic handling of endpoint failures:

```python
async with GPUWorkerPoolClient(service_endpoints="...") as client:
    # Check error recovery status
    recovery_status = client.get_error_recovery_status()
    print(f"Healthy endpoints: {len(recovery_status['healthy_endpoints'])}")
    print(f"Degraded endpoints: {len(recovery_status['degraded_endpoints'])}")
    
    # Manual recovery trigger (optional)
    await client.trigger_endpoint_recovery("server1")
    
    # Queue requests during degradation
    await client.queue_request_for_retry(lambda: client.request_gpu())
```

### Multi-Endpoint Monitoring

#### Detailed Metrics
Get comprehensive metrics across all endpoints:

```python
async with GPUWorkerPoolClient(service_endpoints="...") as client:
    metrics = client.get_detailed_metrics()
    
    # Overall summary
    print(f"Pool efficiency: {metrics['pool_summary']['efficiency_percent']}%")
    
    # Per-endpoint details
    for endpoint_id, endpoint_metrics in metrics['endpoints']['details'].items():
        print(f"Endpoint {endpoint_id}:")
        print(f"  GPUs: {endpoint_metrics['available_gpus']}/{endpoint_metrics['total_gpus']}")
        print(f"  Response time: {endpoint_metrics['avg_response_time_ms']}ms")
        print(f"  Health: {'Healthy' if endpoint_metrics['is_healthy'] else 'Degraded'}")
    
    # Load balancer statistics
    lb_stats = metrics['load_balancer']
    print(f"Load balancing strategy: {lb_stats['strategy_name']}")
    print(f"Total requests: {lb_stats['total_requests']}")
    print(f"Success rate: {lb_stats['success_rate']}%")
```

#### Error Recovery Summary
Monitor system health and recovery status:

```python
async with GPUWorkerPoolClient(service_endpoints="...") as client:
    # Print comprehensive recovery status
    client.print_error_recovery_summary()
    
    # Or get detailed status for custom monitoring
    status = client.get_error_recovery_status()
    
    # Check if system is fully operational
    if not status['degradation_manager']['is_fully_degraded']:
        print("System operating normally")
    else:
        print(f"System degraded: {len(status['degraded_endpoints'])} endpoints unavailable")
```

### Migration from Single-Endpoint

Migrating from single-endpoint to multi-endpoint configuration is seamless:

```python
# Before (single-endpoint)
client = GPUWorkerPoolClient(service_endpoint="http://gpu-server:8000")

# After (multi-endpoint) - just change the parameter name and add more endpoints
client = GPUWorkerPoolClient(service_endpoints="http://gpu-server-1:8000,http://gpu-server-2:8000")

# All other code remains the same!
async with client:
    assignment = await client.request_gpu()  # Now uses global GPU IDs automatically
    await client.release_gpu(assignment)
```

### Best Practices for Multi-Endpoint

#### 1. Use Availability-Based Load Balancing
For production workloads, use availability-based load balancing to maximize resource utilization:

```python
client = GPUWorkerPoolClient(
    service_endpoints="http://server1:8000,http://server2:8000,http://server3:8000",
    load_balancing_strategy="availability"
)
```

#### 2. Monitor Endpoint Health
Regularly check endpoint health and system status:

```python
async with client:
    status = client.get_error_recovery_status()
    if status['endpoint_health_summary']['healthy_count'] < 2:
        print("Warning: Low endpoint availability, consider investigating")
```

#### 3. Configure Appropriate Timeouts
Set endpoint timeouts based on your network characteristics:

```bash
export GPU_ENDPOINT_TIMEOUT="15.0"        # 15 seconds for slower networks
export GPU_ENDPOINT_MAX_RETRIES="5"       # More retries for unreliable networks
```

#### 4. Use Circuit Breakers
The system automatically uses circuit breakers to prevent cascading failures. Monitor circuit breaker status:

```python
recovery_status = client.get_error_recovery_status()
for endpoint_id, cb_stats in recovery_status['circuit_breaker_stats'].items():
    if cb_stats['state'] == 'open':
        print(f"Circuit breaker open for {endpoint_id}, failure rate: {cb_stats['failure_rate']}%")
```

## Configuration

### Environment Variables

The GPU Worker Pool can be configured using environment variables:

#### Single-Endpoint Configuration
```bash
# GPU service endpoint (single)
export GPU_SERVICE_ENDPOINT="http://gpu-service:8080"

# Resource thresholds (0-100%)
export GPU_MEMORY_THRESHOLD_PERCENT="80.0"
export GPU_UTILIZATION_THRESHOLD_PERCENT="90.0"

# Polling interval in seconds
export GPU_POLLING_INTERVAL="5"
```

#### Multi-Endpoint Configuration
```bash
# GPU service endpoints (multiple, comma-separated)
export GPU_STATS_SERVICE_ENDPOINTS="http://gpu-node-1:8000,http://gpu-node-2:8000,http://gpu-node-3:8000"

# Load balancing strategy
export GPU_LOAD_BALANCING_STRATEGY="availability"  # or "round_robin", "weighted"

# Endpoint management
export GPU_ENDPOINT_TIMEOUT="10.0"                       # Timeout per endpoint (seconds)
export GPU_ENDPOINT_MAX_RETRIES="3"                      # Max retries per endpoint

# Resource thresholds (same as single-endpoint)
export GPU_MEMORY_THRESHOLD_PERCENT="80.0"
export GPU_UTILIZATION_THRESHOLD_PERCENT="90.0"
export GPU_POLLING_INTERVAL="5"
```

#### Priority Order
When both single and multi-endpoint variables are set:
1. `GPU_STATS_SERVICE_ENDPOINTS` (multi-endpoint) takes precedence over `GPU_SERVICE_ENDPOINT`
2. If `GPU_STATS_SERVICE_ENDPOINTS` contains only one endpoint, operates in single-endpoint mode
3. All other configuration variables apply to both modes

### Programmatic Configuration

#### Single-Endpoint Configuration
```python
from gpu_worker_pool.client import GPUWorkerPoolClient

client = GPUWorkerPoolClient(
    service_endpoint="http://localhost:8080",
    memory_threshold=70.0,              # Conservative memory limit
    utilization_threshold=80.0,         # Conservative utilization limit
    worker_timeout=300.0,               # 5 minute timeout
    request_timeout=10.0,               # 10 second HTTP timeout
    polling_interval=3                  # Poll every 3 seconds
)
```

#### Multi-Endpoint Configuration
```python
from gpu_worker_pool.client import GPUWorkerPoolClient

client = GPUWorkerPoolClient(
    service_endpoints="http://gpu-1:8000,http://gpu-2:8000,http://gpu-3:8000",
    load_balancing_strategy="availability",    # Load balancing strategy
    memory_threshold=70.0,                           # Conservative memory limit
    utilization_threshold=80.0,                      # Conservative utilization limit
    worker_timeout=300.0,                            # 5 minute timeout
    request_timeout=10.0,                            # 10 second HTTP timeout
    polling_interval=3                               # Poll every 3 seconds
)
```

#### Parameter Priority
When both `service_endpoint` and `service_endpoints` are provided:
- `service_endpoints` takes precedence
- `service_endpoint` is ignored
- All other parameters apply regardless of mode

## Configuration Scenarios

### Production Environment (Single-Endpoint)
```python
# Conservative thresholds for stability
config = {
    "service_endpoint": "http://gpu-prod:8000",
    "memory_threshold": 70.0,
    "utilization_threshold": 80.0,
    "worker_timeout": 600.0,
    "polling_interval": 5
}
```

### Production Environment (Multi-Endpoint)
```python
# High-availability cluster with load balancing
config = {
    "service_endpoints": "http://gpu-prod-1:8000,http://gpu-prod-2:8000,http://gpu-prod-3:8000",
    "load_balancing_strategy": "availability",
    "memory_threshold": 70.0,
    "utilization_threshold": 80.0,
    "worker_timeout": 600.0,
    "polling_interval": 5
}
```

### Development Environment (Multi-Endpoint)
```python
# Balanced thresholds for development with multiple test servers
config = {
    "service_endpoints": "http://gpu-dev-1:8000,http://gpu-dev-2:8000",
    "load_balancing_strategy": "round_robin",  # Even distribution for testing
    "memory_threshold": 75.0,
    "utilization_threshold": 85.0,
    "worker_timeout": 60.0,
    "polling_interval": 2
}
```

### High-Throughput Batch Processing (Multi-Endpoint)
```python
# Aggressive thresholds with weighted load balancing for heterogeneous cluster
config = {
    "service_endpoints": "http://gpu-batch-1:8000,http://gpu-batch-2:8000,http://gpu-batch-3:8000",
    "load_balancing_strategy": "weighted",  # Balance by GPU capacity
    "memory_threshold": 85.0,
    "utilization_threshold": 90.0,
    "worker_timeout": 1800.0,  # 30 minutes for long jobs
    "polling_interval": 5
}
```

### Multi-Tenant Environment (Multi-Endpoint)
```python
# Very conservative for fair sharing across large cluster
config = {
    "service_endpoints": "http://gpu-mt-1:8000,http://gpu-mt-2:8000,http://gpu-mt-3:8000,http://gpu-mt-4:8000",
    "load_balancing_strategy": "availability",  # Optimal resource sharing
    "memory_threshold": 60.0,
    "utilization_threshold": 75.0,
    "worker_timeout": 600.0,
    "polling_interval": 3
}
```

### Edge Computing (Multi-Endpoint with Failover)
```python
# Edge nodes with automatic failover to cloud
config = {
    "service_endpoints": "http://edge-gpu-1:8000,http://edge-gpu-2:8000,http://cloud-gpu:8000",
    "load_balancing_strategy": "availability",
    "memory_threshold": 75.0,
    "utilization_threshold": 85.0,
    "worker_timeout": 300.0,
    "polling_interval": 2  # Faster polling for edge scenarios
}
```

## API Reference

### GPUWorkerPoolClient

The main client interface for the GPU Worker Pool system, supporting both single-endpoint and multi-endpoint configurations.

#### Constructor

```python
GPUWorkerPoolClient(
    service_endpoint: Optional[str] = None,                    # Single-endpoint mode
    service_endpoints: Optional[str] = None,                   # Multi-endpoint mode  
    load_balancing_strategy: Optional[str] = None,             # Multi-endpoint only
    memory_threshold: Optional[float] = None,
    utilization_threshold: Optional[float] = None,
    polling_interval: Optional[int] = None,
    worker_timeout: float = 300.0,
    request_timeout: float = 30.0
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `service_endpoint` | `Optional[str]` | `None` | Single GPU statistics service URL (overrides `GPU_SERVICE_ENDPOINT` env var) |
| `service_endpoints` | `Optional[str]` | `None` | Comma-separated URLs for multi-endpoint mode (overrides `GPU_STATS_SERVICE_ENDPOINTS` env var) |
| `load_balancing_strategy` | `Optional[str]` | `"availability"` | Load balancing strategy: `"availability"`, `"round_robin"`, or `"weighted"` (overrides `GPU_LOAD_BALANCING_STRATEGY` env var) |
| `memory_threshold` | `Optional[float]` | `80.0` | Memory usage threshold percentage 0-100% (overrides `GPU_MEMORY_THRESHOLD_PERCENT` env var) |
| `utilization_threshold` | `Optional[float]` | `90.0` | GPU utilization threshold percentage 0-100% (overrides `GPU_UTILIZATION_THRESHOLD_PERCENT` env var) |
| `polling_interval` | `Optional[int]` | `5` | Statistics polling interval in seconds (overrides `GPU_POLLING_INTERVAL` env var) |
| `worker_timeout` | `float` | `300.0` | Timeout for GPU requests in seconds |
| `request_timeout` | `float` | `30.0` | HTTP request timeout in seconds |

**Mode Selection:**
- If `service_endpoints` is provided: **Multi-endpoint mode**
- If only `service_endpoint` is provided: **Single-endpoint mode**  
- If both are provided: `service_endpoints` takes precedence
- If neither is provided: Uses environment variables with same precedence rules

#### Methods

##### `async start()`
Start the client and all underlying components.

**Raises:**
- `RuntimeError`: If client is already started
- `Exception`: If initialization fails

##### `async stop()`
Stop the client and cleanup all resources.

##### `async request_gpu(timeout: Optional[float] = None) -> GPUAssignment`
Request a GPU assignment.

**Parameters:**
- `timeout`: Optional timeout in seconds

**Returns:**
- `GPUAssignment`: Contains GPU ID, worker ID, and assignment timestamp

**Raises:**
- `RuntimeError`: If client is not started
- `WorkerTimeoutError`: If request times out
- `Exception`: If assignment fails

##### `async release_gpu(assignment: GPUAssignment)`
Release a GPU assignment.

**Parameters:**
- `assignment`: The GPU assignment to release

**Raises:**
- `RuntimeError`: If client is not started
- `ValueError`: If assignment is invalid

##### `get_pool_status() -> PoolStatus`
Get current pool status and metrics.

**Returns:**
- `PoolStatus`: Contains GPU counts, worker counts, and assignments

##### `get_detailed_metrics() -> Dict[str, Any]`
Get detailed pool metrics for monitoring.

**Returns:**
- `Dict`: Comprehensive metrics including GPU details and thresholds

##### `get_endpoints_info() -> Optional[List[Dict[str, Any]]]`
Get information about all configured endpoints (multi-endpoint mode only).

**Returns:**
- `List[Dict]`: Endpoint information including health status, GPU counts, and response times
- `None`: If client is not started or in single-endpoint mode

**Example:**
```python
endpoints = client.get_endpoints_info()
for endpoint in endpoints:
    print(f"{endpoint['endpoint_id']}: {endpoint['available_gpus']}/{endpoint['total_gpus']} GPUs")
```

##### `get_error_recovery_status() -> Dict[str, Any]`
Get detailed error recovery and circuit breaker status (multi-endpoint mode only).

**Returns:**
- `Dict`: Contains degradation status, circuit breaker states, and recovery information

**Raises:**
- `RuntimeError`: If client is not started

**Example:**
```python
status = client.get_error_recovery_status()
print(f"Healthy endpoints: {len(status['degradation_manager']['healthy_endpoints'])}")
print(f"Degraded endpoints: {len(status['degradation_manager']['degraded_endpoints'])}")
```

##### `async trigger_endpoint_recovery(endpoint_id: str) -> bool`
Manually trigger recovery attempt for a specific endpoint (multi-endpoint mode only).

**Parameters:**
- `endpoint_id`: The endpoint identifier to recover

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
- `request_func`: Async callable to execute when endpoints are available

**Returns:**
- Result of the request function

**Raises:**
- `RuntimeError`: If client is not started or not in multi-endpoint mode

**Example:**
```python
# Queue a GPU request for when endpoints become available
result = await client.queue_request_for_retry(
    lambda: client.request_gpu(timeout=300.0)
)
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
```

#### Context Manager Support

```python
async with GPUWorkerPoolClient() as client:
    # Client is automatically started and stopped
    assignment = await client.request_gpu()
    await client.release_gpu(assignment)
```

### GPUContextManager

Context manager for automatic GPU assignment and release.

```python
async with GPUWorkerPoolClient() as client:
    async with GPUContextManager(client) as gpu_id:
        # GPU is automatically assigned
        print(f"Using GPU {gpu_id}")
        # GPU is automatically released when exiting
```

### Factory Functions

#### `gpu_worker_pool_client(**kwargs)`
Async context manager factory for convenient client creation.

```python
async with gpu_worker_pool_client(memory_threshold=75.0) as client:
    # Use client
    pass
```

## Data Models

### GPUAssignment
Represents a GPU assignment to a worker.

```python
@dataclass
class GPUAssignment:
    gpu_id: int              # Assigned GPU ID
    worker_id: str           # Worker identifier
    assigned_at: datetime    # Assignment timestamp
```

### PoolStatus
Current status of the GPU worker pool.

```python
@dataclass
class PoolStatus:
    total_gpus: int                                    # Total available GPUs
    available_gpus: int                                # GPUs available for assignment
    active_workers: int                                # Currently active workers
    blocked_workers: int                               # Workers waiting in queue
    gpu_assignments: Dict[int, List[WorkerInfo]]       # Current assignments per GPU
```

### GPUInfo
Information about a single GPU's current state.

```python
@dataclass
class GPUInfo:
    gpu_id: int                    # GPU identifier
    name: str                      # GPU name/model
    memory_usage_percent: float    # Memory usage percentage (0-100)
    utilization_percent: float     # GPU utilization percentage (0-100)
```

### Multi-Endpoint Data Models

#### GlobalGPUAssignment
GPU assignment with global GPU ID for multi-endpoint mode.

```python
@dataclass
class GlobalGPUAssignment:
    global_gpu_id: str           # Global GPU ID (format: "endpoint_id:local_gpu_id")
    endpoint_id: str             # Endpoint identifier
    local_gpu_id: int            # Local GPU ID on the endpoint
    worker_id: str               # Worker identifier
    assigned_at: datetime        # Assignment timestamp
```

#### MultiEndpointPoolStatus
Extended pool status for multi-endpoint mode.

```python
@dataclass
class MultiEndpointPoolStatus(PoolStatus):
    total_endpoints: int         # Total number of configured endpoints
    healthy_endpoints: int       # Number of healthy endpoints
    endpoint_details: List[Dict] # Per-endpoint statistics
```

#### EndpointInfo
Information about a single endpoint.

```python
@dataclass
class EndpointInfo:
    endpoint_id: str             # Unique endpoint identifier
    url: str                     # Endpoint URL
    is_healthy: bool             # Current health status
    last_seen: datetime          # Last successful communication
    total_gpus: int              # Total GPUs at this endpoint
    available_gpus: int          # Available GPUs at this endpoint
    response_time_ms: float      # Average response time in milliseconds
```

#### GlobalGPUInfo
GPU information with global identification.

```python
@dataclass
class GlobalGPUInfo:
    global_gpu_id: str           # Global GPU ID
    endpoint_id: str             # Endpoint hosting this GPU
    local_gpu_id: int            # Local GPU ID at endpoint
    name: str                    # GPU name/model
    memory_usage_percent: float  # Memory usage percentage
    utilization_percent: float   # GPU utilization percentage
    is_available: bool           # Availability status
```

## Error Handling

The GPU Worker Pool includes comprehensive error handling:

### Common Exceptions

- `WorkerTimeoutError`: Raised when a GPU request times out
- `StaleAssignmentError`: Raised when an assignment becomes stale
- `ServiceUnavailableError`: Raised when the GPU service is unavailable
- `ValueError`: Raised for invalid parameters or assignments

### Error Recovery

The system automatically handles:
- Network failures with exponential backoff
- Service unavailability with circuit breaker pattern
- Invalid responses with retry logic
- Resource exhaustion with worker queuing

## Monitoring and Logging

### Structured Logging

The system provides structured logging with contextual information:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Logs include worker IDs, GPU IDs, timestamps, and operation context
```

### Metrics Collection

Get detailed metrics for monitoring:

```python
async with GPUWorkerPoolClient() as client:
    metrics = client.get_detailed_metrics()
    print(f"Total GPUs: {metrics['total_gpus']}")
    print(f"Available GPUs: {metrics['available_gpus']}")
    print(f"Active workers: {metrics['active_workers']}")
    print(f"Thresholds: {metrics['thresholds']}")
```

### Health Checking

Monitor system health:

```python
status = client.get_pool_status()
if status.available_gpus == 0 and status.blocked_workers > 0:
    print("Warning: All GPUs are busy, workers are queued")
```

## Best Practices

### 1. Use Context Managers
Always use context managers for automatic resource cleanup:

```python
# Good
async with GPUWorkerPoolClient() as client:
    async with GPUContextManager(client) as gpu_id:
        # Use GPU
        pass

# Avoid manual management
client = GPUWorkerPoolClient()
await client.start()
assignment = await client.request_gpu()
# ... easy to forget cleanup
```

### 2. Configure Appropriate Thresholds
Choose thresholds based on your use case:

- **Conservative (Production)**: 70% memory, 80% utilization
- **Balanced (Development)**: 75% memory, 85% utilization  
- **Aggressive (Batch)**: 85% memory, 90% utilization

### 3. Handle Timeouts Gracefully
Set appropriate timeouts for your workload:

```python
try:
    assignment = await client.request_gpu(timeout=300.0)  # 5 minutes
    # Use GPU
except WorkerTimeoutError:
    print("No GPU available within timeout, try again later")
```

### 4. Monitor Pool Status
Regularly check pool status for capacity planning:

```python
status = client.get_pool_status()
utilization = (status.total_gpus - status.available_gpus) / status.total_gpus
if utilization > 0.8:
    print("High GPU utilization, consider scaling")
```

### 5. Use Environment Variables for Configuration
Configure via environment variables for different environments:

```bash
# Production
export GPU_MEMORY_THRESHOLD_PERCENT="70.0"
export GPU_UTILIZATION_THRESHOLD_PERCENT="80.0"

# Development  
export GPU_MEMORY_THRESHOLD_PERCENT="75.0"
export GPU_UTILIZATION_THRESHOLD_PERCENT="85.0"
```

## Documentation

- **[API Reference](docs/API.md)**: Complete API documentation with all methods and parameters
- **[Migration Guide](docs/MIGRATION_GUIDE.md)**: Step-by-step guide for migrating to multi-endpoint
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)**: Common issues and solutions
- **[User Guide](docs/USER_GUIDE.md)**: Comprehensive guide with usage patterns and best practices

## Examples

See the `examples/` directory for complete working examples:

- `basic_client_usage.py`: Basic usage patterns, context managers, and simple multi-endpoint
- `multi_endpoint_demo.py`: Configuration examples and patterns (no running code)
- `multi_endpoint_usage.py`: Comprehensive multi-endpoint examples with all features
- `config_examples.py`: Configuration examples and validation
- Integration tests: `test_integration_error_handling.py`

**Running Examples:**
```bash
# Easy way: Use the helper script (recommended)
python run_examples.py

# Manual way: Use local development code
PYTHONPATH=. python examples/basic_client_usage.py
PYTHONPATH=. python examples/multi_endpoint_demo.py

# Or install in development mode first
pip install -e .
python examples/basic_client_usage.py
```

## Testing

Run the test suite:

```bash
# Core unit tests (all passing)
python -m pytest tests/test_config.py tests/test_models.py tests/test_gpu_allocator.py tests/test_worker_queue.py tests/test_resource_state.py tests/test_monitoring.py -v

# All tests (some integration tests may have mock issues)
python -m pytest tests/ -v

# Run examples
PYTHONPATH=. python examples/basic_client_usage.py
PYTHONPATH=. python examples/config_examples.py
```

## Architecture

The GPU Worker Pool consists of several components:

### Client Components
- **Client Interface**: Simple API for requesting/releasing GPUs
- **Worker Pool Manager**: Orchestrates resource allocation
- **GPU Monitor**: Monitors real-time GPU statistics
- **GPU Allocator**: Implements allocation logic based on thresholds
- **Worker Queue**: Manages blocked workers waiting for resources
- **HTTP Client**: Communicates with GPU statistics service
- **Configuration Manager**: Handles environment and programmatic config

### GPU Statistics Server Components
- **FastAPI Application**: Main web application with REST endpoints
- **GPUMonitor Class**: Handles GPU monitoring operations using nvidia-smi
- **Pydantic Models**: Data validation and serialization for API responses
- **Caching System**: Intelligent caching with configurable refresh intervals
- **Error Handling**: Graceful handling of nvidia-smi errors and unsupported features
- **Configuration System**: Environment-based configuration with helper functions
- **CORS Middleware**: Optional Cross-Origin Resource Sharing support

### Server Endpoints
- `/` - API information and available endpoints
- `/gpu/stats` - Detailed GPU statistics with memory, utilization, temperature, power
- `/gpu/count` - Simple GPU count endpoint
- `/gpu/summary` - Aggregated GPU usage summary for client integration
- `/health` - Health check with nvidia-smi availability status
- `/config` - Current server configuration information
- `/docs` - Interactive Swagger UI documentation
- `/redoc` - Alternative ReDoc API documentation

## Requirements

- Python 3.8+
- aiohttp
- asyncio

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [documentation-url]
- Examples: See `examples/` directory