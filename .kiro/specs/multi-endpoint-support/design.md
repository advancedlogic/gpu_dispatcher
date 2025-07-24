# Multi-Endpoint Support Design

## Overview

This design extends the GPU Worker Pool client to support multiple GPU statistics server endpoints, enabling GPU resource management across multiple machines while maintaining backward compatibility with single-endpoint configurations.

## Architecture

### Core Components

#### 1. Multi-Endpoint Configuration Manager
- Extends existing `ConfigurationManager` to handle multiple endpoints
- Supports both single endpoint (backward compatibility) and multiple endpoints
- Parses comma-separated endpoint URLs from environment variables
- Validates endpoint URLs and removes duplicates

#### 2. Endpoint Manager
- Manages the lifecycle of multiple endpoint connections
- Tracks endpoint health and availability status
- Implements connection pooling and cleanup
- Provides endpoint discovery and health monitoring

#### 3. Global GPU ID System
- Creates unique identifiers for GPUs across all endpoints
- Format: `{endpoint_id}:{local_gpu_id}` (e.g., "server1:0", "server2:1")
- Maintains mapping between global IDs and endpoint/local ID pairs
- Handles GPU assignment routing to correct endpoints

#### 4. Multi-Endpoint HTTP Client Pool
- Manages multiple `AsyncGPUStatsHTTPClient` instances
- Implements per-endpoint circuit breakers and retry logic
- Aggregates responses from multiple endpoints
- Handles partial failures gracefully

#### 5. Intelligent Load Balancer
- Implements multiple load balancing strategies:
  - **Availability-based**: Prefer endpoints with more available GPUs
  - **Round-robin**: Distribute evenly when availability is equal
  - **Weighted**: Consider total GPU capacity of each endpoint
- Adapts to endpoint failures and recoveries
- Provides load balancing metrics and statistics

#### 6. Multi-Endpoint GPU Monitor
- Extends `AsyncGPUMonitor` to poll multiple endpoints
- Aggregates GPU statistics from all available endpoints
- Handles partial endpoint failures gracefully
- Maintains per-endpoint health status

### Data Models

#### EndpointInfo
```python
@dataclass
class EndpointInfo:
    endpoint_id: str          # Unique identifier for the endpoint
    url: str                  # Full URL of the endpoint
    is_healthy: bool          # Current health status
    last_seen: datetime       # Last successful communication
    total_gpus: int          # Total GPUs available at this endpoint
    available_gpus: int      # Currently available GPUs
    response_time_ms: float  # Average response time
```

#### GlobalGPUInfo
```python
@dataclass
class GlobalGPUInfo:
    global_gpu_id: str           # Global unique identifier (endpoint_id:local_id)
    endpoint_id: str             # Source endpoint identifier
    local_gpu_id: int           # GPU ID on the source endpoint
    name: str                   # GPU name/model
    memory_usage_percent: float # Memory usage percentage
    utilization_percent: float  # GPU utilization percentage
    is_available: bool          # Whether GPU is available for assignment
```

#### MultiEndpointPoolStatus
```python
@dataclass
class MultiEndpointPoolStatus:
    total_endpoints: int                           # Total configured endpoints
    healthy_endpoints: int                         # Currently healthy endpoints
    total_gpus: int                               # Total GPUs across all endpoints
    available_gpus: int                           # Available GPUs across all endpoints
    active_workers: int                           # Currently active workers
    blocked_workers: int                          # Workers waiting in queue
    endpoints: List[EndpointInfo]                 # Per-endpoint information
    gpu_assignments: Dict[str, List[WorkerInfo]]  # Assignments by global GPU ID
```

### Configuration

#### Environment Variables
```bash
# Multiple endpoints (comma-separated)
export GPU_SERVICE_ENDPOINTS="http://gpu-server1:8000,http://gpu-server2:8000,http://gpu-server3:8000"

# Single endpoint (backward compatibility)
export GPU_SERVICE_ENDPOINT="http://gpu-server:8000"

# Load balancing strategy
export GPU_LOAD_BALANCING_STRATEGY="availability"  # availability, round_robin, weighted

# Endpoint health check interval
export GPU_ENDPOINT_HEALTH_CHECK_INTERVAL="30"

# Endpoint timeout settings
export GPU_ENDPOINT_TIMEOUT="10.0"
export GPU_ENDPOINT_MAX_RETRIES="3"
```

#### Programmatic Configuration
```python
client = GPUWorkerPoolClient(
    service_endpoints=[
        "http://gpu-server1:8000",
        "http://gpu-server2:8000", 
        "http://gpu-server3:8000"
    ],
    load_balancing_strategy="availability",
    endpoint_timeout=10.0,
    endpoint_health_check_interval=30
)
```

### Load Balancing Strategies

#### 1. Availability-Based (Default)
- Prioritizes endpoints with higher percentage of available GPUs
- Considers both absolute count and percentage availability
- Automatically adapts to changing resource availability

#### 2. Round-Robin
- Distributes requests evenly across all healthy endpoints
- Simple and predictable distribution pattern
- Good for homogeneous endpoint configurations

#### 3. Weighted
- Considers total GPU capacity of each endpoint
- Endpoints with more total GPUs receive proportionally more requests
- Optimal for heterogeneous endpoint configurations

### Error Handling and Fault Tolerance

#### Endpoint Failure Scenarios
1. **Connection Failures**: Network connectivity issues
2. **Service Failures**: GPU server returns errors
3. **Timeout Failures**: Slow response times
4. **Partial Failures**: Some endpoints fail while others succeed

#### Recovery Mechanisms
1. **Circuit Breaker**: Per-endpoint circuit breakers prevent cascading failures
2. **Exponential Backoff**: Failed endpoints are retried with increasing delays
3. **Health Checks**: Periodic health checks detect endpoint recovery
4. **Graceful Degradation**: System continues operating with reduced capacity

### Implementation Plan

#### Phase 1: Core Infrastructure
1. Create `MultiEndpointConfigurationManager`
2. Implement `EndpointManager` for connection lifecycle
3. Develop global GPU ID system
4. Create `MultiEndpointHTTPClientPool`

#### Phase 2: Load Balancing and Monitoring
1. Implement load balancing strategies
2. Create `MultiEndpointGPUMonitor`
3. Develop endpoint health monitoring
4. Add aggregated statistics collection

#### Phase 3: Integration and Testing
1. Update `GPUWorkerPoolClient` to use multi-endpoint components
2. Ensure backward compatibility with single-endpoint configuration
3. Add comprehensive error handling and recovery
4. Create extensive test suite for multi-endpoint scenarios

### Backward Compatibility

The design maintains full backward compatibility:

1. **Environment Variables**: Existing `GPU_SERVICE_ENDPOINT` continues to work
2. **Constructor Parameters**: Existing `service_endpoint` parameter remains functional
3. **API Methods**: All existing client methods work without modification
4. **Behavior**: Single-endpoint behavior is identical to current implementation

### Performance Considerations

1. **Connection Pooling**: Reuse HTTP connections across requests
2. **Concurrent Polling**: Poll multiple endpoints concurrently
3. **Caching**: Cache endpoint health status to reduce overhead
4. **Batching**: Batch multiple requests to the same endpoint when possible

### Security Considerations

1. **Endpoint Validation**: Validate and sanitize endpoint URLs
2. **Authentication**: Support for endpoint-specific authentication (future enhancement)
3. **TLS Support**: Ensure HTTPS endpoints are properly validated
4. **Rate Limiting**: Respect rate limits on individual endpoints

### Monitoring and Observability

1. **Per-Endpoint Metrics**: Track success rates, response times, and availability
2. **Load Balancing Metrics**: Monitor distribution effectiveness
3. **Health Status**: Provide detailed endpoint health information
4. **Performance Metrics**: Track overall system performance with multiple endpoints

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock endpoint responses for predictable testing
- Verify error handling and edge cases

### Integration Tests
- Test multi-endpoint scenarios with real servers
- Verify failover and recovery behavior
- Test load balancing effectiveness

### Performance Tests
- Measure overhead of multi-endpoint support
- Test scalability with increasing number of endpoints
- Verify performance under various failure scenarios

### Compatibility Tests
- Ensure backward compatibility with existing configurations
- Test migration scenarios from single to multi-endpoint
- Verify API compatibility across versions