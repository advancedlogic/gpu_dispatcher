# Changelog

All notable changes to the GPU Worker Pool project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-07-25

### Added
- **GPU Filtering Support**: Added CUDA_VISIBLE_DEVICES environment variable support for filtering visible GPUs
  - Support for multiple formats: comma-separated (`"0,1,2"`), space-separated (`"0 1 2"`), mixed (`"0,1 2"`), single GPU (`"1"`), and empty string (`""`)
  - Automatic parsing and validation of GPU indices with fallback to all GPUs if invalid indices are specified
  - GPU filtering applies to all API endpoints (`/gpu/stats`, `/gpu/count`, `/gpu/summary`) while maintaining original GPU IDs
  - Enhanced `/config` endpoint with `gpu_filtering` section showing current filtering status
  - Comprehensive logging for filtering configuration and invalid GPU indices
  - Graceful handling of edge cases including empty strings and invalid formats

### Enhanced
- **Documentation**: Updated comprehensive documentation for CUDA_VISIBLE_DEVICES filtering
  - Added GPU filtering section to SERVER.md with usage examples and troubleshooting
  - Enhanced API.md with detailed configuration endpoint documentation
  - Updated README.md with GPU filtering examples and configuration
  - Added troubleshooting section with 9 specific GPU filtering scenarios and debugging commands
  - Included filtered vs unfiltered response examples for better understanding

### Fixed
- **GPU Statistics Server**: Improved robustness of GPU data collection with filtering support
  - Better error handling for invalid GPU indices in CUDA_VISIBLE_DEVICES
  - Enhanced logging for GPU filtering configuration and status

## [1.0.0] - 2024-07-24

### Added
- Initial release of GPU Worker Pool
- Core GPU resource allocation and management system
- Intelligent threshold-based GPU assignment
- Worker queue management with blocking/unblocking
- Real-time GPU statistics monitoring
- Fault-tolerant HTTP client with retry logic and circuit breaker
- Comprehensive configuration management via environment variables
- Simple client interface with context manager support
- Automatic GPU assignment and release with `GPUContextManager`
- Factory function `gpu_worker_pool_client()` for convenient usage
- Structured logging and metrics collection
- Health checking and performance monitoring
- Comprehensive error handling and recovery
- Thread-safe resource state management
- Stale assignment cleanup
- Command-line interface for status monitoring

### GPU Statistics Server
- **FastAPI-based Server**: Production-ready GPU statistics server with REST API
- **Real-time Monitoring**: Live GPU statistics using nvidia-smi with configurable caching
- **Interactive Documentation**: Built-in Swagger UI (`/docs`) and ReDoc (`/redoc`) documentation
- **Comprehensive Endpoints**: Multiple API endpoints for different use cases:
  - `/` - API information and endpoint listing
  - `/gpu/stats` - Detailed GPU statistics with memory, utilization, temperature, and power
  - `/gpu/count` - Simple GPU count endpoint
  - `/gpu/summary` - Aggregated GPU usage summary
  - `/health` - Health check with nvidia-smi availability status
  - `/config` - Server configuration information
- **Environment Configuration**: Full configuration via environment variables
- **Error Handling**: Graceful handling of missing nvidia-smi, unsupported GPU features, and data conversion errors
- **CORS Support**: Configurable Cross-Origin Resource Sharing for web integration
- **Caching System**: Intelligent caching with configurable refresh intervals
- **Production Features**: Structured logging, multiple startup methods, and uvicorn integration

### Features
- **Client Interface**: Easy-to-use `GPUWorkerPoolClient` with async context manager support
- **Resource Management**: Intelligent GPU allocation based on memory and utilization thresholds
- **Queue Management**: FIFO worker queue with blocking and unblocking capabilities
- **Monitoring**: Real-time GPU statistics polling with exponential backoff
- **Error Handling**: Comprehensive error recovery with retry logic and circuit breaker patterns
- **Configuration**: Flexible configuration via environment variables and programmatic API
- **GPU Server**: Standalone FastAPI server for GPU statistics with multiple deployment options
- **Type Safety**: Full type hints support with `py.typed` marker
- **Documentation**: Comprehensive API documentation and user guide
- **Testing**: Extensive test suite with unit and integration tests
- **Examples**: Complete working examples for different use cases

### Configuration
- `GPU_SERVICE_ENDPOINT`: GPU statistics service URL
- `GPU_MEMORY_THRESHOLD_PERCENT`: Memory usage threshold (0-100%)
- `GPU_UTILIZATION_THRESHOLD_PERCENT`: GPU utilization threshold (0-100%)
- `GPU_POLLING_INTERVAL`: Statistics polling interval in seconds

### Dependencies
- Python 3.8+
- aiohttp >= 3.8.0
- asyncio-throttle >= 1.0.0

### Documentation
- Complete API reference
- User guide with examples
- Configuration guide for different environments
- Best practices and troubleshooting guide

### Examples
- Basic client usage patterns
- Configuration examples and validation
- Machine learning training pipeline integration
- Error handling and recovery patterns
- Performance monitoring and debugging