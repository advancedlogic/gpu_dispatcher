#!/usr/bin/env python3
"""
Configuration examples for the GPU Worker Pool Client.

This module demonstrates various configuration patterns and validates
common configuration mistakes.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, patch
from datetime import datetime

from gpu_worker_pool.client import GPUWorkerPoolClient, gpu_worker_pool_client
from gpu_worker_pool.config import EnvironmentConfigurationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Validates GPU worker pool configurations."""
    
    @staticmethod
    def validate_thresholds(memory_threshold: float, utilization_threshold: float) -> Dict[str, Any]:
        """
        Validate threshold values and return validation results.
        
        Args:
            memory_threshold: Memory usage threshold percentage (0-100)
            utilization_threshold: GPU utilization threshold percentage (0-100)
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Validate memory threshold
        if not isinstance(memory_threshold, (int, float)):
            results["errors"].append(f"Memory threshold must be a number, got {type(memory_threshold)}")
            results["valid"] = False
        elif not (0 <= memory_threshold <= 100):
            results["errors"].append(f"Memory threshold must be between 0-100%, got {memory_threshold}%")
            results["valid"] = False
        elif memory_threshold < 50:
            results["warnings"].append(f"Memory threshold {memory_threshold}% is very low, may cause frequent blocking")
        elif memory_threshold > 95:
            results["warnings"].append(f"Memory threshold {memory_threshold}% is very high, may not prevent OOM")
        
        # Validate utilization threshold
        if not isinstance(utilization_threshold, (int, float)):
            results["errors"].append(f"Utilization threshold must be a number, got {type(utilization_threshold)}")
            results["valid"] = False
        elif not (0 <= utilization_threshold <= 100):
            results["errors"].append(f"Utilization threshold must be between 0-100%, got {utilization_threshold}%")
            results["valid"] = False
        elif utilization_threshold < 50:
            results["warnings"].append(f"Utilization threshold {utilization_threshold}% is very low, may underutilize GPUs")
        elif utilization_threshold > 98:
            results["warnings"].append(f"Utilization threshold {utilization_threshold}% is very high, may cause performance issues")
        
        # Recommendations
        if results["valid"]:
            if memory_threshold > utilization_threshold:
                results["recommendations"].append("Consider setting memory threshold lower than utilization threshold")
            
            if abs(memory_threshold - utilization_threshold) < 10:
                results["recommendations"].append("Consider having more separation between memory and utilization thresholds")
        
        return results
    
    @staticmethod
    def validate_service_endpoint(endpoint: str) -> Dict[str, Any]:
        """
        Validate service endpoint URL.
        
        Args:
            endpoint: Service endpoint URL
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        if not isinstance(endpoint, str):
            results["errors"].append(f"Endpoint must be a string, got {type(endpoint)}")
            results["valid"] = False
            return results
        
        if not endpoint.strip():
            results["errors"].append("Endpoint cannot be empty")
            results["valid"] = False
            return results
        
        if not (endpoint.startswith('http://') or endpoint.startswith('https://')):
            results["errors"].append("Endpoint must start with http:// or https://")
            results["valid"] = False
        
        if endpoint.startswith('http://') and 'localhost' not in endpoint and '127.0.0.1' not in endpoint:
            results["warnings"].append("Using HTTP (not HTTPS) for non-local endpoint")
        
        if endpoint.endswith('/'):
            results["recommendations"].append("Endpoint should not end with '/', it will be stripped automatically")
        
        return results


def create_mock_http_session(gpu_count: int = 2, high_usage: bool = False):
    """Create a mock HTTP session for testing configurations."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    
    if high_usage:
        # High resource usage scenario
        mock_response.json.return_value = {
            "gpu_count": gpu_count,
            "total_memory_mb": 16384,
            "total_used_memory_mb": 14000,
            "average_utilization_percent": 95.0,
            "gpus_summary": [
                {"gpu_id": i, "name": f"GPU-{i}", "memory_usage_percent": 85.0 + (i * 5), "utilization_percent": 90.0 + (i * 3)}
                for i in range(gpu_count)
            ],
            "total_memory_usage_percent": 85.0,
            "timestamp": datetime.now().isoformat()
        }
    else:
        # Normal resource usage scenario
        mock_response.json.return_value = {
            "gpu_count": gpu_count,
            "total_memory_mb": 16384,
            "total_used_memory_mb": 4096,
            "average_utilization_percent": 25.0,
            "gpus_summary": [
                {"gpu_id": i, "name": f"GPU-{i}", "memory_usage_percent": 20.0 + (i * 10), "utilization_percent": 15.0 + (i * 20)}
                for i in range(gpu_count)
            ],
            "total_memory_usage_percent": 25.0,
            "timestamp": datetime.now().isoformat()
        }
    
    mock_response.status = 200
    mock_session.get.return_value.__aenter__.return_value = mock_response
    return mock_session


async def example_conservative_configuration():
    """Example with conservative thresholds for production environments."""
    print("\n=== Conservative Configuration Example ===")
    print("Suitable for: Production environments, critical workloads")
    
    config = {
        "memory_threshold": 70.0,      # Conservative memory limit
        "utilization_threshold": 80.0,  # Conservative utilization limit
        "worker_timeout": 600.0,        # 10 minutes timeout
        "polling_interval": 3,          # Frequent polling for responsiveness
        "service_endpoint": "https://gpu-service.production.com"
    }
    
    print(f"Configuration: {config}")
    
    # Validate configuration
    validator = ConfigurationValidator()
    threshold_validation = validator.validate_thresholds(
        config["memory_threshold"], 
        config["utilization_threshold"]
    )
    endpoint_validation = validator.validate_service_endpoint(config["service_endpoint"])
    
    print(f"Threshold validation: {'✓ Valid' if threshold_validation['valid'] else '✗ Invalid'}")
    if threshold_validation["warnings"]:
        print(f"  Warnings: {threshold_validation['warnings']}")
    if threshold_validation["recommendations"]:
        print(f"  Recommendations: {threshold_validation['recommendations']}")
    
    print(f"Endpoint validation: {'✓ Valid' if endpoint_validation['valid'] else '✗ Invalid'}")
    if endpoint_validation["warnings"]:
        print(f"  Warnings: {endpoint_validation['warnings']}")
    
    # Test with mock
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        mock_session_class.return_value = create_mock_http_session()
        
        async with gpu_worker_pool_client(**config) as client:
            await asyncio.sleep(0.1)  # Wait for stats
            status = client.get_pool_status()
            print(f"Pool status: {status.total_gpus} GPUs, {status.available_gpus} available")


async def example_aggressive_configuration():
    """Example with aggressive thresholds for maximum utilization."""
    print("\n=== Aggressive Configuration Example ===")
    print("Suitable for: Development environments, maximum GPU utilization")
    
    config = {
        "memory_threshold": 90.0,       # High memory limit
        "utilization_threshold": 95.0,  # High utilization limit
        "worker_timeout": 120.0,        # 2 minutes timeout
        "polling_interval": 10,         # Less frequent polling
        "service_endpoint": "http://localhost:8080"
    }
    
    print(f"Configuration: {config}")
    
    # Validate configuration
    validator = ConfigurationValidator()
    threshold_validation = validator.validate_thresholds(
        config["memory_threshold"], 
        config["utilization_threshold"]
    )
    
    print(f"Threshold validation: {'✓ Valid' if threshold_validation['valid'] else '✗ Invalid'}")
    if threshold_validation["warnings"]:
        print(f"  Warnings: {threshold_validation['warnings']}")
    
    # Test with high usage scenario
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        mock_session_class.return_value = create_mock_http_session(high_usage=True)
        
        async with gpu_worker_pool_client(**config) as client:
            await asyncio.sleep(0.1)  # Wait for stats
            status = client.get_pool_status()
            print(f"Pool status: {status.total_gpus} GPUs, {status.available_gpus} available")
            print("Note: With high GPU usage, fewer GPUs may be available")


async def example_development_configuration():
    """Example configuration for development environments."""
    print("\n=== Development Configuration Example ===")
    print("Suitable for: Local development, testing")
    
    config = {
        "memory_threshold": 75.0,
        "utilization_threshold": 85.0,
        "worker_timeout": 60.0,         # Short timeout for quick feedback
        "polling_interval": 2,          # Fast polling for development
        "service_endpoint": "http://localhost:8080"
    }
    
    print(f"Configuration: {config}")
    
    # Test configuration validation
    validator = ConfigurationValidator()
    threshold_validation = validator.validate_thresholds(
        config["memory_threshold"], 
        config["utilization_threshold"]
    )
    
    print(f"Validation: {'✓ Valid' if threshold_validation['valid'] else '✗ Invalid'}")
    
    # Test with mock
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        mock_session_class.return_value = create_mock_http_session(gpu_count=1)
        
        async with gpu_worker_pool_client(**config) as client:
            await asyncio.sleep(0.1)  # Wait for stats
            
            # Get detailed metrics for development debugging
            metrics = client.get_detailed_metrics()
            print(f"Detailed metrics available: {len(metrics['gpu_metrics'])} GPUs tracked")
            print(f"Thresholds: Memory {metrics['thresholds']['memory_threshold_percent']}%, "
                  f"Utilization {metrics['thresholds']['utilization_threshold_percent']}%")


async def example_environment_variable_configuration():
    """Example using environment variables for configuration."""
    print("\n=== Environment Variable Configuration Example ===")
    print("Suitable for: Containerized deployments, CI/CD pipelines")
    
    # Set environment variables
    original_env = {}
    test_env = {
        "GPU_MEMORY_THRESHOLD_PERCENT": "85.0",
        "GPU_UTILIZATION_THRESHOLD_PERCENT": "92.0",
        "GPU_SERVICE_ENDPOINT": "http://gpu-service:8080",
        "GPU_POLLING_INTERVAL": "7"
    }
    
    # Backup and set environment variables
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        print(f"Environment variables set: {test_env}")
        
        # Create configuration manager to read from environment
        config_manager = EnvironmentConfigurationManager()
        
        print(f"Loaded configuration:")
        print(f"  Memory threshold: {config_manager.get_memory_threshold()}%")
        print(f"  Utilization threshold: {config_manager.get_utilization_threshold()}%")
        print(f"  Service endpoint: {config_manager.get_service_endpoint()}")
        print(f"  Polling interval: {config_manager.get_polling_interval()}s")
        
        # Test with client (no parameters needed, will use environment)
        with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value = create_mock_http_session()
            
            async with GPUWorkerPoolClient() as client:
                await asyncio.sleep(0.1)  # Wait for stats
                status = client.get_pool_status()
                print(f"Pool status: {status.total_gpus} GPUs, {status.available_gpus} available")
    
    finally:
        # Restore original environment variables
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


async def example_invalid_configurations():
    """Examples of invalid configurations and how they're handled."""
    print("\n=== Invalid Configuration Examples ===")
    print("Demonstrating validation and error handling")
    
    invalid_configs = [
        {
            "name": "Invalid memory threshold (too high)",
            "config": {"memory_threshold": 150.0, "utilization_threshold": 80.0}
        },
        {
            "name": "Invalid utilization threshold (negative)",
            "config": {"memory_threshold": 80.0, "utilization_threshold": -10.0}
        },
        {
            "name": "Invalid endpoint (no protocol)",
            "config": {"service_endpoint": "gpu-service.com"}
        },
        {
            "name": "Invalid types",
            "config": {"memory_threshold": "high", "utilization_threshold": "medium"}
        }
    ]
    
    validator = ConfigurationValidator()
    
    for example in invalid_configs:
        print(f"\n--- {example['name']} ---")
        config = example["config"]
        print(f"Configuration: {config}")
        
        # Validate thresholds if present
        if "memory_threshold" in config and "utilization_threshold" in config:
            validation = validator.validate_thresholds(
                config["memory_threshold"],
                config["utilization_threshold"]
            )
            print(f"Validation: {'✓ Valid' if validation['valid'] else '✗ Invalid'}")
            if validation["errors"]:
                print(f"  Errors: {validation['errors']}")
            if validation["warnings"]:
                print(f"  Warnings: {validation['warnings']}")
        
        # Validate endpoint if present
        if "service_endpoint" in config:
            endpoint_validation = validator.validate_service_endpoint(config["service_endpoint"])
            print(f"Endpoint validation: {'✓ Valid' if endpoint_validation['valid'] else '✗ Invalid'}")
            if endpoint_validation["errors"]:
                print(f"  Errors: {endpoint_validation['errors']}")


async def example_configuration_best_practices():
    """Examples demonstrating configuration best practices."""
    print("\n=== Configuration Best Practices ===")
    
    best_practices = [
        {
            "scenario": "High-throughput batch processing",
            "config": {
                "memory_threshold": 85.0,
                "utilization_threshold": 90.0,
                "worker_timeout": 1800.0,  # 30 minutes for long jobs
                "polling_interval": 5
            },
            "rationale": "Higher thresholds for maximum utilization, long timeout for batch jobs"
        },
        {
            "scenario": "Interactive ML development",
            "config": {
                "memory_threshold": 70.0,
                "utilization_threshold": 80.0,
                "worker_timeout": 300.0,   # 5 minutes for interactive work
                "polling_interval": 2
            },
            "rationale": "Conservative thresholds for stability, short timeout for responsiveness"
        },
        {
            "scenario": "Multi-tenant shared environment",
            "config": {
                "memory_threshold": 60.0,
                "utilization_threshold": 75.0,
                "worker_timeout": 600.0,   # 10 minutes
                "polling_interval": 3
            },
            "rationale": "Very conservative thresholds to ensure fair sharing among tenants"
        },
        {
            "scenario": "CI/CD automated testing",
            "config": {
                "memory_threshold": 80.0,
                "utilization_threshold": 85.0,
                "worker_timeout": 120.0,   # 2 minutes for quick feedback
                "polling_interval": 1
            },
            "rationale": "Balanced thresholds with short timeout and fast polling for CI speed"
        }
    ]
    
    validator = ConfigurationValidator()
    
    for practice in best_practices:
        print(f"\n--- {practice['scenario']} ---")
        config = practice["config"]
        print(f"Configuration: {config}")
        print(f"Rationale: {practice['rationale']}")
        
        # Validate configuration
        validation = validator.validate_thresholds(
            config["memory_threshold"],
            config["utilization_threshold"]
        )
        
        print(f"Validation: {'✓ Valid' if validation['valid'] else '✗ Invalid'}")
        if validation["recommendations"]:
            print(f"  Recommendations: {validation['recommendations']}")


async def example_end_to_end_realistic_scenario():
    """End-to-end test with realistic configuration and usage patterns."""
    print("\n=== End-to-End Realistic Scenario ===")
    print("Simulating a machine learning training pipeline")
    
    # Production-like configuration
    config = {
        "memory_threshold": 75.0,
        "utilization_threshold": 85.0,
        "worker_timeout": 900.0,  # 15 minutes for training jobs
        "polling_interval": 5,
        "service_endpoint": "http://gpu-cluster.ml.company.com"
    }
    
    print(f"ML Pipeline Configuration: {config}")
    
    # Validate configuration
    validator = ConfigurationValidator()
    validation = validator.validate_thresholds(
        config["memory_threshold"],
        config["utilization_threshold"]
    )
    
    if not validation["valid"]:
        print(f"❌ Configuration validation failed: {validation['errors']}")
        return
    
    print("✅ Configuration validated successfully")
    
    # Simulate ML training pipeline
    with patch('gpu_worker_pool.http_client.aiohttp.ClientSession') as mock_session_class:
        mock_session_class.return_value = create_mock_http_session(gpu_count=4)
        
        async def training_job(client, job_id: str, duration: float):
            """Simulate a training job."""
            print(f"  Job {job_id}: Starting training...")
            
            try:
                # Request GPU with timeout
                assignment = await client.request_gpu(timeout=60.0)
                print(f"  Job {job_id}: Assigned to GPU {assignment.gpu_id}")
                
                # Simulate training work
                await asyncio.sleep(duration)
                
                # Release GPU
                await client.release_gpu(assignment)
                print(f"  Job {job_id}: Training completed, GPU released")
                
            except Exception as e:
                print(f"  Job {job_id}: Failed - {e}")
        
        async with gpu_worker_pool_client(**config) as client:
            print("Starting ML training pipeline...")
            
            # Wait for initial GPU stats
            await asyncio.sleep(0.2)
            
            # Check initial pool status
            status = client.get_pool_status()
            print(f"Initial pool status: {status.total_gpus} GPUs, {status.available_gpus} available")
            
            # Start multiple training jobs
            training_jobs = [
                training_job(client, f"model-{i}", 0.5 + (i * 0.2))
                for i in range(3)  # 3 training jobs
            ]
            
            # Run jobs concurrently
            await asyncio.gather(*training_jobs)
            
            # Final status
            final_status = client.get_pool_status()
            print(f"Final pool status: {final_status.active_workers} active workers, "
                  f"{final_status.available_gpus} available GPUs")
            
            print("ML training pipeline completed successfully!")


async def main():
    """Run all configuration examples."""
    print("GPU Worker Pool Configuration Examples")
    print("=" * 60)
    
    try:
        await example_conservative_configuration()
        await example_aggressive_configuration()
        await example_development_configuration()
        await example_environment_variable_configuration()
        await example_invalid_configurations()
        await example_configuration_best_practices()
        await example_end_to_end_realistic_scenario()
        
        print("\n" + "=" * 60)
        print("All configuration examples completed successfully!")
        
    except Exception as e:
        print(f"Error running configuration examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())