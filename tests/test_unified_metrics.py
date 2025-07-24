"""Tests for the unified pool status and metrics system."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from gpu_worker_pool.unified_metrics import UnifiedPoolMetrics, MetricsFormatter
from gpu_worker_pool.models import (
    PoolStatus, MultiEndpointPoolStatus, EndpointInfo, WorkerInfo
)


class TestUnifiedPoolMetrics:
    """Test cases for UnifiedPoolMetrics class."""
    
    def test_single_endpoint_mode_initialization(self):
        """Test initialization in single-endpoint mode."""
        metrics = UnifiedPoolMetrics(is_multi_endpoint_mode=False)
        assert not metrics.is_multi_endpoint_mode
    
    def test_multi_endpoint_mode_initialization(self):
        """Test initialization in multi-endpoint mode."""
        metrics = UnifiedPoolMetrics(is_multi_endpoint_mode=True)
        assert metrics.is_multi_endpoint_mode
    
    def test_single_endpoint_pool_status(self):
        """Test unified pool status creation in single-endpoint mode."""
        metrics = UnifiedPoolMetrics(is_multi_endpoint_mode=False)
        
        # Create mock pool status
        pool_status = PoolStatus(
            total_gpus=4,
            available_gpus=2,
            active_workers=2,
            blocked_workers=1,
            gpu_assignments={0: [WorkerInfo(id="worker1", enqueued_at=datetime.now(), callback=lambda x: None, on_error=lambda e: None)]}
        )
        
        # Should return the same pool status
        result = metrics.create_unified_pool_status(pool_status)
        assert result == pool_status
        assert isinstance(result, PoolStatus)
    
    def test_multi_endpoint_pool_status_with_healthy_endpoints(self):
        """Test unified pool status creation in multi-endpoint mode with healthy endpoints."""
        metrics = UnifiedPoolMetrics(is_multi_endpoint_mode=True)
        
        # Create mock pool status
        pool_status = PoolStatus(
            total_gpus=6,  # This will be overridden by endpoint aggregation
            available_gpus=3,
            active_workers=3,
            blocked_workers=1,
            gpu_assignments={}  # Simplified for multi-endpoint test
        )
        
        # Create mock endpoints
        endpoints = [
            EndpointInfo(
                endpoint_id="server1", 
                url="http://server1:8000", 
                is_healthy=True,
                total_gpus=4, 
                available_gpus=2, 
                last_seen=datetime.now(),
                response_time_ms=50.0
            ),
            EndpointInfo(
                endpoint_id="server2", 
                url="http://server2:8000", 
                is_healthy=True,
                total_gpus=2, 
                available_gpus=1, 
                last_seen=datetime.now(),
                response_time_ms=75.0
            ),
            EndpointInfo(
                endpoint_id="server3", 
                url="http://server3:8000", 
                is_healthy=False,
                total_gpus=4, 
                available_gpus=0, 
                last_seen=datetime.now() - timedelta(minutes=5),
                response_time_ms=0.0
            )
        ]
        
        result = metrics.create_unified_pool_status(pool_status, endpoints)
        
        assert isinstance(result, MultiEndpointPoolStatus)
        assert result.total_endpoints == 3
        assert result.healthy_endpoints == 2
        assert result.total_gpus == 6  # Only healthy endpoints: 4 + 2
        assert result.available_gpus == 3  # Only healthy endpoints: 2 + 1
        assert result.active_workers == 3
        assert result.blocked_workers == 1
        assert len(result.endpoints) == 3
    
    def test_multi_endpoint_pool_status_no_healthy_endpoints(self):
        """Test unified pool status creation with no healthy endpoints."""
        metrics = UnifiedPoolMetrics(is_multi_endpoint_mode=True)
        
        pool_status = PoolStatus(
            total_gpus=4,
            available_gpus=0,
            active_workers=0,
            blocked_workers=5,
            gpu_assignments={}
        )
        
        # All endpoints are unhealthy
        endpoints = [
            EndpointInfo(
                endpoint_id="server1", 
                url="http://server1:8000", 
                is_healthy=False,
                total_gpus=4, 
                available_gpus=0, 
                last_seen=datetime.now() - timedelta(minutes=10),
                response_time_ms=0.0
            )
        ]
        
        result = metrics.create_unified_pool_status(pool_status, endpoints)
        
        assert isinstance(result, MultiEndpointPoolStatus)
        assert result.total_endpoints == 1
        assert result.healthy_endpoints == 0
        assert result.total_gpus == 0  # No healthy endpoints
        assert result.available_gpus == 0
    
    def test_single_endpoint_detailed_metrics(self):
        """Test detailed metrics creation in single-endpoint mode."""
        metrics = UnifiedPoolMetrics(is_multi_endpoint_mode=False)
        
        base_metrics = {
            'active_workers': 2,
            'blocked_workers': 1,
            'service_endpoint': 'http://localhost:8000',
            'monitor_healthy': True
        }
        
        result = metrics.create_unified_detailed_metrics(base_metrics)
        
        assert result['mode'] == 'single-endpoint'
        assert 'timestamp' in result
        assert 'single_endpoint' in result
        assert result['single_endpoint']['endpoint_url'] == 'http://localhost:8000'
        assert result['single_endpoint']['is_healthy'] == True
    
    def test_multi_endpoint_detailed_metrics(self):
        """Test detailed metrics creation in multi-endpoint mode."""
        metrics = UnifiedPoolMetrics(is_multi_endpoint_mode=True)
        
        base_metrics = {
            'active_workers': 3,
            'blocked_workers': 2,
            'gpu_assignments': {'server1:0': [], 'server2:1': []}
        }
        
        # Create mock endpoints
        endpoints = [
            EndpointInfo(
                endpoint_id="server1", 
                url="http://server1:8000", 
                is_healthy=True,
                total_gpus=4, 
                available_gpus=2, 
                last_seen=datetime.now(),
                response_time_ms=50.0
            ),
            EndpointInfo(
                endpoint_id="server2", 
                url="http://server2:8000", 
                is_healthy=True,
                total_gpus=2, 
                available_gpus=0, 
                last_seen=datetime.now(),
                response_time_ms=100.0
            )
        ]
        
        # Mock endpoint health
        endpoint_health = {
            'server1': {
                'connectivity_status': 'healthy',
                'health_score': 95.0,
                'consecutive_failures': 0,
                'last_successful_communication': datetime.now().isoformat(),
                'time_since_last_success': '0 seconds'
            },
            'server2': {
                'connectivity_status': 'healthy',
                'health_score': 85.0,
                'consecutive_failures': 0,
                'last_successful_communication': datetime.now().isoformat(),
                'time_since_last_success': '0 seconds'
            }
        }
        
        # Mock load balancer info
        load_balancer_info = {
            'strategy': 'availability_based',
            'strategy_description': 'Distributes load based on available GPU percentage'
        }
        
        result = metrics.create_unified_detailed_metrics(
            base_metrics=base_metrics,
            endpoints=endpoints,
            endpoint_health=endpoint_health,
            load_balancer_info=load_balancer_info
        )
        
        assert result['mode'] == 'multi-endpoint'
        assert 'timestamp' in result
        assert 'aggregated_stats' in result
        assert 'endpoints' in result
        assert 'load_balancer' in result
        assert 'endpoint_health_summary' in result
        assert 'load_balancing_effectiveness' in result
        
        # Check endpoint summary
        endpoint_summary = result['endpoints']['summary']
        assert endpoint_summary['total_endpoints'] == 2
        assert endpoint_summary['healthy_endpoints'] == 2
        assert endpoint_summary['total_gpus_healthy_endpoints'] == 6
        assert endpoint_summary['available_gpus_healthy_endpoints'] == 2
        
        # Check aggregated stats
        agg_stats = result['aggregated_stats']
        assert agg_stats['total_gpus'] == 6
        assert agg_stats['available_gpus'] == 2
        assert agg_stats['utilized_gpus'] == 4
        assert agg_stats['endpoints_contributing'] == 2
    
    def test_load_balancing_effectiveness_calculation(self):
        """Test load balancing effectiveness calculation."""
        metrics = UnifiedPoolMetrics(is_multi_endpoint_mode=True)
        
        # Create endpoints with different utilization rates
        endpoints = [
            EndpointInfo(
                endpoint_id="server1", 
                url="http://server1:8000", 
                is_healthy=True,
                total_gpus=4, 
                available_gpus=2,  # 50% utilization
                last_seen=datetime.now(),
                response_time_ms=50.0
            ),
            EndpointInfo(
                endpoint_id="server2", 
                url="http://server2:8000", 
                is_healthy=True,
                total_gpus=4, 
                available_gpus=2,  # 50% utilization (perfect balance)
                last_seen=datetime.now(),
                response_time_ms=60.0
            )
        ]
        
        effectiveness = metrics._calculate_load_balancing_effectiveness(endpoints, {})
        
        assert 'distribution_score' in effectiveness
        assert 'utilization_variance' in effectiveness
        assert 'balancing_quality' in effectiveness
        assert effectiveness['utilization_variance'] == 0.0  # Perfect balance
        assert effectiveness['balancing_quality'] == 'excellent'
    
    def test_endpoint_health_summary(self):
        """Test endpoint health summary creation."""
        metrics = UnifiedPoolMetrics(is_multi_endpoint_mode=True)
        
        endpoint_health = {
            'server1': {
                'connectivity_status': 'healthy',
                'health_score': 95.0
            },
            'server2': {
                'connectivity_status': 'degraded', 
                'health_score': 75.0
            },
            'server3': {
                'connectivity_status': 'offline',
                'health_score': 0.0
            }
        }
        
        health_summary = metrics._create_health_summary(endpoint_health)
        
        assert health_summary['total_endpoints'] == 3
        assert health_summary['status_distribution']['healthy'] == 1
        assert health_summary['status_distribution']['degraded'] == 1
        assert health_summary['status_distribution']['offline'] == 1
        assert health_summary['health_scores']['average'] == 56.67
        assert health_summary['healthy_percentage'] == 33.33


class TestMetricsFormatter:
    """Test cases for MetricsFormatter class."""
    
    def test_console_format_single_endpoint(self):
        """Test console formatting for single-endpoint metrics."""
        metrics = {
            'mode': 'single-endpoint',
            'timestamp': '2023-01-01T12:00:00',
            'active_workers': 2,
            'blocked_workers': 1,
            'single_endpoint': {
                'endpoint_url': 'http://localhost:8000',
                'is_healthy': True
            }
        }
        
        result = MetricsFormatter.format_for_console(metrics)
        
        assert '=== GPU Worker Pool Metrics ===' in result
        assert 'Mode: single-endpoint' in result
        assert 'Endpoint URL: http://localhost:8000' in result
        assert 'Active Workers: 2' in result
        assert 'Blocked Workers: 1' in result
    
    def test_console_format_multi_endpoint(self):
        """Test console formatting for multi-endpoint metrics."""
        metrics = {
            'mode': 'multi-endpoint',
            'timestamp': '2023-01-01T12:00:00',
            'active_workers': 3,
            'blocked_workers': 2,
            'endpoints': {
                'summary': {
                    'total_endpoints': 3,
                    'healthy_endpoints': 2,
                    'total_gpus_healthy_endpoints': 6,
                    'available_gpus_healthy_endpoints': 3,
                    'overall_availability_rate': 50.0
                }
            },
            'load_balancing_effectiveness': {
                'distribution_score': 85.0,
                'balancing_quality': 'good'
            }
        }
        
        result = MetricsFormatter.format_for_console(metrics)
        
        assert '=== GPU Worker Pool Metrics ===' in result
        assert 'Mode: multi-endpoint' in result
        assert 'Total Endpoints: 3' in result
        assert 'Healthy Endpoints: 2' in result
        assert 'Distribution Score: 85.0/100' in result
        assert 'Balancing Quality: good' in result
    
    def test_health_summary_format(self):
        """Test health summary formatting."""
        endpoint_health = {
            'server1': {
                'connectivity_status': 'healthy',
                'health_score': 95.0,
                'response_time_ms': 50.0,
                'time_since_last_success': '10 seconds'
            },
            'server2': {
                'connectivity_status': 'degraded',
                'health_score': 70.0,
                'response_time_ms': 150.0,
                'time_since_last_success': '2 minutes'
            }
        }
        
        result = MetricsFormatter.format_health_summary(endpoint_health)
        
        assert '=== Endpoint Health Status ===' in result
        assert 'Endpoint: server1' in result
        assert 'Status: healthy' in result
        assert 'Health Score: 95.0/100' in result
        assert 'Endpoint: server2' in result
        assert 'Status: degraded' in result
    
    def test_json_format(self):
        """Test JSON formatting (should return metrics as-is)."""
        metrics = {'test': 'data', 'mode': 'single-endpoint'}
        result = MetricsFormatter.format_for_json(metrics)
        assert result == metrics