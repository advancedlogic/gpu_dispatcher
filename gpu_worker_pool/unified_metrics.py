"""Unified pool status and metrics aggregation for both single and multi-endpoint modes."""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import asdict

from .models import (
    PoolStatus, MultiEndpointPoolStatus, EndpointInfo, 
    GPUStats, GlobalGPUInfo, WorkerInfo
)

logger = logging.getLogger(__name__)


class UnifiedPoolMetrics:
    """
    Unified pool metrics aggregator that works with both single and multi-endpoint modes.
    
    This class provides a consistent interface for retrieving pool status and detailed
    metrics regardless of whether the system is operating in single-endpoint or
    multi-endpoint mode.
    """
    
    def __init__(self, is_multi_endpoint_mode: bool = False):
        """
        Initialize the unified metrics aggregator.
        
        Args:
            is_multi_endpoint_mode: Whether the system is operating in multi-endpoint mode
        """
        self.is_multi_endpoint_mode = is_multi_endpoint_mode
        
    def create_unified_pool_status(self,
                                 pool_status: PoolStatus,
                                 endpoints: Optional[List[EndpointInfo]] = None,
                                 endpoint_health: Optional[Dict[str, Dict[str, Any]]] = None) -> Union[PoolStatus, MultiEndpointPoolStatus]:
        """
        Create a unified pool status object.
        
        Args:
            pool_status: Base pool status from worker pool manager
            endpoints: List of endpoint information (multi-endpoint mode only)
            endpoint_health: Endpoint health status information (multi-endpoint mode only)
            
        Returns:
            PoolStatus (single-endpoint) or MultiEndpointPoolStatus (multi-endpoint)
        """
        if not self.is_multi_endpoint_mode:
            # Single-endpoint mode: return the base pool status as-is
            return pool_status
        
        # Multi-endpoint mode: create enhanced status
        if not endpoints:
            endpoints = []
        
        # Filter to only healthy endpoints for current statistics
        healthy_endpoints = [ep for ep in endpoints if ep.is_healthy]
        
        # Calculate aggregated statistics from healthy endpoints only
        total_gpus = sum(ep.total_gpus for ep in healthy_endpoints)
        available_gpus = sum(ep.available_gpus for ep in healthy_endpoints)
        
        # Convert GPU assignments from local to global format if needed
        global_gpu_assignments = self._convert_gpu_assignments_to_global(
            pool_status.gpu_assignments, endpoints
        )
        
        return MultiEndpointPoolStatus(
            total_endpoints=len(endpoints),
            healthy_endpoints=len(healthy_endpoints),
            total_gpus=total_gpus,
            available_gpus=available_gpus,
            active_workers=pool_status.active_workers,
            blocked_workers=pool_status.blocked_workers,
            endpoints=endpoints,
            gpu_assignments=global_gpu_assignments
        )
    
    def create_unified_detailed_metrics(self,
                                      base_metrics: Dict[str, Any],
                                      endpoints: Optional[List[EndpointInfo]] = None,
                                      endpoint_health: Optional[Dict[str, Dict[str, Any]]] = None,
                                      load_balancer_info: Optional[Dict[str, Any]] = None,
                                      monitor_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create unified detailed metrics with per-endpoint breakdown.
        
        Args:
            base_metrics: Base metrics from worker pool manager
            endpoints: List of endpoint information (multi-endpoint mode only)
            endpoint_health: Detailed endpoint health status
            load_balancer_info: Load balancer metrics and configuration
            monitor_status: GPU monitor status information
            
        Returns:
            Dictionary with comprehensive unified metrics
        """
        # Start with base metrics
        unified_metrics = base_metrics.copy()
        unified_metrics['mode'] = 'multi-endpoint' if self.is_multi_endpoint_mode else 'single-endpoint'
        unified_metrics['timestamp'] = datetime.now().isoformat()
        
        if not self.is_multi_endpoint_mode:
            # Single-endpoint mode: add basic mode information
            unified_metrics['single_endpoint'] = {
                'endpoint_url': base_metrics.get('service_endpoint', 'unknown'),
                'is_healthy': base_metrics.get('monitor_healthy', True)
            }
            return unified_metrics
        
        # Multi-endpoint mode: add comprehensive endpoint breakdown
        if not endpoints:
            endpoints = []
        
        healthy_endpoints = [ep for ep in endpoints if ep.is_healthy]
        unhealthy_endpoints = [ep for ep in endpoints if not ep.is_healthy]
        
        # Aggregate statistics from healthy endpoints only
        aggregated_stats = self._create_aggregated_endpoint_stats(healthy_endpoints)
        unified_metrics['aggregated_stats'] = aggregated_stats
        
        # Per-endpoint breakdown
        unified_metrics['endpoints'] = {
            'summary': {
                'total_endpoints': len(endpoints),
                'healthy_endpoints': len(healthy_endpoints),
                'unhealthy_endpoints': len(unhealthy_endpoints),
                'total_gpus_all_endpoints': sum(ep.total_gpus for ep in endpoints),
                'total_gpus_healthy_endpoints': sum(ep.total_gpus for ep in healthy_endpoints),
                'available_gpus_healthy_endpoints': sum(ep.available_gpus for ep in healthy_endpoints),
                'overall_availability_rate': (
                    sum(ep.available_gpus for ep in healthy_endpoints) / 
                    sum(ep.total_gpus for ep in healthy_endpoints) * 100
                    if sum(ep.total_gpus for ep in healthy_endpoints) > 0 else 0.0
                )
            },
            'healthy_endpoints': [self._create_endpoint_metrics(ep, endpoint_health) for ep in healthy_endpoints],
            'unhealthy_endpoints': [self._create_endpoint_metrics(ep, endpoint_health) for ep in unhealthy_endpoints]
        }
        
        # Load balancer metrics
        if load_balancer_info:
            unified_metrics['load_balancer'] = load_balancer_info
        
        # Monitor status information
        if monitor_status:
            unified_metrics['monitor_status'] = monitor_status
        
        # Endpoint health summary
        if endpoint_health:
            unified_metrics['endpoint_health_summary'] = self._create_health_summary(endpoint_health)
        
        # Load balancing effectiveness metrics
        unified_metrics['load_balancing_effectiveness'] = self._calculate_load_balancing_effectiveness(
            healthy_endpoints, base_metrics.get('gpu_assignments', {})
        )
        
        return unified_metrics
    
    def _convert_gpu_assignments_to_global(self, 
                                         local_assignments: Dict[int, List[WorkerInfo]], 
                                         endpoints: List[EndpointInfo]) -> Dict[str, List[WorkerInfo]]:
        """
        Convert local GPU assignments to global GPU assignments.
        
        Args:
            local_assignments: Assignments with local GPU IDs
            endpoints: List of endpoint information
            
        Returns:
            Assignments with global GPU IDs
        """
        if not self.is_multi_endpoint_mode or not endpoints:
            # Convert local integer keys to strings for consistency
            return {str(gpu_id): workers for gpu_id, workers in local_assignments.items()}
        
        global_assignments = {}
        
        # In multi-endpoint mode, we need to map local GPU IDs to global ones
        # This is a simplified mapping - in practice, this would need more sophisticated logic
        for gpu_id, workers in local_assignments.items():
            if isinstance(gpu_id, str) and ':' in str(gpu_id):
                # Already a global GPU ID
                global_assignments[str(gpu_id)] = workers
            else:
                # Convert local GPU ID to global - this is a simplified approach
                # In reality, you'd need to track which endpoint each assignment came from
                global_assignments[str(gpu_id)] = workers
        
        return global_assignments
    
    def _create_aggregated_endpoint_stats(self, healthy_endpoints: List[EndpointInfo]) -> Dict[str, Any]:
        """Create aggregated statistics from healthy endpoints."""
        if not healthy_endpoints:
            return {
                'total_gpus': 0,
                'available_gpus': 0,
                'utilized_gpus': 0,
                'average_utilization_rate': 0.0,
                'average_response_time_ms': 0.0,
                'endpoints_contributing': 0
            }
        
        total_gpus = sum(ep.total_gpus for ep in healthy_endpoints)
        available_gpus = sum(ep.available_gpus for ep in healthy_endpoints)
        utilized_gpus = total_gpus - available_gpus
        
        # Calculate weighted average response time
        total_response_time = sum(ep.response_time_ms * ep.total_gpus for ep in healthy_endpoints)
        weighted_avg_response_time = (
            total_response_time / total_gpus if total_gpus > 0 else 0.0
        )
        
        return {
            'total_gpus': total_gpus,
            'available_gpus': available_gpus,
            'utilized_gpus': utilized_gpus,
            'average_utilization_rate': (utilized_gpus / total_gpus * 100) if total_gpus > 0 else 0.0,
            'average_response_time_ms': weighted_avg_response_time,
            'endpoints_contributing': len(healthy_endpoints)
        }
    
    def _create_endpoint_metrics(self, 
                               endpoint: EndpointInfo, 
                               endpoint_health: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Create detailed metrics for a single endpoint."""
        metrics = {
            'endpoint_id': endpoint.endpoint_id,
            'url': endpoint.url,
            'is_healthy': endpoint.is_healthy,
            'total_gpus': endpoint.total_gpus,
            'available_gpus': endpoint.available_gpus,
            'utilized_gpus': endpoint.total_gpus - endpoint.available_gpus,
            'utilization_rate': (
                (endpoint.total_gpus - endpoint.available_gpus) / endpoint.total_gpus * 100
                if endpoint.total_gpus > 0 else 0.0
            ),
            'response_time_ms': endpoint.response_time_ms,
            'last_seen': endpoint.last_seen.isoformat(),
            'last_seen_seconds_ago': int((datetime.now() - endpoint.last_seen).total_seconds())
        }
        
        # Add detailed health information if available
        if endpoint_health and endpoint.endpoint_id in endpoint_health:
            health_info = endpoint_health[endpoint.endpoint_id]
            metrics.update({
                'connectivity_status': health_info.get('connectivity_status', 'unknown'),
                'health_score': health_info.get('health_score', 0.0),
                'consecutive_failures': health_info.get('consecutive_failures', 0),
                'last_successful_communication': health_info.get('last_successful_communication'),
                'time_since_last_success': health_info.get('time_since_last_success', 'unknown')
            })
        
        return metrics
    
    def _create_health_summary(self, endpoint_health: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of endpoint health status."""
        if not endpoint_health:
            return {}
        
        health_values = list(endpoint_health.values())
        total_endpoints = len(health_values)
        
        # Count endpoints by connectivity status
        status_counts = {}
        health_scores = []
        
        for health_info in health_values:
            status = health_info.get('connectivity_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            score = health_info.get('health_score', 0.0)
            if isinstance(score, (int, float)):
                health_scores.append(score)
        
        # Calculate health statistics
        avg_health_score = sum(health_scores) / len(health_scores) if health_scores else 0.0
        min_health_score = min(health_scores) if health_scores else 0.0
        max_health_score = max(health_scores) if health_scores else 0.0
        
        return {
            'total_endpoints': total_endpoints,
            'status_distribution': status_counts,
            'health_scores': {
                'average': round(avg_health_score, 2),
                'minimum': round(min_health_score, 2),
                'maximum': round(max_health_score, 2)
            },
            'healthy_percentage': round(
                (status_counts.get('healthy', 0) / total_endpoints * 100) if total_endpoints > 0 else 0.0, 2
            )
        }
    
    def _calculate_load_balancing_effectiveness(self, 
                                              healthy_endpoints: List[EndpointInfo],
                                              gpu_assignments: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate load balancing effectiveness metrics."""
        if not healthy_endpoints:
            return {
                'distribution_score': 0.0,
                'utilization_variance': 0.0,
                'balancing_quality': 'no_endpoints'
            }
        
        # Calculate utilization rates for each endpoint
        utilization_rates = []
        for endpoint in healthy_endpoints:
            if endpoint.total_gpus > 0:
                utilization_rate = (endpoint.total_gpus - endpoint.available_gpus) / endpoint.total_gpus
                utilization_rates.append(utilization_rate)
        
        if not utilization_rates:
            return {
                'distribution_score': 0.0,
                'utilization_variance': 0.0,
                'balancing_quality': 'no_gpus'
            }
        
        # Calculate variance in utilization rates (lower is better for load balancing)
        mean_utilization = sum(utilization_rates) / len(utilization_rates)
        variance = sum((rate - mean_utilization) ** 2 for rate in utilization_rates) / len(utilization_rates)
        
        # Calculate distribution score (0-100, higher is better)
        # Perfect distribution would have variance = 0, score = 100
        max_possible_variance = 0.25  # Theoretical maximum for reasonable scenarios
        distribution_score = max(0.0, 100.0 * (1.0 - variance / max_possible_variance))
        
        # Determine balancing quality
        if variance < 0.01:
            quality = 'excellent'
        elif variance < 0.05:
            quality = 'good'
        elif variance < 0.15:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'distribution_score': round(distribution_score, 2),
            'utilization_variance': round(variance, 4),
            'balancing_quality': quality,
            'mean_utilization': round(mean_utilization, 3),
            'utilization_range': {
                'min': round(min(utilization_rates), 3),
                'max': round(max(utilization_rates), 3)
            } if utilization_rates else {'min': 0.0, 'max': 0.0}
        }


class MetricsFormatter:
    """Utility class for formatting metrics output in different formats."""
    
    @staticmethod
    def format_for_console(metrics: Dict[str, Any]) -> str:
        """Format metrics for console output."""
        lines = []
        lines.append("=== GPU Worker Pool Metrics ===")
        lines.append(f"Mode: {metrics.get('mode', 'unknown')}")
        lines.append(f"Timestamp: {metrics.get('timestamp', 'unknown')}")
        lines.append("")
        
        if metrics.get('mode') == 'multi-endpoint':
            # Multi-endpoint format
            endpoints = metrics.get('endpoints', {})
            summary = endpoints.get('summary', {})
            
            lines.append("--- Endpoint Summary ---")
            lines.append(f"Total Endpoints: {summary.get('total_endpoints', 0)}")
            lines.append(f"Healthy Endpoints: {summary.get('healthy_endpoints', 0)}")
            lines.append(f"Total GPUs (Healthy): {summary.get('total_gpus_healthy_endpoints', 0)}")
            lines.append(f"Available GPUs: {summary.get('available_gpus_healthy_endpoints', 0)}")
            lines.append(f"Availability Rate: {summary.get('overall_availability_rate', 0.0):.1f}%")
            lines.append("")
            
            # Load balancing effectiveness
            lb_eff = metrics.get('load_balancing_effectiveness', {})
            lines.append("--- Load Balancing ---")
            lines.append(f"Distribution Score: {lb_eff.get('distribution_score', 0.0):.1f}/100")
            lines.append(f"Balancing Quality: {lb_eff.get('balancing_quality', 'unknown')}")
            lines.append("")
            
        else:
            # Single-endpoint format
            lines.append("--- Single Endpoint ---")
            single_ep = metrics.get('single_endpoint', {})
            lines.append(f"Endpoint URL: {single_ep.get('endpoint_url', 'unknown')}")
            lines.append(f"Is Healthy: {single_ep.get('is_healthy', False)}")
            lines.append("")
        
        # Worker information
        lines.append("--- Workers ---")
        lines.append(f"Active Workers: {metrics.get('active_workers', 0)}")
        lines.append(f"Blocked Workers: {metrics.get('blocked_workers', 0)}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_for_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format metrics for JSON output (already in the right format)."""
        return metrics
    
    @staticmethod
    def format_health_summary(endpoint_health: Dict[str, Dict[str, Any]]) -> str:
        """Format endpoint health summary for console output."""
        if not endpoint_health:
            return "No endpoint health information available."
        
        lines = []
        lines.append("=== Endpoint Health Status ===")
        
        for endpoint_id, health_info in endpoint_health.items():
            lines.append(f"\nEndpoint: {endpoint_id}")
            lines.append(f"  Status: {health_info.get('connectivity_status', 'unknown')}")
            lines.append(f"  Health Score: {health_info.get('health_score', 0.0):.1f}/100")
            lines.append(f"  Response Time: {health_info.get('response_time_ms', 0):.1f}ms")
            lines.append(f"  Last Seen: {health_info.get('time_since_last_success', 'unknown')} ago")
        
        return "\n".join(lines)