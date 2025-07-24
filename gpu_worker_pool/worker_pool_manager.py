"""Main worker pool manager that orchestrates GPU resource allocation."""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime

from .models import GPUAssignment, PoolStatus, WorkerInfo, GPUStats
from .config import ConfigurationManager
from .gpu_monitor import GPUMonitor
from .gpu_allocator import GPUAllocator
from .worker_queue import WorkerQueue
from .resource_state import GPUResourceState, WorkerAssignmentTracker
from .monitoring import StructuredLogger, MetricsCollector, HealthChecker, PerformanceMonitor, HealthCheckResult


class StaleAssignmentError(Exception):
    """Raised when a worker assignment has become stale."""
    pass


class WorkerTimeoutError(Exception):
    """Raised when a worker request times out."""
    pass

logger = logging.getLogger(__name__)


class WorkerPoolManager(ABC):
    """Abstract base class for worker pool management."""
    
    @abstractmethod
    async def request_gpu(self) -> GPUAssignment:
        """Request GPU assignment for a worker."""
        pass
    
    @abstractmethod
    async def release_gpu(self, assignment: GPUAssignment) -> None:
        """Release a GPU assignment."""
        pass
    
    @abstractmethod
    def get_pool_status(self) -> PoolStatus:
        """Get current pool status."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the worker pool manager."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the worker pool manager."""
        pass


class AsyncWorkerPoolManager(WorkerPoolManager):
    """Async implementation of the worker pool manager."""
    
    def __init__(self,
                 config: ConfigurationManager,
                 gpu_monitor: GPUMonitor,
                 gpu_allocator: GPUAllocator,
                 worker_queue: WorkerQueue,
                 worker_timeout: float = 300.0,
                 stale_assignment_threshold: float = 3600.0,
                 metrics_collector: Optional[MetricsCollector] = None,
                 health_checker: Optional[HealthChecker] = None):
        """Initialize the worker pool manager.
        
        Args:
            config: Configuration manager
            gpu_monitor: GPU monitoring system
            gpu_allocator: GPU allocation logic
            worker_queue: Worker queue management
            worker_timeout: Timeout for worker requests in seconds
            stale_assignment_threshold: Time after which assignments are considered stale
            metrics_collector: Optional metrics collector for monitoring
            health_checker: Optional health checker for system health monitoring
        """
        self.config = config
        self.gpu_monitor = gpu_monitor
        self.gpu_allocator = gpu_allocator
        self.worker_queue = worker_queue
        self.worker_timeout = worker_timeout
        self.stale_assignment_threshold = stale_assignment_threshold
        
        # Monitoring components
        self.metrics_collector = metrics_collector
        self.health_checker = health_checker
        
        # Initialize structured logger and performance monitor
        self.logger = StructuredLogger("worker_pool_manager", metrics_collector)
        if metrics_collector:
            self.performance_monitor = PerformanceMonitor(metrics_collector, self.logger)
        else:
            self.performance_monitor = None
        
        # Internal state management
        self.resource_state = GPUResourceState()
        self.assignment_tracker = WorkerAssignmentTracker()
        
        # Synchronization
        self._allocation_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._is_running = False
        self._is_starting = False
        self._is_stopping = False
        
        # Lifecycle tracking
        self._start_time: Optional[datetime] = None
        self._stop_time: Optional[datetime] = None
        
        # Cleanup task for stale assignments
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 60.0  # Check for stale assignments every minute
        
        # Register for GPU stats updates
        self.gpu_monitor.on_stats_update(self._on_gpu_stats_update)
        
        # Register health checks
        self._register_health_checks()
        
        self.logger.info("WorkerPoolManager initialized", 
                        worker_timeout=worker_timeout,
                        stale_assignment_threshold=stale_assignment_threshold)
    
    async def request_gpu(self, timeout: Optional[float] = None) -> GPUAssignment:
        """
        Request GPU assignment for a worker with blocking logic and timeout.
        
        This method will either immediately assign an available GPU or block
        the worker in a queue until resources become available.
        
        Args:
            timeout: Optional timeout in seconds. If None, uses default worker_timeout
        
        Returns:
            GPUAssignment when a GPU becomes available
            
        Raises:
            RuntimeError: If the pool manager is not running
            WorkerTimeoutError: If the request times out
        """
        if not self._is_running:
            error_msg = "Worker pool manager is not running"
            logger.error(f"GPU request rejected: {error_msg}")
            raise RuntimeError(error_msg)
        
        worker_id = str(uuid.uuid4())
        request_start_time = datetime.now()
        
        logger.info(
            f"Worker {worker_id} requesting GPU assignment at {request_start_time.isoformat()}"
        )
        
        # Log current pool state for context
        current_status = self.get_pool_status()
        logger.debug(
            f"Pool state at request time - Total GPUs: {current_status.total_gpus}, "
            f"Available: {current_status.available_gpus}, Active workers: {current_status.active_workers}, "
            f"Blocked workers: {current_status.blocked_workers}"
        )
        
        # Create a future that will be resolved when GPU is assigned
        assignment_future = asyncio.Future()
        
        def on_gpu_assigned(gpu_id: int) -> None:
            """Callback when GPU is assigned to worker."""
            try:
                assignment = self.assignment_tracker.assign(worker_id, gpu_id)
                assignment_future.set_result(assignment)
                
                assignment_duration = (datetime.now() - request_start_time).total_seconds()
                logger.info(
                    f"WORKER_ASSIGNED: Worker {worker_id} assigned to GPU {gpu_id} "
                    f"after {assignment_duration:.2f}s at {assignment.assigned_at.isoformat()}"
                )
                
                # Log detailed assignment context
                current_stats = self.resource_state.get_current_stats()
                if current_stats:
                    gpu_info = next((gpu for gpu in current_stats.gpus_summary if gpu.gpu_id == gpu_id), None)
                    if gpu_info:
                        logger.info(
                            f"GPU {gpu_id} assignment context - Memory: {gpu_info.memory_usage_percent}%, "
                            f"Utilization: {gpu_info.utilization_percent}%, Name: {gpu_info.name}"
                        )
                
            except Exception as e:
                assignment_future.set_exception(e)
                logger.error(
                    f"ASSIGNMENT_FAILED: Failed to assign worker {worker_id} to GPU {gpu_id}: {e}",
                    exc_info=True
                )
        
        def on_assignment_error(error: Exception) -> None:
            """Callback when assignment fails."""
            assignment_future.set_exception(error)
            error_duration = (datetime.now() - request_start_time).total_seconds()
            logger.error(
                f"WORKER_ASSIGNMENT_ERROR: Worker {worker_id} assignment failed after {error_duration:.2f}s: {error}",
                exc_info=True
            )
        
        # Create worker info
        worker = WorkerInfo(
            id=worker_id,
            enqueued_at=request_start_time,
            callback=on_gpu_assigned,
            on_error=on_assignment_error
        )
        
        # Try immediate assignment or queue the worker
        async with self._allocation_lock:
            assigned_gpu = await self._try_immediate_assignment(worker)
            if assigned_gpu is not None:
                # Immediate assignment successful
                logger.debug(f"Immediate assignment successful for worker {worker_id} to GPU {assigned_gpu}")
                on_gpu_assigned(assigned_gpu)
            else:
                # Block worker in queue
                blocking_reason = "No GPUs available or all GPUs exceed resource thresholds"
                logger.info(
                    f"WORKER_BLOCKED: Worker {worker_id} blocked in queue - {blocking_reason}. "
                    f"Queue size before blocking: {self.worker_queue.size()}"
                )
                self.worker_queue.block_worker(worker, blocking_reason)
                
                logger.debug(
                    f"Worker {worker_id} added to queue. New queue size: {self.worker_queue.size()}"
                )
        
        # Wait for assignment with timeout
        request_timeout = timeout if timeout is not None else self.worker_timeout
        
        try:
            return await asyncio.wait_for(assignment_future, timeout=request_timeout)
        except asyncio.TimeoutError:
            # Remove worker from queue if it was queued
            if hasattr(self.worker_queue, 'remove_worker'):
                self.worker_queue.remove_worker(worker_id)
            
            error_msg = f"Worker {worker_id} request timed out after {request_timeout:.1f} seconds"
            logger.error(f"WORKER_TIMEOUT: {error_msg}")
            raise WorkerTimeoutError(error_msg)
    
    async def release_gpu(self, assignment: GPUAssignment) -> None:
        """
        Release a GPU assignment with proper cleanup and worker unblocking.
        
        Args:
            assignment: The GPU assignment to release
            
        Raises:
            ValueError: If assignment is invalid or not found
        """
        if not isinstance(assignment, GPUAssignment):
            error_msg = f"assignment must be a GPUAssignment instance, got {type(assignment)}"
            logger.error(f"GPU release failed: {error_msg}")
            raise ValueError(error_msg)
        
        release_start_time = datetime.now()
        logger.info(
            f"GPU_RELEASE_REQUESTED: Releasing GPU {assignment.gpu_id} from worker {assignment.worker_id} "
            f"at {release_start_time.isoformat()}"
        )
        
        # Log assignment duration
        assignment_duration = (release_start_time - assignment.assigned_at).total_seconds()
        logger.info(
            f"Assignment duration: {assignment_duration:.2f}s for worker {assignment.worker_id} on GPU {assignment.gpu_id}"
        )
        
        async with self._allocation_lock:
            # Log pre-release state
            pre_release_status = self.get_pool_status()
            logger.debug(
                f"Pre-release pool state - Active workers: {pre_release_status.active_workers}, "
                f"Blocked workers: {pre_release_status.blocked_workers}, Available GPUs: {pre_release_status.available_gpus}"
            )
            
            # Remove from assignment tracker
            released_assignment = self.assignment_tracker.release(assignment.worker_id)
            if released_assignment is None:
                logger.warning(
                    f"RELEASE_WARNING: Worker {assignment.worker_id} was not found in assignment tracker. "
                    f"Possible duplicate release or worker was never properly assigned."
                )
                return
            
            # Remove from resource state
            success = self.resource_state.remove_assignment(assignment.gpu_id, assignment.worker_id)
            if not success:
                logger.warning(
                    f"RELEASE_WARNING: Worker {assignment.worker_id} was not found in GPU {assignment.gpu_id} assignments. "
                    f"Resource state may be inconsistent."
                )
            
            release_duration = (datetime.now() - release_start_time).total_seconds()
            logger.info(
                f"GPU_RELEASED: Successfully released worker {assignment.worker_id} from GPU {assignment.gpu_id} "
                f"in {release_duration:.3f}s"
            )
            
            # Log GPU context after release
            current_stats = self.resource_state.get_current_stats()
            if current_stats:
                gpu_info = next((gpu for gpu in current_stats.gpus_summary if gpu.gpu_id == assignment.gpu_id), None)
                if gpu_info:
                    logger.info(
                        f"GPU {assignment.gpu_id} post-release state - Memory: {gpu_info.memory_usage_percent}%, "
                        f"Utilization: {gpu_info.utilization_percent}%"
                    )
            
            # Try to unblock waiting workers
            queue_size_before = self.worker_queue.size()
            if queue_size_before > 0:
                logger.info(f"Processing {queue_size_before} blocked workers after GPU release")
                await self._process_worker_queue()
                
                queue_size_after = self.worker_queue.size()
                unblocked_count = queue_size_before - queue_size_after
                if unblocked_count > 0:
                    logger.info(f"Unblocked {unblocked_count} workers after GPU {assignment.gpu_id} release")
            else:
                logger.debug("No blocked workers to process after GPU release")
    
    def get_pool_status(self) -> PoolStatus:
        """
        Get current pool status with metrics.
        
        Returns:
            PoolStatus with current pool metrics
        """
        current_stats = self.resource_state.get_current_stats()
        all_assignments = self.resource_state.get_all_assignments()
        
        total_gpus = current_stats.gpu_count if current_stats else 0
        active_workers = self.assignment_tracker.get_assignment_count()
        blocked_workers = self.worker_queue.size()
        
        # Calculate available GPUs
        available_gpus = 0
        if current_stats:
            for gpu in current_stats.gpus_summary:
                assigned_workers = all_assignments.get(gpu.gpu_id, [])
                if self.gpu_allocator.is_gpu_available(gpu, assigned_workers):
                    available_gpus += 1
        
        status = PoolStatus(
            total_gpus=total_gpus,
            available_gpus=available_gpus,
            active_workers=active_workers,
            blocked_workers=blocked_workers,
            gpu_assignments=all_assignments
        )
        
        # Log pool status metrics
        logger.info(
            f"Pool status: {total_gpus} total GPUs, {available_gpus} available, "
            f"{active_workers} active workers, {blocked_workers} blocked workers"
        )
        
        return status
    
    def get_detailed_pool_metrics(self) -> Dict[str, Any]:
        """
        Get detailed pool metrics for monitoring and debugging.
        
        Returns:
            Dictionary with comprehensive pool metrics
        """
        current_stats = self.resource_state.get_current_stats()
        all_assignments = self.resource_state.get_all_assignments()
        all_worker_assignments = self.assignment_tracker.get_all_assignments()
        
        # GPU-level metrics
        gpu_metrics = []
        if current_stats:
            for gpu in current_stats.gpus_summary:
                assigned_workers = all_assignments.get(gpu.gpu_id, [])
                is_available = self.gpu_allocator.is_gpu_available(gpu, assigned_workers)
                
                gpu_metrics.append({
                    "gpu_id": gpu.gpu_id,
                    "name": gpu.name,
                    "memory_usage_percent": gpu.memory_usage_percent,
                    "utilization_percent": gpu.utilization_percent,
                    "assigned_workers": len(assigned_workers),
                    "worker_ids": [w.id for w in assigned_workers],
                    "is_available": is_available,
                    "resource_score": self.gpu_allocator.calculate_gpu_score(gpu)
                })
        
        # Worker assignment metrics
        assignment_metrics = []
        for assignment in all_worker_assignments:
            assignment_metrics.append({
                "worker_id": assignment.worker_id,
                "gpu_id": assignment.gpu_id,
                "assigned_at": assignment.assigned_at.isoformat(),
                "duration_seconds": (datetime.now() - assignment.assigned_at).total_seconds()
            })
        
        # Queue metrics
        queue_workers = []
        if hasattr(self.worker_queue, 'get_all_workers'):
            for worker in self.worker_queue.get_all_workers():
                queue_workers.append({
                    "worker_id": worker.id,
                    "enqueued_at": worker.enqueued_at.isoformat(),
                    "wait_time_seconds": (datetime.now() - worker.enqueued_at).total_seconds()
                })
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "is_running": self._is_running,
            "total_gpus": len(gpu_metrics),
            "available_gpus": sum(1 for gpu in gpu_metrics if gpu["is_available"]),
            "active_workers": len(assignment_metrics),
            "blocked_workers": len(queue_workers),
            "gpu_metrics": gpu_metrics,
            "assignment_metrics": assignment_metrics,
            "queue_metrics": queue_workers,
            "thresholds": {
                "memory_threshold_percent": self.config.get_memory_threshold(),
                "utilization_threshold_percent": self.config.get_utilization_threshold()
            }
        }
        
        logger.debug(f"Generated detailed pool metrics with {len(gpu_metrics)} GPUs and {len(assignment_metrics)} assignments")
        
        return metrics
    
    async def start(self) -> None:
        """Start the worker pool manager and all components with proper lifecycle management."""
        async with self._lifecycle_lock:
            # Check current state
            if self._is_running:
                logger.warning("POOL_START_WARNING: Worker pool manager is already running")
                return
            
            if self._is_starting:
                logger.warning("POOL_START_WARNING: Worker pool manager is already starting")
                return
            
            if self._is_stopping:
                logger.error("POOL_START_ERROR: Cannot start while pool is stopping")
                raise RuntimeError("Cannot start worker pool manager while it is stopping")
            
            self._is_starting = True
            start_time = datetime.now()
            logger.info(f"POOL_STARTING: Starting worker pool manager at {start_time.isoformat()}")
            
            try:
                # Log configuration for debugging
                # Check if config has multi-endpoint support
                if hasattr(self.config, 'get_service_endpoints'):
                    endpoints_info = f"Service endpoints: {', '.join(self.config.get_service_endpoints())}"
                else:
                    endpoints_info = f"Service endpoint: {self.config.get_service_endpoint()}"
                    
                logger.info(
                    f"Pool configuration - Memory threshold: {self.config.get_memory_threshold()}%, "
                    f"Utilization threshold: {self.config.get_utilization_threshold()}%, "
                    f"Polling interval: {self.config.get_polling_interval()}s, "
                    f"{endpoints_info}"
                )
                
                # Initialize state
                logger.debug("Initializing pool state")
                self.resource_state.clear_all()
                self.assignment_tracker.clear_all()
                self.worker_queue.clear()
                
                # Start GPU monitor
                logger.debug("Starting GPU monitor component")
                await self.gpu_monitor.start()
                logger.debug("GPU monitor started successfully")
                
                # Start stale assignment cleanup task
                logger.debug("Starting stale assignment cleanup task")
                self._cleanup_task = asyncio.create_task(self._stale_assignment_cleanup_loop())
                logger.debug("Stale assignment cleanup task started")
                
                # Mark as running
                self._is_running = True
                self._start_time = start_time
                self._stop_time = None
                
                startup_duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"POOL_STARTED: Worker pool manager started successfully in {startup_duration:.3f}s"
                )
                
            except Exception as e:
                logger.error(
                    f"POOL_START_FAILED: Failed to start worker pool manager: {e}",
                    exc_info=True
                )
                
                # Cleanup on failure
                await self._cleanup_on_start_failure()
                raise
                
            finally:
                self._is_starting = False
    
    async def stop(self) -> None:
        """Stop the worker pool manager and cleanup resources with proper lifecycle management."""
        async with self._lifecycle_lock:
            # Check current state
            if not self._is_running:
                logger.debug("POOL_STOP_SKIPPED: Worker pool manager is not running")
                return
            
            if self._is_stopping:
                logger.warning("POOL_STOP_WARNING: Worker pool manager is already stopping")
                return
            
            if self._is_starting:
                logger.warning("POOL_STOP_WARNING: Stopping worker pool manager while it is starting")
            
            self._is_stopping = True
            stop_time = datetime.now()
            logger.info(f"POOL_STOPPING: Stopping worker pool manager at {stop_time.isoformat()}")
            
            # Log uptime if we have start time
            if self._start_time:
                uptime = (stop_time - self._start_time).total_seconds()
                logger.info(f"Pool uptime: {uptime:.1f} seconds")
            
            try:
                # Log final pool state before shutdown
                final_status = self.get_pool_status()
                logger.info(
                    f"Final pool state - Active workers: {final_status.active_workers}, "
                    f"Blocked workers: {final_status.blocked_workers}, Total GPUs: {final_status.total_gpus}"
                )
                
                # Mark as not running to prevent new requests
                self._is_running = False
                
                # Stop stale assignment cleanup task
                if self._cleanup_task and not self._cleanup_task.done():
                    logger.debug("Stopping stale assignment cleanup task")
                    self._cleanup_task.cancel()
                    try:
                        await self._cleanup_task
                    except asyncio.CancelledError:
                        pass
                    logger.debug("Stale assignment cleanup task stopped")
                
                # Stop GPU monitor first to prevent new stats updates
                logger.debug("Stopping GPU monitor component")
                await self.gpu_monitor.stop()
                logger.debug("GPU monitor stopped successfully")
                
                # Clear all queued workers with cancellation
                blocked_worker_count = self.worker_queue.size()
                if blocked_worker_count > 0:
                    logger.info(f"Cancelling {blocked_worker_count} blocked workers")
                    self.worker_queue.clear()
                    logger.info(f"Cancelled {blocked_worker_count} blocked workers")
                
                # Log active assignments before cleanup
                active_assignments = self.assignment_tracker.get_assignment_count()
                if active_assignments > 0:
                    logger.warning(
                        f"CLEANUP_WARNING: Clearing {active_assignments} active assignments during shutdown. "
                        f"Workers may not have been properly released."
                    )
                    
                    # Log details of active assignments for debugging
                    all_assignments = self.assignment_tracker.get_all_assignments()
                    for assignment in all_assignments:
                        duration = (stop_time - assignment.assigned_at).total_seconds()
                        logger.debug(
                            f"Active assignment: Worker {assignment.worker_id} on GPU {assignment.gpu_id} "
                            f"for {duration:.1f}s"
                        )
                
                # Clear all state
                self.resource_state.clear_all()
                self.assignment_tracker.clear_all()
                
                # Update lifecycle tracking
                self._stop_time = stop_time
                
                shutdown_duration = (datetime.now() - stop_time).total_seconds()
                logger.info(
                    f"POOL_STOPPED: Worker pool manager stopped successfully in {shutdown_duration:.3f}s"
                )
                
            except Exception as e:
                logger.error(
                    f"POOL_STOP_ERROR: Error during worker pool manager shutdown: {e}",
                    exc_info=True
                )
                # Ensure we're marked as stopped even if cleanup fails
                self._is_running = False
                raise
                
            finally:
                self._is_stopping = False
    
    async def _try_immediate_assignment(self, worker: WorkerInfo) -> Optional[int]:
        """
        Try to immediately assign a worker to an available GPU.
        
        Args:
            worker: Worker requesting assignment
            
        Returns:
            GPU ID if assignment successful, None if no GPUs available
        """
        current_stats = self.resource_state.get_current_stats()
        if not current_stats:
            logger.debug("No GPU statistics available for immediate assignment")
            return None
        
        current_assignments = self.resource_state.get_all_assignments()
        
        # Find available GPU
        gpu_id = self.gpu_allocator.find_available_gpu(current_stats, current_assignments)
        if gpu_id is None:
            logger.debug("No GPUs available for immediate assignment")
            return None
        
        # Add assignment to resource state
        try:
            self.resource_state.add_assignment(gpu_id, worker)
            logger.debug(f"Added worker {worker.id} to GPU {gpu_id} resource state")
            return gpu_id
        except Exception as e:
            logger.error(f"Failed to add assignment to resource state: {e}")
            return None
    
    async def _process_worker_queue(self) -> None:
        """Process the worker queue and assign GPUs to waiting workers."""
        if self.worker_queue.size() == 0:
            return
        
        logger.debug(f"Processing worker queue with {self.worker_queue.size()} waiting workers")
        
        current_stats = self.resource_state.get_current_stats()
        if not current_stats:
            logger.debug("No GPU statistics available for queue processing")
            return
        
        # Process workers one by one until no more GPUs are available
        while self.worker_queue.size() > 0:
            current_assignments = self.resource_state.get_all_assignments()
            
            # Find available GPU
            gpu_id = self.gpu_allocator.find_available_gpu(current_stats, current_assignments)
            if gpu_id is None:
                logger.debug("No more GPUs available for queue processing")
                break
            
            # Unblock next worker
            worker = self.worker_queue.unblock_next_worker()
            if worker is None:
                logger.debug("No workers in queue to unblock")
                break
            
            try:
                # Add assignment to resource state
                self.resource_state.add_assignment(gpu_id, worker)
                
                # Notify worker of assignment
                worker.callback(gpu_id)
                
                logger.info(f"Unblocked and assigned worker {worker.id} to GPU {gpu_id}")
                
            except Exception as e:
                logger.error(f"Failed to assign unblocked worker {worker.id}: {e}")
                # Notify worker of error
                try:
                    worker.on_error(e)
                except Exception:
                    pass  # Ignore errors in error callbacks
    
    def _on_gpu_stats_update(self, stats: GPUStats) -> None:
        """
        Handle GPU statistics updates.
        
        This callback is called whenever new GPU statistics are received
        from the monitoring system.
        
        Args:
            stats: Updated GPU statistics
        """
        logger.debug(f"Received GPU stats update: {stats.gpu_count} GPUs")
        
        # Update resource state
        self.resource_state.update_stats(stats)
        
        # Process worker queue in case new GPUs became available
        # Use asyncio.create_task to avoid blocking the callback
        if self._is_running:
            asyncio.create_task(self._safe_process_worker_queue())
    
    async def _safe_process_worker_queue(self) -> None:
        """Safely process worker queue with error handling."""
        try:
            async with self._allocation_lock:
                await self._process_worker_queue()
        except Exception as e:
            self.logger.error("Error processing worker queue", error=str(e))
    
    def _register_health_checks(self) -> None:
        """Register health checks for the worker pool manager."""
        if not self.health_checker:
            return
        
        # Register pool manager health check
        self.health_checker.register_health_check("pool_manager", self._pool_manager_health_check)
        
        # Register GPU monitor health check
        self.health_checker.register_health_check("gpu_monitor", self._gpu_monitor_health_check)
        
        # Register resource state health check
        self.health_checker.register_health_check("resource_state", self._resource_state_health_check)
    
    def _pool_manager_health_check(self) -> HealthCheckResult:
        """Health check for the pool manager itself."""
        if not self._is_running:
            return HealthCheckResult(
                name="pool_manager",
                status="unhealthy",
                message="Pool manager is not running",
                timestamp=datetime.now(),
                details={"is_running": False}
            )
        
        # Check for excessive stale assignments
        stale_status = self.get_stale_assignment_status()
        stale_count = stale_status["stale_assignments"]
        total_assignments = stale_status["total_assignments"]
        
        if total_assignments > 0 and stale_count / total_assignments > 0.5:
            return HealthCheckResult(
                name="pool_manager",
                status="degraded",
                message=f"High number of stale assignments: {stale_count}/{total_assignments}",
                timestamp=datetime.now(),
                details=stale_status
            )
        
        # Check for excessive queue size
        queue_size = self.worker_queue.size()
        if queue_size > 100:  # Arbitrary threshold
            return HealthCheckResult(
                name="pool_manager",
                status="degraded",
                message=f"Large worker queue: {queue_size} workers blocked",
                timestamp=datetime.now(),
                details={"queue_size": queue_size}
            )
        
        return HealthCheckResult(
            name="pool_manager",
            status="healthy",
            message="Pool manager is running normally",
            timestamp=datetime.now(),
            details={
                "is_running": self._is_running,
                "queue_size": queue_size,
                "stale_assignments": stale_count,
                "total_assignments": total_assignments
            }
        )
    
    def _gpu_monitor_health_check(self) -> HealthCheckResult:
        """Health check for the GPU monitor."""
        current_stats = self.resource_state.get_current_stats()
        
        if current_stats is None:
            return HealthCheckResult(
                name="gpu_monitor",
                status="unhealthy",
                message="No GPU statistics available",
                timestamp=datetime.now(),
                details={"has_stats": False}
            )
        
        # Check if stats are recent (within last 30 seconds)
        if hasattr(self.gpu_monitor, 'get_monitor_status'):
            monitor_status = self.gpu_monitor.get_monitor_status()
            consecutive_failures = monitor_status.get("consecutive_failures", 0)
            
            if consecutive_failures > 5:
                return HealthCheckResult(
                    name="gpu_monitor",
                    status="degraded",
                    message=f"GPU monitor has {consecutive_failures} consecutive failures",
                    timestamp=datetime.now(),
                    details=monitor_status
                )
        
        return HealthCheckResult(
            name="gpu_monitor",
            status="healthy",
            message="GPU monitor is functioning normally",
            timestamp=datetime.now(),
            details={
                "has_stats": True,
                "gpu_count": current_stats.gpu_count,
                "timestamp": current_stats.timestamp
            }
        )
    
    def _resource_state_health_check(self) -> HealthCheckResult:
        """Health check for resource state consistency."""
        try:
            # Check for consistency between assignment tracker and resource state
            tracker_assignments = self.assignment_tracker.get_all_assignments()
            resource_assignments = self.resource_state.get_all_assignments()
            
            # Count total assignments from both sources
            tracker_count = len(tracker_assignments)
            resource_count = sum(len(workers) for workers in resource_assignments.values())
            
            if tracker_count != resource_count:
                return HealthCheckResult(
                    name="resource_state",
                    status="degraded",
                    message=f"Assignment count mismatch: tracker={tracker_count}, resource={resource_count}",
                    timestamp=datetime.now(),
                    details={
                        "tracker_assignments": tracker_count,
                        "resource_assignments": resource_count
                    }
                )
            
            return HealthCheckResult(
                name="resource_state",
                status="healthy",
                message="Resource state is consistent",
                timestamp=datetime.now(),
                details={
                    "total_assignments": tracker_count,
                    "consistency_check": "passed"
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="resource_state",
                status="unhealthy",
                message=f"Error checking resource state: {e}",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
    
    async def get_health_status(self) -> Dict[str, HealthCheckResult]:
        """Get health status for all components.
        
        Returns:
            Dictionary of health check results
        """
        if not self.health_checker:
            return {}
        
        return await self.health_checker.run_health_checks()
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring metrics.
        
        Returns:
            Dictionary with all monitoring data
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "pool_metrics": self.get_detailed_pool_metrics(),
            "lifecycle_status": self.get_lifecycle_status(),
            "stale_assignment_status": self.get_stale_assignment_status()
        }
        
        # Add performance metrics if available
        if self.performance_monitor:
            metrics["performance_summary"] = self.performance_monitor.get_performance_summary()
        
        # Add collected metrics if available
        if self.metrics_collector:
            if hasattr(self.metrics_collector, 'get_metric_summary'):
                metrics["metric_summary"] = self.metrics_collector.get_metric_summary()
            else:
                metrics["raw_metrics"] = self.metrics_collector.get_metrics()
        
        return metrics
    
    async def _stale_assignment_cleanup_loop(self) -> None:
        """Background task to cleanup stale worker assignments."""
        self.logger.info("Starting stale assignment cleanup loop", cleanup_interval=self._cleanup_interval)
        
        while self._is_running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                
                if not self._is_running:
                    break
                
                await self._cleanup_stale_assignments()
                
            except asyncio.CancelledError:
                self.logger.debug("Stale assignment cleanup loop cancelled")
                break
            except Exception as e:
                self.logger.error("Error in stale assignment cleanup loop", error=str(e))
                # Continue running even if cleanup fails
    
    async def _cleanup_stale_assignments(self) -> None:
        """Clean up stale worker assignments."""
        current_time = datetime.now()
        stale_assignments = []
        
        # Find stale assignments
        all_assignments = self.assignment_tracker.get_all_assignments()
        for assignment in all_assignments:
            assignment_age = (current_time - assignment.assigned_at).total_seconds()
            if assignment_age > self.stale_assignment_threshold:
                stale_assignments.append(assignment)
        
        if not stale_assignments:
            return
        
        self.logger.warning("Found stale assignments to cleanup", count=len(stale_assignments))
        
        # Cleanup stale assignments
        async with self._allocation_lock:
            for assignment in stale_assignments:
                try:
                    assignment_age = (current_time - assignment.assigned_at).total_seconds()
                    self.logger.warning(
                        "Cleaning up stale assignment",
                        worker_id=assignment.worker_id,
                        gpu_id=assignment.gpu_id,
                        age_seconds=assignment_age
                    )
                    
                    # Remove from assignment tracker
                    self.assignment_tracker.release(assignment.worker_id)
                    
                    # Remove from resource state
                    self.resource_state.remove_assignment(assignment.gpu_id, assignment.worker_id)
                    
                    self.logger.info("Cleaned up stale assignment", worker_id=assignment.worker_id)
                    
                except Exception as e:
                    self.logger.error("Error cleaning up stale assignment", 
                                    worker_id=assignment.worker_id, error=str(e))
            
            # Process worker queue after cleanup to assign newly available GPUs
            if stale_assignments:
                await self._process_worker_queue()
    
    def get_stale_assignment_status(self) -> Dict[str, Any]:
        """Get status of stale assignment cleanup for monitoring."""
        current_time = datetime.now()
        all_assignments = self.assignment_tracker.get_all_assignments()
        
        assignment_ages = []
        stale_count = 0
        
        for assignment in all_assignments:
            age_seconds = (current_time - assignment.assigned_at).total_seconds()
            assignment_ages.append({
                "worker_id": assignment.worker_id,
                "gpu_id": assignment.gpu_id,
                "age_seconds": age_seconds,
                "is_stale": age_seconds > self.stale_assignment_threshold
            })
            
            if age_seconds > self.stale_assignment_threshold:
                stale_count += 1
        
        return {
            "stale_assignment_threshold": self.stale_assignment_threshold,
            "cleanup_interval": self._cleanup_interval,
            "total_assignments": len(all_assignments),
            "stale_assignments": stale_count,
            "assignment_ages": assignment_ages,
            "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done()
        }
    
    async def _cleanup_on_start_failure(self) -> None:
        """Cleanup resources when start fails."""
        self.logger.debug("Performing cleanup after start failure")
        
        try:
            # Ensure we're not marked as running
            self._is_running = False
            self._start_time = None
            
            # Try to stop GPU monitor if it was started
            try:
                await self.gpu_monitor.stop()
                self.logger.debug("GPU monitor stopped during cleanup")
            except Exception as e:
                self.logger.debug("Error stopping GPU monitor during cleanup", error=str(e))
            
            # Clear any state that might have been initialized
            try:
                self.resource_state.clear_all()
                self.assignment_tracker.clear_all()
                self.worker_queue.clear()
                self.logger.debug("State cleared during cleanup")
            except Exception as e:
                self.logger.debug("Error clearing state during cleanup", error=str(e))
                
        except Exception as e:
            self.logger.error("Error during start failure cleanup", error=str(e))
    
    def get_lifecycle_status(self) -> Dict[str, Any]:
        """
        Get current lifecycle status for monitoring and debugging.
        
        Returns:
            Dictionary with lifecycle status information
        """
        status = {
            "is_running": self._is_running,
            "is_starting": self._is_starting,
            "is_stopping": self._is_stopping,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "stop_time": self._stop_time.isoformat() if self._stop_time else None,
            "uptime_seconds": None
        }
        
        # Calculate uptime if running
        if self._is_running and self._start_time:
            status["uptime_seconds"] = (datetime.now() - self._start_time).total_seconds()
        elif self._stop_time and self._start_time:
            status["uptime_seconds"] = (self._stop_time - self._start_time).total_seconds()
        
        return status
        """Register health checks for the worker pool manager."""
        if not self.health_checker:
            return
        
        # Register pool manager health check
        self.health_checker.register_health_check("pool_manager", self._pool_manager_health_check)
        
        # Register GPU monitor health check
        self.health_checker.register_health_check("gpu_monitor", self._gpu_monitor_health_check)
        
        # Register resource state health check
        self.health_checker.register_health_check("resource_state", self._resource_state_health_check)
    
    def _pool_manager_health_check(self) -> HealthCheckResult:
        """Health check for the pool manager itself."""
        if not self._is_running:
            return HealthCheckResult(
                name="pool_manager",
                status="unhealthy",
                message="Pool manager is not running",
                timestamp=datetime.now(),
                details={"is_running": False}
            )
        
        # Check for excessive stale assignments
        stale_status = self.get_stale_assignment_status()
        stale_count = stale_status["stale_assignments"]
        total_assignments = stale_status["total_assignments"]
        
        if total_assignments > 0 and stale_count / total_assignments > 0.5:
            return HealthCheckResult(
                name="pool_manager",
                status="degraded",
                message=f"High number of stale assignments: {stale_count}/{total_assignments}",
                timestamp=datetime.now(),
                details=stale_status
            )
        
        # Check for excessive queue size
        queue_size = self.worker_queue.size()
        if queue_size > 100:  # Arbitrary threshold
            return HealthCheckResult(
                name="pool_manager",
                status="degraded",
                message=f"Large worker queue: {queue_size} workers blocked",
                timestamp=datetime.now(),
                details={"queue_size": queue_size}
            )
        
        return HealthCheckResult(
            name="pool_manager",
            status="healthy",
            message="Pool manager is running normally",
            timestamp=datetime.now(),
            details={
                "is_running": self._is_running,
                "queue_size": queue_size,
                "stale_assignments": stale_count,
                "total_assignments": total_assignments
            }
        )
    
    def _gpu_monitor_health_check(self) -> HealthCheckResult:
        """Health check for the GPU monitor."""
        current_stats = self.resource_state.get_current_stats()
        
        if current_stats is None:
            return HealthCheckResult(
                name="gpu_monitor",
                status="unhealthy",
                message="No GPU statistics available",
                timestamp=datetime.now(),
                details={"has_stats": False}
            )
        
        # Check if stats are recent (within last 30 seconds)
        if hasattr(self.gpu_monitor, 'get_monitor_status'):
            monitor_status = self.gpu_monitor.get_monitor_status()
            consecutive_failures = monitor_status.get("consecutive_failures", 0)
            
            if consecutive_failures > 5:
                return HealthCheckResult(
                    name="gpu_monitor",
                    status="degraded",
                    message=f"GPU monitor has {consecutive_failures} consecutive failures",
                    timestamp=datetime.now(),
                    details=monitor_status
                )
        
        return HealthCheckResult(
            name="gpu_monitor",
            status="healthy",
            message="GPU monitor is functioning normally",
            timestamp=datetime.now(),
            details={
                "has_stats": True,
                "gpu_count": current_stats.gpu_count,
                "timestamp": current_stats.timestamp
            }
        )
    
    def _resource_state_health_check(self) -> HealthCheckResult:
        """Health check for resource state consistency."""
        try:
            # Check for consistency between assignment tracker and resource state
            tracker_assignments = self.assignment_tracker.get_all_assignments()
            resource_assignments = self.resource_state.get_all_assignments()
            
            # Count total assignments from both sources
            tracker_count = len(tracker_assignments)
            resource_count = sum(len(workers) for workers in resource_assignments.values())
            
            if tracker_count != resource_count:
                return HealthCheckResult(
                    name="resource_state",
                    status="degraded",
                    message=f"Assignment count mismatch: tracker={tracker_count}, resource={resource_count}",
                    timestamp=datetime.now(),
                    details={
                        "tracker_assignments": tracker_count,
                        "resource_assignments": resource_count
                    }
                )
            
            return HealthCheckResult(
                name="resource_state",
                status="healthy",
                message="Resource state is consistent",
                timestamp=datetime.now(),
                details={
                    "total_assignments": tracker_count,
                    "consistency_check": "passed"
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="resource_state",
                status="unhealthy",
                message=f"Error checking resource state: {e}",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
    
    async def get_health_status(self) -> Dict[str, HealthCheckResult]:
        """Get health status for all components.
        
        Returns:
            Dictionary of health check results
        """
        if not self.health_checker:
            return {}
        
        return await self.health_checker.run_health_checks()
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring metrics.
        
        Returns:
            Dictionary with all monitoring data
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "pool_metrics": self.get_detailed_pool_metrics(),
            "lifecycle_status": self.get_lifecycle_status(),
            "stale_assignment_status": self.get_stale_assignment_status()
        }
        
        # Add performance metrics if available
        if self.performance_monitor:
            metrics["performance_summary"] = self.performance_monitor.get_performance_summary()
        
        # Add collected metrics if available
        if self.metrics_collector:
            if hasattr(self.metrics_collector, 'get_metric_summary'):
                metrics["metric_summary"] = self.metrics_collector.get_metric_summary()
            else:
                metrics["raw_metrics"] = self.metrics_collector.get_metrics()
        
        return metrics

    async def _safe_process_worker_queue(self) -> None:
        """Safely process worker queue with error handling."""
        try:
            async with self._allocation_lock:
                await self._process_worker_queue()
        except Exception as e:
            logger.error(f"Error processing worker queue: {e}")
    
    async def _stale_assignment_cleanup_loop(self) -> None:
        """Background task to cleanup stale worker assignments."""
        logger.info(f"Starting stale assignment cleanup loop with {self._cleanup_interval}s interval")
        
        while self._is_running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                
                if not self._is_running:
                    break
                
                await self._cleanup_stale_assignments()
                
            except asyncio.CancelledError:
                logger.debug("Stale assignment cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in stale assignment cleanup loop: {e}", exc_info=True)
                # Continue running even if cleanup fails
    
    async def _cleanup_stale_assignments(self) -> None:
        """Clean up stale worker assignments."""
        current_time = datetime.now()
        stale_assignments = []
        
        # Find stale assignments
        all_assignments = self.assignment_tracker.get_all_assignments()
        for assignment in all_assignments:
            assignment_age = (current_time - assignment.assigned_at).total_seconds()
            if assignment_age > self.stale_assignment_threshold:
                stale_assignments.append(assignment)
        
        if not stale_assignments:
            return
        
        logger.warning(f"Found {len(stale_assignments)} stale assignments to cleanup")
        
        # Cleanup stale assignments
        async with self._allocation_lock:
            for assignment in stale_assignments:
                try:
                    logger.warning(
                        f"STALE_ASSIGNMENT_CLEANUP: Cleaning up stale assignment - "
                        f"Worker {assignment.worker_id} on GPU {assignment.gpu_id} "
                        f"(age: {(current_time - assignment.assigned_at).total_seconds():.1f}s)"
                    )
                    
                    # Remove from assignment tracker
                    self.assignment_tracker.release(assignment.worker_id)
                    
                    # Remove from resource state
                    self.resource_state.remove_assignment(assignment.gpu_id, assignment.worker_id)
                    
                    logger.info(f"Cleaned up stale assignment for worker {assignment.worker_id}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning up stale assignment {assignment.worker_id}: {e}")
            
            # Process worker queue after cleanup to assign newly available GPUs
            if stale_assignments:
                await self._process_worker_queue()
    
    def get_stale_assignment_status(self) -> Dict[str, Any]:
        """Get status of stale assignment cleanup for monitoring."""
        current_time = datetime.now()
        all_assignments = self.assignment_tracker.get_all_assignments()
        
        assignment_ages = []
        stale_count = 0
        
        for assignment in all_assignments:
            age_seconds = (current_time - assignment.assigned_at).total_seconds()
            assignment_ages.append({
                "worker_id": assignment.worker_id,
                "gpu_id": assignment.gpu_id,
                "age_seconds": age_seconds,
                "is_stale": age_seconds > self.stale_assignment_threshold
            })
            
            if age_seconds > self.stale_assignment_threshold:
                stale_count += 1
        
        return {
            "stale_assignment_threshold": self.stale_assignment_threshold,
            "cleanup_interval": self._cleanup_interval,
            "total_assignments": len(all_assignments),
            "stale_assignments": stale_count,
            "assignment_ages": assignment_ages,
            "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done()
        }

    async def _cleanup_on_start_failure(self) -> None:
        """Cleanup resources when start fails."""
        logger.debug("Performing cleanup after start failure")
        
        try:
            # Ensure we're not marked as running
            self._is_running = False
            self._start_time = None
            
            # Try to stop GPU monitor if it was started
            try:
                await self.gpu_monitor.stop()
                logger.debug("GPU monitor stopped during cleanup")
            except Exception as e:
                logger.debug(f"Error stopping GPU monitor during cleanup: {e}")
            
            # Clear any state that might have been initialized
            try:
                self.resource_state.clear_all()
                self.assignment_tracker.clear_all()
                self.worker_queue.clear()
                logger.debug("State cleared during cleanup")
            except Exception as e:
                logger.debug(f"Error clearing state during cleanup: {e}")
                
        except Exception as e:
            logger.error(f"Error during start failure cleanup: {e}")
    
    def get_lifecycle_status(self) -> Dict[str, Any]:
        """
        Get current lifecycle status for monitoring and debugging.
        
        Returns:
            Dictionary with lifecycle status information
        """
        status = {
            "is_running": self._is_running,
            "is_starting": self._is_starting,
            "is_stopping": self._is_stopping,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "stop_time": self._stop_time.isoformat() if self._stop_time else None,
            "uptime_seconds": None
        }
        
        # Calculate uptime if running
        if self._is_running and self._start_time:
            status["uptime_seconds"] = (datetime.now() - self._start_time).total_seconds()
        elif self._stop_time and self._start_time:
            status["uptime_seconds"] = (self._stop_time - self._start_time).total_seconds()
        
        return status
    
    def _on_gpu_stats_update(self, stats: GPUStats) -> None:
        """
        Handle GPU statistics updates.
        
        This callback is called whenever new GPU statistics are received
        from the monitoring system.
        
        Args:
            stats: Updated GPU statistics
        """
        logger.debug(f"Received GPU stats update: {stats.gpu_count} GPUs")
        
        # Update resource state
        self.resource_state.update_stats(stats)
        
        # Process worker queue in case new GPUs became available
        # Use asyncio.create_task to avoid blocking the callback
        if self._is_running:
            asyncio.create_task(self._safe_process_worker_queue())
    
    def _register_health_checks(self) -> None:
        """Register health checks for the worker pool manager."""
        if not self.health_checker:
            return
        
        # Register pool manager health check
        self.health_checker.register_health_check("pool_manager", self._pool_manager_health_check)
        
        # Register GPU monitor health check
        self.health_checker.register_health_check("gpu_monitor", self._gpu_monitor_health_check)
        
        # Register resource state health check
        self.health_checker.register_health_check("resource_state", self._resource_state_health_check)
    
    def _pool_manager_health_check(self) -> HealthCheckResult:
        """Health check for the pool manager itself."""
        if not self._is_running:
            return HealthCheckResult(
                name="pool_manager",
                status="unhealthy",
                message="Pool manager is not running",
                timestamp=datetime.now(),
                details={"is_running": False}
            )
        
        # Check for excessive stale assignments
        stale_status = self.get_stale_assignment_status()
        stale_count = stale_status["stale_assignments"]
        total_assignments = stale_status["total_assignments"]
        
        if total_assignments > 0 and stale_count / total_assignments > 0.5:
            return HealthCheckResult(
                name="pool_manager",
                status="degraded",
                message=f"High number of stale assignments: {stale_count}/{total_assignments}",
                timestamp=datetime.now(),
                details=stale_status
            )
        
        # Check for excessive queue size
        queue_size = self.worker_queue.size()
        if queue_size > 100:  # Arbitrary threshold
            return HealthCheckResult(
                name="pool_manager",
                status="degraded",
                message=f"Large worker queue: {queue_size} workers blocked",
                timestamp=datetime.now(),
                details={"queue_size": queue_size}
            )
        
        return HealthCheckResult(
            name="pool_manager",
            status="healthy",
            message="Pool manager is running normally",
            timestamp=datetime.now(),
            details={
                "is_running": self._is_running,
                "queue_size": queue_size,
                "stale_assignments": stale_count,
                "total_assignments": total_assignments
            }
        )
    
    def _gpu_monitor_health_check(self) -> HealthCheckResult:
        """Health check for the GPU monitor."""
        current_stats = self.resource_state.get_current_stats()
        
        if current_stats is None:
            return HealthCheckResult(
                name="gpu_monitor",
                status="unhealthy",
                message="No GPU statistics available",
                timestamp=datetime.now(),
                details={"has_stats": False}
            )
        
        # Check if stats are recent (within last 30 seconds)
        if hasattr(self.gpu_monitor, 'get_monitor_status'):
            monitor_status = self.gpu_monitor.get_monitor_status()
            consecutive_failures = monitor_status.get("consecutive_failures", 0)
            
            if consecutive_failures > 5:
                return HealthCheckResult(
                    name="gpu_monitor",
                    status="degraded",
                    message=f"GPU monitor has {consecutive_failures} consecutive failures",
                    timestamp=datetime.now(),
                    details=monitor_status
                )
        
        return HealthCheckResult(
            name="gpu_monitor",
            status="healthy",
            message="GPU monitor is functioning normally",
            timestamp=datetime.now(),
            details={
                "has_stats": True,
                "gpu_count": current_stats.gpu_count,
                "timestamp": current_stats.timestamp
            }
        )
    
    def _resource_state_health_check(self) -> HealthCheckResult:
        """Health check for resource state consistency."""
        try:
            # Check for consistency between assignment tracker and resource state
            tracker_assignments = self.assignment_tracker.get_all_assignments()
            resource_assignments = self.resource_state.get_all_assignments()
            
            # Count total assignments from both sources
            tracker_count = len(tracker_assignments)
            resource_count = sum(len(workers) for workers in resource_assignments.values())
            
            if tracker_count != resource_count:
                return HealthCheckResult(
                    name="resource_state",
                    status="degraded",
                    message=f"Assignment count mismatch: tracker={tracker_count}, resource={resource_count}",
                    timestamp=datetime.now(),
                    details={
                        "tracker_assignments": tracker_count,
                        "resource_assignments": resource_count
                    }
                )
            
            return HealthCheckResult(
                name="resource_state",
                status="healthy",
                message="Resource state is consistent",
                timestamp=datetime.now(),
                details={
                    "total_assignments": tracker_count,
                    "consistency_check": "passed"
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="resource_state",
                status="unhealthy",
                message=f"Error checking resource state: {e}",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
    
    async def get_health_status(self) -> Dict[str, HealthCheckResult]:
        """Get health status for all components.
        
        Returns:
            Dictionary of health check results
        """
        if not self.health_checker:
            return {}
        
        return await self.health_checker.run_health_checks()
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring metrics.
        
        Returns:
            Dictionary with all monitoring data
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "pool_metrics": self.get_detailed_pool_metrics(),
            "lifecycle_status": self.get_lifecycle_status(),
            "stale_assignment_status": self.get_stale_assignment_status()
        }
        
        # Add performance metrics if available
        if self.performance_monitor:
            metrics["performance_summary"] = self.performance_monitor.get_performance_summary()
        
        # Add collected metrics if available
        if self.metrics_collector:
            if hasattr(self.metrics_collector, 'get_metric_summary'):
                metrics["metric_summary"] = self.metrics_collector.get_metric_summary()
            else:
                metrics["raw_metrics"] = self.metrics_collector.get_metrics()
        
        return metrics

    async def _safe_process_worker_queue(self) -> None:
        """Safely process worker queue with error handling."""
        try:
            async with self._allocation_lock:
                await self._process_worker_queue()
        except Exception as e:
            logger.error(f"Error processing worker queue: {e}")
    
    async def _stale_assignment_cleanup_loop(self) -> None:
        """Background task to cleanup stale worker assignments."""
        logger.info(f"Starting stale assignment cleanup loop with {self._cleanup_interval}s interval")
        
        while self._is_running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                
                if not self._is_running:
                    break
                
                await self._cleanup_stale_assignments()
                
            except asyncio.CancelledError:
                logger.debug("Stale assignment cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in stale assignment cleanup loop: {e}", exc_info=True)
                # Continue running even if cleanup fails
    
    async def _cleanup_stale_assignments(self) -> None:
        """Clean up stale worker assignments."""
        current_time = datetime.now()
        stale_assignments = []
        
        # Find stale assignments
        all_assignments = self.assignment_tracker.get_all_assignments()
        for assignment in all_assignments:
            assignment_age = (current_time - assignment.assigned_at).total_seconds()
            if assignment_age > self.stale_assignment_threshold:
                stale_assignments.append(assignment)
        
        if not stale_assignments:
            return
        
        logger.warning(f"Found {len(stale_assignments)} stale assignments to cleanup")
        
        # Cleanup stale assignments
        async with self._allocation_lock:
            for assignment in stale_assignments:
                try:
                    logger.warning(
                        f"STALE_ASSIGNMENT_CLEANUP: Cleaning up stale assignment - "
                        f"Worker {assignment.worker_id} on GPU {assignment.gpu_id} "
                        f"(age: {(current_time - assignment.assigned_at).total_seconds():.1f}s)"
                    )
                    
                    # Remove from assignment tracker
                    self.assignment_tracker.release(assignment.worker_id)
                    
                    # Remove from resource state
                    self.resource_state.remove_assignment(assignment.gpu_id, assignment.worker_id)
                    
                    logger.info(f"Cleaned up stale assignment for worker {assignment.worker_id}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning up stale assignment {assignment.worker_id}: {e}")
            
            # Process worker queue after cleanup to assign newly available GPUs
            if stale_assignments:
                await self._process_worker_queue()
    
    def get_stale_assignment_status(self) -> Dict[str, Any]:
        """Get status of stale assignment cleanup for monitoring."""
        current_time = datetime.now()
        all_assignments = self.assignment_tracker.get_all_assignments()
        
        assignment_ages = []
        stale_count = 0
        
        for assignment in all_assignments:
            age_seconds = (current_time - assignment.assigned_at).total_seconds()
            assignment_ages.append({
                "worker_id": assignment.worker_id,
                "gpu_id": assignment.gpu_id,
                "age_seconds": age_seconds,
                "is_stale": age_seconds > self.stale_assignment_threshold
            })
            
            if age_seconds > self.stale_assignment_threshold:
                stale_count += 1
        
        return {
            "stale_assignment_threshold": self.stale_assignment_threshold,
            "cleanup_interval": self._cleanup_interval,
            "total_assignments": len(all_assignments),
            "stale_assignments": stale_count,
            "assignment_ages": assignment_ages,
            "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done()
        }

    async def _cleanup_on_start_failure(self) -> None:
        """Cleanup resources when start fails."""
        logger.debug("Performing cleanup after start failure")
        
        try:
            # Ensure we're not marked as running
            self._is_running = False
            self._start_time = None
            
            # Try to stop GPU monitor if it was started
            try:
                await self.gpu_monitor.stop()
                logger.debug("GPU monitor stopped during cleanup")
            except Exception as e:
                logger.debug(f"Error stopping GPU monitor during cleanup: {e}")
            
            # Clear any state that might have been initialized
            try:
                self.resource_state.clear_all()
                self.assignment_tracker.clear_all()
                self.worker_queue.clear()
                logger.debug("State cleared during cleanup")
            except Exception as e:
                logger.debug(f"Error clearing state during cleanup: {e}")
                
        except Exception as e:
            logger.error(f"Error during start failure cleanup: {e}")
    
    def get_lifecycle_status(self) -> Dict[str, Any]:
        """
        Get current lifecycle status for monitoring and debugging.
        
        Returns:
            Dictionary with lifecycle status information
        """
        status = {
            "is_running": self._is_running,
            "is_starting": self._is_starting,
            "is_stopping": self._is_stopping,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "stop_time": self._stop_time.isoformat() if self._stop_time else None,
            "uptime_seconds": None
        }
        
        # Calculate uptime if running
        if self._is_running and self._start_time:
            status["uptime_seconds"] = (datetime.now() - self._start_time).total_seconds()
        elif self._stop_time and self._start_time:
            status["uptime_seconds"] = (self._stop_time - self._start_time).total_seconds()
        
        return status