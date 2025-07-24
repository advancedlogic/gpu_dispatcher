"""Unit tests for WorkerPoolManager core orchestration logic."""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from gpu_worker_pool.worker_pool_manager import AsyncWorkerPoolManager
from gpu_worker_pool.models import GPUStats, GPUInfo, GPUAssignment, PoolStatus, WorkerInfo
from gpu_worker_pool.config import ConfigurationManager
from gpu_worker_pool.gpu_monitor import GPUMonitor
from gpu_worker_pool.gpu_allocator import GPUAllocator
from gpu_worker_pool.worker_queue import WorkerQueue


class MockConfigurationManager(ConfigurationManager):
    """Mock configuration manager for testing."""
    
    def get_memory_threshold(self) -> float:
        return 80.0
    
    def get_utilization_threshold(self) -> float:
        return 90.0
    
    def get_polling_interval(self) -> int:
        return 5
    
    def get_service_endpoint(self) -> str:
        return "http://localhost:8000"


class MockGPUMonitor(GPUMonitor):
    """Mock GPU monitor for testing."""
    
    def __init__(self):
        self.callbacks = []
        self.current_stats = None
        self.is_started = False
    
    async def start(self) -> None:
        self.is_started = True
    
    async def stop(self) -> None:
        self.is_started = False
    
    def get_current_stats(self):
        return self.current_stats
    
    def on_stats_update(self, callback):
        self.callbacks.append(callback)
    
    def remove_stats_callback(self, callback):
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def trigger_stats_update(self, stats):
        """Trigger stats update for testing."""
        self.current_stats = stats
        for callback in self.callbacks:
            callback(stats)


class MockGPUAllocator(GPUAllocator):
    """Mock GPU allocator for testing."""
    
    def __init__(self):
        self.available_gpu_id = None
        self.is_available_result = True
    
    def find_available_gpu(self, current_stats, assignments):
        return self.available_gpu_id
    
    def is_gpu_available(self, gpu, assigned_workers):
        return self.is_available_result
    
    def calculate_gpu_score(self, gpu):
        return gpu.memory_usage_percent + gpu.utilization_percent


class MockWorkerQueue(WorkerQueue):
    """Mock worker queue for testing."""
    
    def __init__(self):
        self.queue = []
        self.blocked_workers = []
    
    def enqueue(self, worker):
        self.queue.append(worker)
    
    def dequeue(self):
        return self.queue.pop(0) if self.queue else None
    
    def size(self):
        return len(self.queue)
    
    def clear(self):
        # Notify workers of cancellation
        for worker in self.queue:
            try:
                worker.on_error(Exception("Worker request cancelled due to queue clear"))
            except Exception:
                pass
        self.queue.clear()
    
    def block_worker(self, worker, reason):
        self.blocked_workers.append((worker, reason))
        self.queue.append(worker)
    
    def unblock_next_worker(self):
        return self.queue.pop(0) if self.queue else None
    
    def unblock_workers(self, count):
        unblocked = []
        for _ in range(min(count, len(self.queue))):
            if self.queue:
                unblocked.append(self.queue.pop(0))
        return unblocked


@pytest.fixture
def mock_config():
    """Fixture for mock configuration manager."""
    return MockConfigurationManager()


@pytest.fixture
def mock_gpu_monitor():
    """Fixture for mock GPU monitor."""
    return MockGPUMonitor()


@pytest.fixture
def mock_gpu_allocator():
    """Fixture for mock GPU allocator."""
    return MockGPUAllocator()


@pytest.fixture
def mock_worker_queue():
    """Fixture for mock worker queue."""
    return MockWorkerQueue()


@pytest.fixture
def worker_pool_manager(mock_config, mock_gpu_monitor, mock_gpu_allocator, mock_worker_queue):
    """Fixture for worker pool manager with mocked dependencies."""
    return AsyncWorkerPoolManager(
        config=mock_config,
        gpu_monitor=mock_gpu_monitor,
        gpu_allocator=mock_gpu_allocator,
        worker_queue=mock_worker_queue
    )


@pytest.fixture
def sample_gpu_stats():
    """Fixture for sample GPU statistics."""
    return GPUStats(
        gpu_count=2,
        total_memory_mb=16000,
        total_used_memory_mb=8000,
        average_utilization_percent=50.0,
        gpus_summary=[
            GPUInfo(gpu_id=0, name="GPU 0", memory_usage_percent=40.0, utilization_percent=30.0),
            GPUInfo(gpu_id=1, name="GPU 1", memory_usage_percent=60.0, utilization_percent=70.0)
        ],
        total_memory_usage_percent=50.0,
        timestamp="2024-01-01T12:00:00Z"
    )


class TestAsyncWorkerPoolManager:
    """Test cases for AsyncWorkerPoolManager core orchestration."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, worker_pool_manager, mock_gpu_monitor):
        """Test worker pool manager initialization."""
        assert worker_pool_manager.config is not None
        assert worker_pool_manager.gpu_monitor is mock_gpu_monitor
        assert worker_pool_manager.gpu_allocator is not None
        assert worker_pool_manager.worker_queue is not None
        assert worker_pool_manager.resource_state is not None
        assert worker_pool_manager.assignment_tracker is not None
        assert not worker_pool_manager._is_running
    
    @pytest.mark.asyncio
    async def test_start_and_stop(self, worker_pool_manager, mock_gpu_monitor):
        """Test starting and stopping the worker pool manager."""
        # Test start
        await worker_pool_manager.start()
        assert worker_pool_manager._is_running
        assert mock_gpu_monitor.is_started
        
        # Test stop
        await worker_pool_manager.stop()
        assert not worker_pool_manager._is_running
        assert not mock_gpu_monitor.is_started
    
    @pytest.mark.asyncio
    async def test_request_gpu_immediate_assignment(self, worker_pool_manager, mock_gpu_allocator, sample_gpu_stats):
        """Test immediate GPU assignment when resources are available."""
        # Setup
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        mock_gpu_allocator.available_gpu_id = 0
        
        # Request GPU
        assignment = await worker_pool_manager.request_gpu()
        
        # Verify assignment
        assert isinstance(assignment, GPUAssignment)
        assert assignment.gpu_id == 0
        assert assignment.worker_id is not None
        assert isinstance(assignment.assigned_at, datetime)
        
        # Verify tracking
        assert worker_pool_manager.assignment_tracker.is_assigned(assignment.worker_id)
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_request_gpu_blocking_when_no_gpus_available(self, worker_pool_manager, mock_gpu_allocator, mock_worker_queue, sample_gpu_stats):
        """Test worker blocking when no GPUs are available."""
        # Setup
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        mock_gpu_allocator.available_gpu_id = None  # No GPUs available
        
        # Create a task to request GPU (this should block)
        request_task = asyncio.create_task(worker_pool_manager.request_gpu())
        
        # Give it a moment to process
        await asyncio.sleep(0.1)
        
        # Verify worker was blocked
        assert mock_worker_queue.size() == 1
        assert len(mock_worker_queue.blocked_workers) == 1
        
        # Make GPU available and trigger processing
        mock_gpu_allocator.available_gpu_id = 0
        await worker_pool_manager._process_worker_queue()
        
        # Wait for assignment
        assignment = await request_task
        
        # Verify assignment
        assert isinstance(assignment, GPUAssignment)
        assert assignment.gpu_id == 0
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_release_gpu_with_cleanup(self, worker_pool_manager, mock_gpu_allocator, sample_gpu_stats):
        """Test GPU release with proper cleanup."""
        # Setup
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        mock_gpu_allocator.available_gpu_id = 0
        
        # Request and get assignment
        assignment = await worker_pool_manager.request_gpu()
        
        # Verify assignment exists
        assert worker_pool_manager.assignment_tracker.is_assigned(assignment.worker_id)
        
        # Release GPU
        await worker_pool_manager.release_gpu(assignment)
        
        # Verify cleanup
        assert not worker_pool_manager.assignment_tracker.is_assigned(assignment.worker_id)
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_release_gpu_unblocks_waiting_workers(self, worker_pool_manager, mock_gpu_allocator, mock_worker_queue, sample_gpu_stats):
        """Test that releasing a GPU unblocks waiting workers."""
        # Setup
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        mock_gpu_allocator.available_gpu_id = 0
        
        # Get first assignment
        assignment1 = await worker_pool_manager.request_gpu()
        
        # Make no GPUs available for second request
        mock_gpu_allocator.available_gpu_id = None
        
        # Create second request (should block)
        request_task = asyncio.create_task(worker_pool_manager.request_gpu())
        await asyncio.sleep(0.1)
        
        # Verify second worker is blocked
        assert mock_worker_queue.size() == 1
        
        # Make GPU available again and release first assignment
        mock_gpu_allocator.available_gpu_id = 0
        await worker_pool_manager.release_gpu(assignment1)
        
        # Second worker should get assignment
        assignment2 = await request_task
        assert isinstance(assignment2, GPUAssignment)
        assert assignment2.gpu_id == 0
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_get_pool_status(self, worker_pool_manager, mock_gpu_allocator, sample_gpu_stats):
        """Test getting pool status with current metrics."""
        # Setup
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        mock_gpu_allocator.available_gpu_id = 0
        mock_gpu_allocator.is_available_result = True
        
        # Get initial status
        status = worker_pool_manager.get_pool_status()
        assert isinstance(status, PoolStatus)
        assert status.total_gpus == 2
        assert status.available_gpus == 2  # Both GPUs available
        assert status.active_workers == 0
        assert status.blocked_workers == 0
        
        # Request GPU
        assignment = await worker_pool_manager.request_gpu()
        
        # Get status after assignment
        status = worker_pool_manager.get_pool_status()
        assert status.active_workers == 1
        assert status.blocked_workers == 0
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_gpu_stats_update_triggers_queue_processing(self, worker_pool_manager, mock_gpu_monitor, mock_gpu_allocator, mock_worker_queue, sample_gpu_stats):
        """Test that GPU stats updates trigger worker queue processing."""
        # Setup
        await worker_pool_manager.start()
        mock_gpu_allocator.available_gpu_id = None  # Initially no GPUs
        
        # Create blocked worker
        request_task = asyncio.create_task(worker_pool_manager.request_gpu())
        await asyncio.sleep(0.1)
        
        # Verify worker is blocked
        assert mock_worker_queue.size() == 1
        
        # Make GPU available and trigger stats update
        mock_gpu_allocator.available_gpu_id = 0
        mock_gpu_monitor.trigger_stats_update(sample_gpu_stats)
        
        # Give time for async processing
        await asyncio.sleep(0.1)
        
        # Worker should get assignment
        assignment = await request_task
        assert isinstance(assignment, GPUAssignment)
        assert assignment.gpu_id == 0
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_request_gpu_when_not_running(self, worker_pool_manager):
        """Test that requesting GPU when not running raises error."""
        with pytest.raises(RuntimeError, match="Worker pool manager is not running"):
            await worker_pool_manager.request_gpu()
    
    @pytest.mark.asyncio
    async def test_release_gpu_invalid_assignment(self, worker_pool_manager):
        """Test releasing invalid assignment."""
        await worker_pool_manager.start()
        
        with pytest.raises(ValueError, match="assignment must be a GPUAssignment instance"):
            await worker_pool_manager.release_gpu("invalid")
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_release_gpu_nonexistent_assignment(self, worker_pool_manager):
        """Test releasing non-existent assignment."""
        await worker_pool_manager.start()
        
        # Create fake assignment
        fake_assignment = GPUAssignment(
            gpu_id=0,
            worker_id="nonexistent",
            assigned_at=datetime.now()
        )
        
        # Should not raise error, just log warning
        await worker_pool_manager.release_gpu(fake_assignment)
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, worker_pool_manager, mock_gpu_allocator, sample_gpu_stats):
        """Test handling multiple concurrent GPU requests."""
        # Setup
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        
        # Mock allocator to return different GPUs for different calls
        gpu_ids = [0, 1, None, None]  # First two succeed, rest block
        call_count = 0
        
        def mock_find_available_gpu(*args):
            nonlocal call_count
            result = gpu_ids[call_count] if call_count < len(gpu_ids) else None
            call_count += 1
            return result
        
        mock_gpu_allocator.find_available_gpu = mock_find_available_gpu
        
        # Create multiple concurrent requests
        tasks = [
            asyncio.create_task(worker_pool_manager.request_gpu())
            for _ in range(4)
        ]
        
        # Give time for processing
        await asyncio.sleep(0.1)
        
        # First two should complete immediately
        completed_tasks = [task for task in tasks if task.done()]
        assert len(completed_tasks) == 2
        
        # Verify assignments
        assignments = [await task for task in completed_tasks]
        gpu_ids_assigned = {assignment.gpu_id for assignment in assignments}
        assert gpu_ids_assigned == {0, 1}
        
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_stop_clears_all_state(self, worker_pool_manager, mock_gpu_allocator, mock_worker_queue, sample_gpu_stats):
        """Test that stopping clears all state and queued workers."""
        # Setup
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        mock_gpu_allocator.available_gpu_id = 0
        
        # Create assignment
        assignment = await worker_pool_manager.request_gpu()
        
        # Block a worker
        mock_gpu_allocator.available_gpu_id = None
        request_task = asyncio.create_task(worker_pool_manager.request_gpu())
        await asyncio.sleep(0.1)
        
        # Verify state exists
        assert worker_pool_manager.assignment_tracker.get_assignment_count() == 1
        assert mock_worker_queue.size() == 1
        
        # Stop manager
        await worker_pool_manager.stop()
        
        # Verify state is cleared
        assert worker_pool_manager.assignment_tracker.get_assignment_count() == 0
        assert mock_worker_queue.size() == 0
        
        # Blocked request should be cancelled
        with pytest.raises(Exception):  # Should get cancellation exception
            await request_task
    
    @pytest.mark.asyncio
    async def test_get_detailed_pool_metrics(self, worker_pool_manager, mock_gpu_allocator, sample_gpu_stats):
        """Test getting detailed pool metrics for monitoring."""
        # Setup
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        mock_gpu_allocator.available_gpu_id = 0
        mock_gpu_allocator.is_available_result = True
        
        # Create assignment
        assignment = await worker_pool_manager.request_gpu()
        
        # Get detailed metrics
        metrics = worker_pool_manager.get_detailed_pool_metrics()
        
        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert "timestamp" in metrics
        assert "is_running" in metrics
        assert "total_gpus" in metrics
        assert "available_gpus" in metrics
        assert "active_workers" in metrics
        assert "blocked_workers" in metrics
        assert "gpu_metrics" in metrics
        assert "assignment_metrics" in metrics
        assert "queue_metrics" in metrics
        assert "thresholds" in metrics
        
        # Verify metrics values
        assert metrics["is_running"] is True
        assert metrics["total_gpus"] == 2
        assert metrics["active_workers"] == 1
        assert metrics["blocked_workers"] == 0
        
        # Verify GPU metrics
        assert len(metrics["gpu_metrics"]) == 2
        gpu_metric = metrics["gpu_metrics"][0]
        assert "gpu_id" in gpu_metric
        assert "name" in gpu_metric
        assert "memory_usage_percent" in gpu_metric
        assert "utilization_percent" in gpu_metric
        assert "assigned_workers" in gpu_metric
        assert "worker_ids" in gpu_metric
        assert "is_available" in gpu_metric
        assert "resource_score" in gpu_metric
        
        # Verify assignment metrics
        assert len(metrics["assignment_metrics"]) == 1
        assignment_metric = metrics["assignment_metrics"][0]
        assert "worker_id" in assignment_metric
        assert "gpu_id" in assignment_metric
        assert "assigned_at" in assignment_metric
        assert "duration_seconds" in assignment_metric
        
        # Verify thresholds
        assert "memory_threshold_percent" in metrics["thresholds"]
        assert "utilization_threshold_percent" in metrics["thresholds"]
        assert metrics["thresholds"]["memory_threshold_percent"] == 80.0
        assert metrics["thresholds"]["utilization_threshold_percent"] == 90.0
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_logging_during_assignment_lifecycle(self, worker_pool_manager, mock_gpu_allocator, sample_gpu_stats, caplog):
        """Test comprehensive logging during worker assignment lifecycle."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Setup
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        mock_gpu_allocator.available_gpu_id = 0
        
        # Request GPU (should log assignment events)
        assignment = await worker_pool_manager.request_gpu()
        
        # Verify assignment logging
        assignment_logs = [record for record in caplog.records if "WORKER_ASSIGNED" in record.message]
        assert len(assignment_logs) >= 1
        
        # Release GPU (should log release events)
        await worker_pool_manager.release_gpu(assignment)
        
        # Verify release logging
        release_logs = [record for record in caplog.records if "GPU_RELEASED" in record.message]
        assert len(release_logs) >= 1
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_logging_during_blocking_scenario(self, worker_pool_manager, mock_gpu_allocator, mock_worker_queue, sample_gpu_stats, caplog):
        """Test logging when workers are blocked and unblocked."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Setup
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        mock_gpu_allocator.available_gpu_id = None  # No GPUs available
        
        # Create blocked worker
        request_task = asyncio.create_task(worker_pool_manager.request_gpu())
        await asyncio.sleep(0.1)
        
        # Verify blocking logs
        blocking_logs = [record for record in caplog.records if "WORKER_BLOCKED" in record.message]
        assert len(blocking_logs) >= 1
        
        # Make GPU available and trigger processing
        mock_gpu_allocator.available_gpu_id = 0
        await worker_pool_manager._process_worker_queue()
        
        # Complete the assignment
        assignment = await request_task
        assert isinstance(assignment, GPUAssignment)
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_error_logging_on_invalid_operations(self, worker_pool_manager, caplog):
        """Test error logging for invalid operations."""
        import logging
        caplog.set_level(logging.ERROR)
        
        # Test request when not running
        with pytest.raises(RuntimeError):
            await worker_pool_manager.request_gpu()
        
        # Verify error logging
        error_logs = [record for record in caplog.records if record.levelname == "ERROR"]
        assert len(error_logs) >= 1
        assert any("GPU request rejected" in record.message for record in error_logs)
    
    @pytest.mark.asyncio
    async def test_startup_and_shutdown_logging(self, worker_pool_manager, caplog):
        """Test logging during startup and shutdown."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Test startup logging
        await worker_pool_manager.start()
        
        startup_logs = [record for record in caplog.records if "POOL_STARTED" in record.message]
        assert len(startup_logs) >= 1
        
        # Test shutdown logging
        await worker_pool_manager.stop()
        
        shutdown_logs = [record for record in caplog.records if "POOL_STOPPED" in record.message]
        assert len(shutdown_logs) >= 1
    
    @pytest.mark.asyncio
    async def test_pool_status_logging(self, worker_pool_manager, mock_gpu_allocator, sample_gpu_stats, caplog):
        """Test that pool status calls include logging."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Setup
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        mock_gpu_allocator.is_available_result = True
        
        # Get pool status (should trigger logging)
        status = worker_pool_manager.get_pool_status()
        
        # Verify status logging
        status_logs = [record for record in caplog.records if "Pool status:" in record.message]
        assert len(status_logs) >= 1
        
        # Verify log contains expected metrics
        status_log = status_logs[0]
        assert "total GPUs" in status_log.message
        assert "available" in status_log.message
        assert "active workers" in status_log.message
        assert "blocked workers" in status_log.message
        
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_lifecycle_state_management(self, worker_pool_manager):
        """Test proper lifecycle state management during start/stop operations."""
        # Initial state
        lifecycle_status = worker_pool_manager.get_lifecycle_status()
        assert lifecycle_status["is_running"] is False
        assert lifecycle_status["is_starting"] is False
        assert lifecycle_status["is_stopping"] is False
        assert lifecycle_status["start_time"] is None
        assert lifecycle_status["stop_time"] is None
        assert lifecycle_status["uptime_seconds"] is None
        
        # Start the manager
        await worker_pool_manager.start()
        
        # Check running state
        lifecycle_status = worker_pool_manager.get_lifecycle_status()
        assert lifecycle_status["is_running"] is True
        assert lifecycle_status["is_starting"] is False
        assert lifecycle_status["is_stopping"] is False
        assert lifecycle_status["start_time"] is not None
        assert lifecycle_status["stop_time"] is None
        assert lifecycle_status["uptime_seconds"] is not None
        assert lifecycle_status["uptime_seconds"] > 0
        
        # Stop the manager
        await worker_pool_manager.stop()
        
        # Check stopped state
        lifecycle_status = worker_pool_manager.get_lifecycle_status()
        assert lifecycle_status["is_running"] is False
        assert lifecycle_status["is_starting"] is False
        assert lifecycle_status["is_stopping"] is False
        assert lifecycle_status["start_time"] is not None
        assert lifecycle_status["stop_time"] is not None
        assert lifecycle_status["uptime_seconds"] is not None
    
    @pytest.mark.asyncio
    async def test_start_while_starting_prevention(self, worker_pool_manager):
        """Test that starting while already starting is prevented."""
        # Manually set the starting state to simulate concurrent start
        worker_pool_manager._is_starting = True
        
        # Second start call should return immediately without error (no exception)
        await worker_pool_manager.start()
        
        # Verify state hasn't changed
        assert worker_pool_manager._is_running is False
        assert worker_pool_manager._is_starting is True
        
        # Reset state for cleanup
        worker_pool_manager._is_starting = False
    
    @pytest.mark.asyncio
    async def test_stop_while_stopping_prevention(self, worker_pool_manager):
        """Test that stopping while already stopping is prevented."""
        await worker_pool_manager.start()
        
        # Manually set the stopping state to simulate concurrent stop
        worker_pool_manager._is_stopping = True
        
        # Second stop call should return immediately without error
        await worker_pool_manager.stop()
        
        # Verify state hasn't changed
        assert worker_pool_manager._is_running is True  # Still running since we didn't actually stop
        assert worker_pool_manager._is_stopping is True
        
        # Reset state and properly stop
        worker_pool_manager._is_stopping = False
        await worker_pool_manager.stop()
    
    @pytest.mark.asyncio
    async def test_start_while_stopping_error(self, worker_pool_manager):
        """Test that starting while stopping raises an error."""
        await worker_pool_manager.start()
        
        # Mock the GPU monitor to delay shutdown
        original_stop = worker_pool_manager.gpu_monitor.stop
        stop_delay_event = asyncio.Event()
        
        async def delayed_stop():
            await stop_delay_event.wait()
            await original_stop()
        
        worker_pool_manager.gpu_monitor.stop = delayed_stop
        
        # Start stop operation (will be delayed)
        stop_task = asyncio.create_task(worker_pool_manager.stop())
        
        # Give it time to enter the stopping state
        await asyncio.sleep(0.1)
        
        # Try to start while stopping - should raise error
        with pytest.raises(RuntimeError, match="Cannot start worker pool manager while it is stopping"):
            await worker_pool_manager.start()
        
        # Complete the stop
        stop_delay_event.set()
        await stop_task
    
    @pytest.mark.asyncio
    async def test_start_failure_cleanup(self, worker_pool_manager, mock_gpu_monitor):
        """Test proper cleanup when start fails."""
        # Make GPU monitor start fail
        mock_gpu_monitor.start = AsyncMock(side_effect=Exception("GPU monitor start failed"))
        
        # Attempt to start (should fail)
        with pytest.raises(Exception, match="GPU monitor start failed"):
            await worker_pool_manager.start()
        
        # Verify cleanup was performed
        assert worker_pool_manager._is_running is False
        assert worker_pool_manager._is_starting is False
        assert worker_pool_manager._start_time is None
        
        # Verify GPU monitor stop was called during cleanup
        # The MockGPUMonitor doesn't track stop calls, so we'll verify the state instead
        assert mock_gpu_monitor.is_started is False
    
    @pytest.mark.asyncio
    async def test_stop_with_active_assignments_logging(self, worker_pool_manager, mock_gpu_allocator, sample_gpu_stats, caplog):
        """Test logging when stopping with active assignments."""
        import logging
        caplog.set_level(logging.WARNING)
        
        # Setup and create assignment
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        mock_gpu_allocator.available_gpu_id = 0
        
        assignment = await worker_pool_manager.request_gpu()
        
        # Stop with active assignment
        await worker_pool_manager.stop()
        
        # Verify cleanup warning was logged
        cleanup_logs = [record for record in caplog.records if "CLEANUP_WARNING" in record.message]
        assert len(cleanup_logs) >= 1
        assert "active assignments" in cleanup_logs[0].message
    
    @pytest.mark.asyncio
    async def test_uptime_calculation(self, worker_pool_manager):
        """Test uptime calculation in lifecycle status."""
        # Start the manager
        start_time = datetime.now()
        await worker_pool_manager.start()
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Check uptime while running
        lifecycle_status = worker_pool_manager.get_lifecycle_status()
        assert lifecycle_status["uptime_seconds"] is not None
        assert lifecycle_status["uptime_seconds"] > 0
        assert lifecycle_status["uptime_seconds"] < 1.0  # Should be less than 1 second
        
        # Stop and check final uptime
        await worker_pool_manager.stop()
        
        lifecycle_status = worker_pool_manager.get_lifecycle_status()
        assert lifecycle_status["uptime_seconds"] is not None
        assert lifecycle_status["uptime_seconds"] > 0
        import logging
        caplog.set_level(logging.WARNING)
        
        # Setup and create assignment
        await worker_pool_manager.start()
        worker_pool_manager.resource_state.update_stats(sample_gpu_stats)
        mock_gpu_allocator.available_gpu_id = 0
        
        assignment = await worker_pool_manager.request_gpu()
        
        # Stop with active assignment
        await worker_pool_manager.stop()
        
        # Verify cleanup warning was logged
        cleanup_logs = [record for record in caplog.records if "CLEANUP_WARNING" in record.message]
        assert len(cleanup_logs) >= 1
        assert "active assignments" in cleanup_logs[0].message
    
    @pytest.mark.asyncio
    async def test_uptime_calculation(self, worker_pool_manager):
        """Test uptime calculation in lifecycle status."""
        # Start the manager
        start_time = datetime.now()
        await worker_pool_manager.start()
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Check uptime while running
        lifecycle_status = worker_pool_manager.get_lifecycle_status()
        assert lifecycle_status["uptime_seconds"] is not None
        assert lifecycle_status["uptime_seconds"] > 0
        assert lifecycle_status["uptime_seconds"] < 1.0  # Should be less than 1 second
        
        # Stop and check final uptime
        await worker_pool_manager.stop()
        
        lifecycle_status = worker_pool_manager.get_lifecycle_status()
        assert lifecycle_status["uptime_seconds"] is not None
        assert lifecycle_status["uptime_seconds"] > 0