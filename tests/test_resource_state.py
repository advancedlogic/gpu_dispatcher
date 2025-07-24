"""Unit tests for GPU resource state management."""

import pytest
import threading
import time
from datetime import datetime
from unittest.mock import Mock

from gpu_worker_pool.resource_state import GPUResourceState, WorkerAssignmentTracker
from gpu_worker_pool.models import GPUInfo, GPUStats, WorkerInfo, GPUAssignment


class TestGPUResourceState:
    """Test cases for GPUResourceState class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state = GPUResourceState()
        
        # Create sample GPU data
        self.gpu_info_1 = GPUInfo(gpu_id=0, name="GPU 0", memory_usage_percent=50.0, utilization_percent=30.0)
        self.gpu_info_2 = GPUInfo(gpu_id=1, name="GPU 1", memory_usage_percent=70.0, utilization_percent=60.0)
        
        self.gpu_stats = GPUStats(
            gpu_count=2,
            total_memory_mb=16384,
            total_used_memory_mb=8192,
            average_utilization_percent=45.0,
            gpus_summary=[self.gpu_info_1, self.gpu_info_2],
            total_memory_usage_percent=50.0,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        # Create sample worker
        self.worker = WorkerInfo(
            id="worker-1",
            enqueued_at=datetime.now(),
            callback=Mock(),
            on_error=Mock()
        )
    
    def test_initial_state(self):
        """Test initial state of GPUResourceState."""
        assert self.state.get_current_stats() is None
        assert self.state.get_last_update() is None
        assert self.state.get_available_gpus() == []
        assert self.state.get_all_assignments() == {}
        assert self.state.get_total_assignments() == 0
    
    def test_update_stats(self):
        """Test updating GPU statistics."""
        # Update stats
        self.state.update_stats(self.gpu_stats)
        
        # Verify stats are stored
        current_stats = self.state.get_current_stats()
        assert current_stats == self.gpu_stats
        
        # Verify last update timestamp is set
        last_update = self.state.get_last_update()
        assert last_update is not None
        assert isinstance(last_update, datetime)
        
        # Verify available GPUs
        available_gpus = self.state.get_available_gpus()
        assert len(available_gpus) == 2
        assert available_gpus[0] == self.gpu_info_1
        assert available_gpus[1] == self.gpu_info_2
    
    def test_update_stats_cleans_invalid_assignments(self):
        """Test that updating stats cleans up assignments for non-existent GPUs."""
        # Add assignment for GPU that will be removed
        self.state.add_assignment(5, self.worker)
        assert self.state.get_assignment_count(5) == 1
        
        # Update stats with only GPUs 0 and 1
        self.state.update_stats(self.gpu_stats)
        
        # Verify assignment for GPU 5 is removed
        assert self.state.get_assignment_count(5) == 0
        assert 5 not in self.state.get_all_assignments()
    
    def test_add_assignment(self):
        """Test adding worker assignments."""
        # Add assignment
        self.state.add_assignment(0, self.worker)
        
        # Verify assignment is stored
        assignments = self.state.get_assignments(0)
        assert len(assignments) == 1
        assert assignments[0] == self.worker
        
        # Verify assignment count
        assert self.state.get_assignment_count(0) == 1
        assert self.state.get_total_assignments() == 1
    
    def test_add_assignment_invalid_gpu_id(self):
        """Test adding assignment with invalid GPU ID."""
        with pytest.raises(ValueError, match="gpu_id must be a non-negative integer"):
            self.state.add_assignment(-1, self.worker)
        
        with pytest.raises(ValueError, match="gpu_id must be a non-negative integer"):
            self.state.add_assignment("invalid", self.worker)
    
    def test_add_assignment_invalid_worker(self):
        """Test adding assignment with invalid worker."""
        with pytest.raises(ValueError, match="worker must be a WorkerInfo instance"):
            self.state.add_assignment(0, "invalid")
    
    def test_add_duplicate_assignment(self):
        """Test adding duplicate worker assignment to same GPU."""
        # Add initial assignment
        self.state.add_assignment(0, self.worker)
        
        # Try to add same worker again
        with pytest.raises(ValueError, match="Worker worker-1 is already assigned to GPU 0"):
            self.state.add_assignment(0, self.worker)
    
    def test_remove_assignment(self):
        """Test removing worker assignments."""
        # Add assignment first
        self.state.add_assignment(0, self.worker)
        assert self.state.get_assignment_count(0) == 1
        
        # Remove assignment
        result = self.state.remove_assignment(0, "worker-1")
        assert result is True
        
        # Verify assignment is removed
        assert self.state.get_assignment_count(0) == 0
        assert self.state.get_total_assignments() == 0
        assert 0 not in self.state.get_all_assignments()
    
    def test_remove_nonexistent_assignment(self):
        """Test removing non-existent worker assignment."""
        # Try to remove from non-existent GPU
        result = self.state.remove_assignment(0, "worker-1")
        assert result is False
        
        # Add assignment and try to remove different worker
        self.state.add_assignment(0, self.worker)
        result = self.state.remove_assignment(0, "worker-2")
        assert result is False
        
        # Verify original assignment still exists
        assert self.state.get_assignment_count(0) == 1
    
    def test_remove_assignment_invalid_parameters(self):
        """Test removing assignment with invalid parameters."""
        with pytest.raises(ValueError, match="gpu_id must be a non-negative integer"):
            self.state.remove_assignment(-1, "worker-1")
        
        with pytest.raises(ValueError, match="worker_id must be a non-empty string"):
            self.state.remove_assignment(0, "")
        
        with pytest.raises(ValueError, match="worker_id must be a non-empty string"):
            self.state.remove_assignment(0, None)
    
    def test_multiple_assignments_same_gpu(self):
        """Test multiple worker assignments to the same GPU."""
        worker2 = WorkerInfo(
            id="worker-2",
            enqueued_at=datetime.now(),
            callback=Mock(),
            on_error=Mock()
        )
        
        # Add multiple assignments
        self.state.add_assignment(0, self.worker)
        self.state.add_assignment(0, worker2)
        
        # Verify both assignments exist
        assignments = self.state.get_assignments(0)
        assert len(assignments) == 2
        assert self.worker in assignments
        assert worker2 in assignments
        
        assert self.state.get_assignment_count(0) == 2
        assert self.state.get_total_assignments() == 2
    
    def test_assignments_across_multiple_gpus(self):
        """Test worker assignments across multiple GPUs."""
        worker2 = WorkerInfo(
            id="worker-2",
            enqueued_at=datetime.now(),
            callback=Mock(),
            on_error=Mock()
        )
        
        # Add assignments to different GPUs
        self.state.add_assignment(0, self.worker)
        self.state.add_assignment(1, worker2)
        
        # Verify assignments are separate
        assert self.state.get_assignment_count(0) == 1
        assert self.state.get_assignment_count(1) == 1
        assert self.state.get_total_assignments() == 2
        
        # Verify correct assignments
        assignments_0 = self.state.get_assignments(0)
        assignments_1 = self.state.get_assignments(1)
        
        assert len(assignments_0) == 1
        assert assignments_0[0] == self.worker
        
        assert len(assignments_1) == 1
        assert assignments_1[0] == worker2
    
    def test_get_all_assignments(self):
        """Test getting all assignments across GPUs."""
        worker2 = WorkerInfo(
            id="worker-2",
            enqueued_at=datetime.now(),
            callback=Mock(),
            on_error=Mock()
        )
        
        # Add assignments
        self.state.add_assignment(0, self.worker)
        self.state.add_assignment(1, worker2)
        
        # Get all assignments
        all_assignments = self.state.get_all_assignments()
        
        assert len(all_assignments) == 2
        assert 0 in all_assignments
        assert 1 in all_assignments
        assert len(all_assignments[0]) == 1
        assert len(all_assignments[1]) == 1
        assert all_assignments[0][0] == self.worker
        assert all_assignments[1][0] == worker2
    
    def test_clear_assignments(self):
        """Test clearing all assignments."""
        # Add assignments
        self.state.add_assignment(0, self.worker)
        assert self.state.get_total_assignments() == 1
        
        # Clear assignments
        self.state.clear_assignments()
        
        # Verify all assignments are cleared
        assert self.state.get_total_assignments() == 0
        assert self.state.get_all_assignments() == {}
    
    def test_clear_all(self):
        """Test clearing all state."""
        # Set up state
        self.state.update_stats(self.gpu_stats)
        self.state.add_assignment(0, self.worker)
        
        # Verify state exists
        assert self.state.get_current_stats() is not None
        assert self.state.get_last_update() is not None
        assert self.state.get_total_assignments() == 1
        
        # Clear all state
        self.state.clear_all()
        
        # Verify all state is cleared
        assert self.state.get_current_stats() is None
        assert self.state.get_last_update() is None
        assert self.state.get_total_assignments() == 0
        assert self.state.get_all_assignments() == {}
    
    def test_thread_safety(self):
        """Test thread-safe operations on GPUResourceState."""
        results = []
        errors = []
        
        def worker_thread(worker_id):
            """Worker thread that performs operations."""
            try:
                worker = WorkerInfo(
                    id=f"worker-{worker_id}",
                    enqueued_at=datetime.now(),
                    callback=Mock(),
                    on_error=Mock()
                )
                
                # Add assignment
                self.state.add_assignment(worker_id % 2, worker)  # Distribute across 2 GPUs
                
                # Update stats (some threads)
                if worker_id % 3 == 0:
                    self.state.update_stats(self.gpu_stats)
                
                # Get assignments
                assignments = self.state.get_assignments(worker_id % 2)
                results.append(len(assignments))
                
                # Remove assignment
                removed = self.state.remove_assignment(worker_id % 2, f"worker-{worker_id}")
                results.append(removed)
                
            except Exception as e:
                errors.append(e)
        
        # Create and start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # Verify final state is consistent
        assert self.state.get_total_assignments() == 0


class TestWorkerAssignmentTracker:
    """Test cases for WorkerAssignmentTracker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = WorkerAssignmentTracker()
    
    def test_initial_state(self):
        """Test initial state of WorkerAssignmentTracker."""
        assert self.tracker.get_assignment_count() == 0
        assert self.tracker.get_all_assignments() == []
        assert self.tracker.get_assignment("worker-1") is None
        assert self.tracker.is_assigned("worker-1") is False
    
    def test_assign_worker(self):
        """Test assigning a worker to a GPU."""
        # Assign worker
        assignment = self.tracker.assign("worker-1", 0)
        
        # Verify assignment properties
        assert isinstance(assignment, GPUAssignment)
        assert assignment.worker_id == "worker-1"
        assert assignment.gpu_id == 0
        assert isinstance(assignment.assigned_at, datetime)
        
        # Verify tracker state
        assert self.tracker.get_assignment_count() == 1
        assert self.tracker.is_assigned("worker-1") is True
        
        # Verify assignment retrieval
        retrieved = self.tracker.get_assignment("worker-1")
        assert retrieved == assignment
    
    def test_assign_invalid_parameters(self):
        """Test assigning with invalid parameters."""
        with pytest.raises(ValueError, match="worker_id must be a non-empty string"):
            self.tracker.assign("", 0)
        
        with pytest.raises(ValueError, match="worker_id must be a non-empty string"):
            self.tracker.assign(None, 0)
        
        with pytest.raises(ValueError, match="gpu_id must be a non-negative integer"):
            self.tracker.assign("worker-1", -1)
        
        with pytest.raises(ValueError, match="gpu_id must be a non-negative integer"):
            self.tracker.assign("worker-1", "invalid")
    
    def test_assign_duplicate_worker(self):
        """Test assigning the same worker twice."""
        # Assign worker first time
        self.tracker.assign("worker-1", 0)
        
        # Try to assign same worker again
        with pytest.raises(ValueError, match="Worker worker-1 is already assigned to GPU 0"):
            self.tracker.assign("worker-1", 1)
    
    def test_release_worker(self):
        """Test releasing a worker assignment."""
        # Assign worker first
        original_assignment = self.tracker.assign("worker-1", 0)
        assert self.tracker.get_assignment_count() == 1
        
        # Release worker
        released_assignment = self.tracker.release("worker-1")
        
        # Verify released assignment matches original
        assert released_assignment == original_assignment
        
        # Verify tracker state
        assert self.tracker.get_assignment_count() == 0
        assert self.tracker.is_assigned("worker-1") is False
        assert self.tracker.get_assignment("worker-1") is None
    
    def test_release_nonexistent_worker(self):
        """Test releasing a non-existent worker."""
        result = self.tracker.release("worker-1")
        assert result is None
        assert self.tracker.get_assignment_count() == 0
    
    def test_release_invalid_parameters(self):
        """Test releasing with invalid parameters."""
        with pytest.raises(ValueError, match="worker_id must be a non-empty string"):
            self.tracker.release("")
        
        with pytest.raises(ValueError, match="worker_id must be a non-empty string"):
            self.tracker.release(None)
    
    def test_get_workers_for_gpu(self):
        """Test getting workers assigned to a specific GPU."""
        # Assign workers to different GPUs
        self.tracker.assign("worker-1", 0)
        self.tracker.assign("worker-2", 0)
        self.tracker.assign("worker-3", 1)
        
        # Get workers for GPU 0
        workers_gpu_0 = self.tracker.get_workers_for_gpu(0)
        assert len(workers_gpu_0) == 2
        assert "worker-1" in workers_gpu_0
        assert "worker-2" in workers_gpu_0
        
        # Get workers for GPU 1
        workers_gpu_1 = self.tracker.get_workers_for_gpu(1)
        assert len(workers_gpu_1) == 1
        assert "worker-3" in workers_gpu_1
        
        # Get workers for GPU with no assignments
        workers_gpu_2 = self.tracker.get_workers_for_gpu(2)
        assert len(workers_gpu_2) == 0
    
    def test_get_workers_for_gpu_invalid_parameters(self):
        """Test getting workers for GPU with invalid parameters."""
        with pytest.raises(ValueError, match="gpu_id must be a non-negative integer"):
            self.tracker.get_workers_for_gpu(-1)
        
        with pytest.raises(ValueError, match="gpu_id must be a non-negative integer"):
            self.tracker.get_workers_for_gpu("invalid")
    
    def test_get_all_assignments(self):
        """Test getting all assignments."""
        # Assign multiple workers
        assignment1 = self.tracker.assign("worker-1", 0)
        assignment2 = self.tracker.assign("worker-2", 1)
        
        # Get all assignments
        all_assignments = self.tracker.get_all_assignments()
        
        assert len(all_assignments) == 2
        assert assignment1 in all_assignments
        assert assignment2 in all_assignments
    
    def test_clear_all(self):
        """Test clearing all assignments."""
        # Assign workers
        self.tracker.assign("worker-1", 0)
        self.tracker.assign("worker-2", 1)
        assert self.tracker.get_assignment_count() == 2
        
        # Clear all assignments
        self.tracker.clear_all()
        
        # Verify all assignments are cleared
        assert self.tracker.get_assignment_count() == 0
        assert self.tracker.get_all_assignments() == []
        assert self.tracker.get_assignment("worker-1") is None
        assert self.tracker.get_assignment("worker-2") is None
    
    def test_thread_safety(self):
        """Test thread-safe operations on WorkerAssignmentTracker."""
        results = []
        errors = []
        
        def worker_thread(worker_id):
            """Worker thread that performs operations."""
            try:
                # Assign worker
                assignment = self.tracker.assign(f"worker-{worker_id}", worker_id % 3)
                results.append(assignment)
                
                # Check assignment
                retrieved = self.tracker.get_assignment(f"worker-{worker_id}")
                assert retrieved == assignment
                
                # Get workers for GPU
                workers = self.tracker.get_workers_for_gpu(worker_id % 3)
                results.append(len(workers))
                
                # Release worker
                released = self.tracker.release(f"worker-{worker_id}")
                assert released == assignment
                
            except Exception as e:
                errors.append(e)
        
        # Create and start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # Verify final state is consistent
        assert self.tracker.get_assignment_count() == 0
    
    def test_assignment_lifecycle(self):
        """Test complete assignment lifecycle."""
        # Initial state
        assert not self.tracker.is_assigned("worker-1")
        assert self.tracker.get_assignment("worker-1") is None
        
        # Assign worker
        assignment = self.tracker.assign("worker-1", 0)
        assert self.tracker.is_assigned("worker-1")
        assert self.tracker.get_assignment("worker-1") == assignment
        assert "worker-1" in self.tracker.get_workers_for_gpu(0)
        
        # Release worker
        released = self.tracker.release("worker-1")
        assert released == assignment
        assert not self.tracker.is_assigned("worker-1")
        assert self.tracker.get_assignment("worker-1") is None
        assert "worker-1" not in self.tracker.get_workers_for_gpu(0)


if __name__ == "__main__":
    pytest.main([__file__])