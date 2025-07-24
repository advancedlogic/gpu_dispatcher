"""Unit tests for the worker queue implementation."""

import pytest
import threading
import time
import logging
from datetime import datetime
from unittest.mock import Mock, call, patch

from gpu_worker_pool.worker_queue import FIFOWorkerQueue, WorkerQueue
from gpu_worker_pool.models import WorkerInfo


class TestFIFOWorkerQueue:
    """Test cases for the FIFO worker queue implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.queue = FIFOWorkerQueue()
        self.mock_callback = Mock()
        self.mock_error_handler = Mock()
    
    def create_worker(self, worker_id: str = "test-worker") -> WorkerInfo:
        """Create a test worker with mock callbacks."""
        return WorkerInfo(
            id=worker_id,
            enqueued_at=datetime.now(),
            callback=self.mock_callback,
            on_error=self.mock_error_handler
        )
    
    def test_empty_queue_initialization(self):
        """Test that a new queue is empty."""
        assert self.queue.size() == 0
        assert self.queue.dequeue() is None
        assert self.queue.peek() is None
        assert self.queue.get_all_workers() == []
    
    def test_enqueue_single_worker(self):
        """Test enqueueing a single worker."""
        worker = self.create_worker("worker-1")
        
        self.queue.enqueue(worker)
        
        assert self.queue.size() == 1
        assert self.queue.peek() == worker
        assert self.queue.get_all_workers() == [worker]
    
    def test_enqueue_multiple_workers(self):
        """Test enqueueing multiple workers."""
        worker1 = self.create_worker("worker-1")
        worker2 = self.create_worker("worker-2")
        worker3 = self.create_worker("worker-3")
        
        self.queue.enqueue(worker1)
        self.queue.enqueue(worker2)
        self.queue.enqueue(worker3)
        
        assert self.queue.size() == 3
        assert self.queue.peek() == worker1
        assert self.queue.get_all_workers() == [worker1, worker2, worker3]
    
    def test_dequeue_single_worker(self):
        """Test dequeueing a single worker."""
        worker = self.create_worker("worker-1")
        self.queue.enqueue(worker)
        
        dequeued = self.queue.dequeue()
        
        assert dequeued == worker
        assert self.queue.size() == 0
        assert self.queue.dequeue() is None
    
    def test_dequeue_multiple_workers_fifo_order(self):
        """Test that workers are dequeued in FIFO order."""
        worker1 = self.create_worker("worker-1")
        worker2 = self.create_worker("worker-2")
        worker3 = self.create_worker("worker-3")
        
        self.queue.enqueue(worker1)
        self.queue.enqueue(worker2)
        self.queue.enqueue(worker3)
        
        assert self.queue.dequeue() == worker1
        assert self.queue.dequeue() == worker2
        assert self.queue.dequeue() == worker3
        assert self.queue.dequeue() is None
    
    def test_dequeue_empty_queue(self):
        """Test dequeueing from an empty queue."""
        assert self.queue.dequeue() is None
        assert self.queue.size() == 0
    
    def test_clear_empty_queue(self):
        """Test clearing an empty queue."""
        self.queue.clear()
        assert self.queue.size() == 0
    
    def test_clear_queue_with_workers(self):
        """Test clearing a queue with workers calls error handlers."""
        worker1 = self.create_worker("worker-1")
        worker2 = self.create_worker("worker-2")
        
        # Use separate mock error handlers for each worker
        error_handler1 = Mock()
        error_handler2 = Mock()
        worker1.on_error = error_handler1
        worker2.on_error = error_handler2
        
        self.queue.enqueue(worker1)
        self.queue.enqueue(worker2)
        
        self.queue.clear()
        
        assert self.queue.size() == 0
        assert self.queue.dequeue() is None
        
        # Verify error handlers were called
        error_handler1.assert_called_once()
        error_handler2.assert_called_once()
        
        # Check that the exception message indicates cancellation
        call_args1 = error_handler1.call_args[0][0]
        call_args2 = error_handler2.call_args[0][0]
        assert "cancelled due to queue clear" in str(call_args1)
        assert "cancelled due to queue clear" in str(call_args2)
    
    def test_clear_queue_ignores_error_handler_exceptions(self):
        """Test that clear() continues even if error handlers raise exceptions."""
        worker1 = self.create_worker("worker-1")
        worker2 = self.create_worker("worker-2")
        
        # Make error handlers raise exceptions
        error_handler1 = Mock(side_effect=Exception("Handler error"))
        error_handler2 = Mock(side_effect=RuntimeError("Another error"))
        worker1.on_error = error_handler1
        worker2.on_error = error_handler2
        
        self.queue.enqueue(worker1)
        self.queue.enqueue(worker2)
        
        # Should not raise an exception
        self.queue.clear()
        
        assert self.queue.size() == 0
        error_handler1.assert_called_once()
        error_handler2.assert_called_once()
    
    def test_peek_does_not_modify_queue(self):
        """Test that peek() doesn't modify the queue."""
        worker1 = self.create_worker("worker-1")
        worker2 = self.create_worker("worker-2")
        
        self.queue.enqueue(worker1)
        self.queue.enqueue(worker2)
        
        # Multiple peeks should return the same worker
        assert self.queue.peek() == worker1
        assert self.queue.peek() == worker1
        assert self.queue.size() == 2
        
        # Dequeue should still return the first worker
        assert self.queue.dequeue() == worker1
        assert self.queue.size() == 1
    
    def test_remove_worker_by_id(self):
        """Test removing a specific worker by ID."""
        worker1 = self.create_worker("worker-1")
        worker2 = self.create_worker("worker-2")
        worker3 = self.create_worker("worker-3")
        
        self.queue.enqueue(worker1)
        self.queue.enqueue(worker2)
        self.queue.enqueue(worker3)
        
        # Remove middle worker
        assert self.queue.remove_worker("worker-2") is True
        assert self.queue.size() == 2
        assert self.queue.get_all_workers() == [worker1, worker3]
        
        # Verify error handler was called for removed worker
        worker2.on_error.assert_called_once()
        call_args = worker2.on_error.call_args[0][0]
        assert "worker-2 removed from queue" in str(call_args)
    
    def test_remove_worker_not_found(self):
        """Test removing a worker that doesn't exist."""
        worker1 = self.create_worker("worker-1")
        self.queue.enqueue(worker1)
        
        assert self.queue.remove_worker("nonexistent") is False
        assert self.queue.size() == 1
        assert self.queue.get_all_workers() == [worker1]
    
    def test_remove_worker_empty_id(self):
        """Test removing a worker with empty ID."""
        worker1 = self.create_worker("worker-1")
        self.queue.enqueue(worker1)
        
        assert self.queue.remove_worker("") is False
        assert self.queue.remove_worker(None) is False
        assert self.queue.size() == 1
    
    def test_enqueue_none_worker_raises_error(self):
        """Test that enqueueing None raises ValueError."""
        with pytest.raises(ValueError, match="Worker cannot be None"):
            self.queue.enqueue(None)
    
    def test_enqueue_invalid_worker_type_raises_error(self):
        """Test that enqueueing invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Worker must be a WorkerInfo instance"):
            self.queue.enqueue("not a worker")
    
    def test_thread_safety_concurrent_enqueue_dequeue(self):
        """Test thread safety with concurrent enqueue and dequeue operations."""
        num_workers = 100
        workers_enqueued = []
        workers_dequeued = []
        
        def enqueue_workers():
            for i in range(num_workers):
                worker = self.create_worker(f"worker-{i}")
                workers_enqueued.append(worker)
                self.queue.enqueue(worker)
                time.sleep(0.001)  # Small delay to encourage interleaving
        
        def dequeue_workers():
            while len(workers_dequeued) < num_workers:
                worker = self.queue.dequeue()
                if worker is not None:
                    workers_dequeued.append(worker)
                time.sleep(0.001)  # Small delay to encourage interleaving
        
        # Start threads
        enqueue_thread = threading.Thread(target=enqueue_workers)
        dequeue_thread = threading.Thread(target=dequeue_workers)
        
        enqueue_thread.start()
        dequeue_thread.start()
        
        # Wait for completion
        enqueue_thread.join()
        dequeue_thread.join()
        
        # Verify all workers were processed
        assert len(workers_enqueued) == num_workers
        assert len(workers_dequeued) == num_workers
        assert self.queue.size() == 0
        
        # Verify FIFO order (workers should be dequeued in the same order they were enqueued)
        enqueued_ids = [w.id for w in workers_enqueued]
        dequeued_ids = [w.id for w in workers_dequeued]
        assert enqueued_ids == dequeued_ids
    
    def test_thread_safety_concurrent_size_operations(self):
        """Test thread safety of size() method with concurrent operations."""
        results = []
        
        def size_checker():
            for _ in range(50):
                size = self.queue.size()
                results.append(size)
                time.sleep(0.001)
        
        def queue_modifier():
            for i in range(25):
                worker = self.create_worker(f"worker-{i}")
                self.queue.enqueue(worker)
                time.sleep(0.001)
                self.queue.dequeue()
                time.sleep(0.001)
        
        # Start threads
        size_thread = threading.Thread(target=size_checker)
        modifier_thread = threading.Thread(target=queue_modifier)
        
        size_thread.start()
        modifier_thread.start()
        
        # Wait for completion
        size_thread.join()
        modifier_thread.join()
        
        # Verify that all size results are non-negative integers
        assert all(isinstance(size, int) and size >= 0 for size in results)
        assert len(results) == 50
    
    def test_thread_safety_concurrent_clear(self):
        """Test thread safety of clear() method with concurrent operations."""
        # Add some workers
        for i in range(10):
            worker = self.create_worker(f"worker-{i}")
            self.queue.enqueue(worker)
        
        clear_completed = threading.Event()
        
        def clear_queue():
            self.queue.clear()
            clear_completed.set()
        
        def try_dequeue():
            while not clear_completed.is_set():
                self.queue.dequeue()
                time.sleep(0.001)
        
        # Start threads
        clear_thread = threading.Thread(target=clear_queue)
        dequeue_thread = threading.Thread(target=try_dequeue)
        
        clear_thread.start()
        dequeue_thread.start()
        
        # Wait for completion
        clear_thread.join()
        dequeue_thread.join()
        
        # Queue should be empty after clear
        assert self.queue.size() == 0
        assert self.queue.dequeue() is None
    
    @patch('gpu_worker_pool.worker_queue.logging.getLogger')
    def test_block_worker_with_logging(self, mock_get_logger):
        """Test blocking a worker with proper logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        worker = self.create_worker("worker-1")
        reason = "No GPUs available"
        
        self.queue.block_worker(worker, reason)
        
        assert self.queue.size() == 1
        assert self.queue.peek() == worker
        
        # Verify logging was called
        mock_get_logger.assert_called_once_with('gpu_worker_pool.worker_queue')
        mock_logger.info.assert_called_once()
        
        # Check log message contains worker ID and reason
        log_call_args = mock_logger.info.call_args[0][0]
        assert "worker-1" in log_call_args
        assert "No GPUs available" in log_call_args
        assert "Blocking worker" in log_call_args
    
    def test_block_worker_none_raises_error(self):
        """Test that blocking None worker raises ValueError."""
        with pytest.raises(ValueError, match="Worker cannot be None"):
            self.queue.block_worker(None, "test reason")
    
    def test_block_worker_invalid_type_raises_error(self):
        """Test that blocking invalid worker type raises ValueError."""
        with pytest.raises(ValueError, match="Worker must be a WorkerInfo instance"):
            self.queue.block_worker("not a worker", "test reason")
    
    @patch('gpu_worker_pool.worker_queue.logging.getLogger')
    def test_block_worker_empty_reason(self, mock_get_logger):
        """Test blocking a worker with empty reason uses default."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        worker = self.create_worker("worker-1")
        
        self.queue.block_worker(worker, "")
        
        # Should use default reason
        log_call_args = mock_logger.info.call_args[0][0]
        assert "No reason provided" in log_call_args
    
    @patch('gpu_worker_pool.worker_queue.logging.getLogger')
    def test_unblock_next_worker_with_logging(self, mock_get_logger):
        """Test unblocking next worker with proper logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        worker1 = self.create_worker("worker-1")
        worker2 = self.create_worker("worker-2")
        
        self.queue.enqueue(worker1)
        self.queue.enqueue(worker2)
        
        unblocked = self.queue.unblock_next_worker()
        
        assert unblocked == worker1
        assert self.queue.size() == 1
        assert self.queue.peek() == worker2
        
        # Verify logging was called
        mock_get_logger.assert_called_once_with('gpu_worker_pool.worker_queue')
        mock_logger.info.assert_called_once()
        
        # Check log message contains worker ID
        log_call_args = mock_logger.info.call_args[0][0]
        assert "worker-1" in log_call_args
        assert "Unblocking worker" in log_call_args
    
    def test_unblock_next_worker_empty_queue(self):
        """Test unblocking from empty queue returns None."""
        assert self.queue.unblock_next_worker() is None
        assert self.queue.size() == 0
    
    @patch('gpu_worker_pool.worker_queue.logging.getLogger')
    def test_unblock_workers_multiple(self, mock_get_logger):
        """Test unblocking multiple workers."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        worker1 = self.create_worker("worker-1")
        worker2 = self.create_worker("worker-2")
        worker3 = self.create_worker("worker-3")
        
        self.queue.enqueue(worker1)
        self.queue.enqueue(worker2)
        self.queue.enqueue(worker3)
        
        unblocked = self.queue.unblock_workers(2)
        
        assert len(unblocked) == 2
        assert unblocked[0] == worker1
        assert unblocked[1] == worker2
        assert self.queue.size() == 1
        assert self.queue.peek() == worker3
        
        # Verify logging was called for each worker plus summary
        assert mock_logger.info.call_count == 3  # 2 individual + 1 summary
        
        # Check individual log messages
        individual_calls = mock_logger.info.call_args_list[:2]
        assert "worker-1" in individual_calls[0][0][0]
        assert "worker-2" in individual_calls[1][0][0]
        
        # Check summary log message
        summary_call = mock_logger.info.call_args_list[2][0][0]
        assert "Unblocked 2 workers" in summary_call
    
    def test_unblock_workers_more_than_available(self):
        """Test unblocking more workers than available."""
        worker1 = self.create_worker("worker-1")
        worker2 = self.create_worker("worker-2")
        
        self.queue.enqueue(worker1)
        self.queue.enqueue(worker2)
        
        unblocked = self.queue.unblock_workers(5)
        
        assert len(unblocked) == 2
        assert unblocked[0] == worker1
        assert unblocked[1] == worker2
        assert self.queue.size() == 0
    
    def test_unblock_workers_zero_count(self):
        """Test unblocking zero workers returns empty list."""
        worker = self.create_worker("worker-1")
        self.queue.enqueue(worker)
        
        unblocked = self.queue.unblock_workers(0)
        
        assert unblocked == []
        assert self.queue.size() == 1
    
    def test_unblock_workers_negative_count_raises_error(self):
        """Test that negative count raises ValueError."""
        with pytest.raises(ValueError, match="Count must be non-negative"):
            self.queue.unblock_workers(-1)
    
    @patch('gpu_worker_pool.worker_queue.logging.getLogger')
    def test_unblock_workers_empty_queue(self, mock_get_logger):
        """Test unblocking from empty queue."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        unblocked = self.queue.unblock_workers(3)
        
        assert unblocked == []
        assert self.queue.size() == 0
        
        # No logging should occur for empty queue
        mock_logger.info.assert_not_called()
    
    def test_blocking_unblocking_integration(self):
        """Test integration of blocking and unblocking operations."""
        worker1 = self.create_worker("worker-1")
        worker2 = self.create_worker("worker-2")
        worker3 = self.create_worker("worker-3")
        
        # Block workers with different reasons
        self.queue.block_worker(worker1, "GPU memory threshold exceeded")
        self.queue.block_worker(worker2, "GPU utilization threshold exceeded")
        self.queue.block_worker(worker3, "All GPUs assigned")
        
        assert self.queue.size() == 3
        
        # Unblock workers one by one
        unblocked1 = self.queue.unblock_next_worker()
        assert unblocked1 == worker1
        assert self.queue.size() == 2
        
        # Unblock remaining workers in batch
        remaining = self.queue.unblock_workers(2)
        assert len(remaining) == 2
        assert remaining[0] == worker2
        assert remaining[1] == worker3
        assert self.queue.size() == 0
    
    def test_thread_safety_concurrent_blocking_unblocking(self):
        """Test thread safety with concurrent blocking and unblocking operations."""
        num_workers = 50
        blocked_workers = []
        unblocked_workers = []
        
        def block_workers():
            for i in range(num_workers):
                worker = self.create_worker(f"worker-{i}")
                blocked_workers.append(worker)
                self.queue.block_worker(worker, f"Blocking reason {i}")
                time.sleep(0.001)
        
        def unblock_workers():
            while len(unblocked_workers) < num_workers:
                worker = self.queue.unblock_next_worker()
                if worker is not None:
                    unblocked_workers.append(worker)
                time.sleep(0.001)
        
        # Start threads
        block_thread = threading.Thread(target=block_workers)
        unblock_thread = threading.Thread(target=unblock_workers)
        
        block_thread.start()
        unblock_thread.start()
        
        # Wait for completion
        block_thread.join()
        unblock_thread.join()
        
        # Verify all workers were processed
        assert len(blocked_workers) == num_workers
        assert len(unblocked_workers) == num_workers
        assert self.queue.size() == 0
        
        # Verify FIFO order
        blocked_ids = [w.id for w in blocked_workers]
        unblocked_ids = [w.id for w in unblocked_workers]
        assert blocked_ids == unblocked_ids


class TestWorkerQueueInterface:
    """Test the abstract WorkerQueue interface."""
    
    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        
        class IncompleteQueue(WorkerQueue):
            pass
        
        with pytest.raises(TypeError):
            IncompleteQueue()
    
    def test_fifo_queue_implements_interface(self):
        """Test that FIFOWorkerQueue properly implements the interface."""
        queue = FIFOWorkerQueue()
        assert isinstance(queue, WorkerQueue)
        
        # Verify all abstract methods are implemented
        assert hasattr(queue, 'enqueue')
        assert hasattr(queue, 'dequeue')
        assert hasattr(queue, 'size')
        assert hasattr(queue, 'clear')
        assert hasattr(queue, 'block_worker')
        assert hasattr(queue, 'unblock_next_worker')
        assert hasattr(queue, 'unblock_workers')
        
        assert callable(queue.enqueue)
        assert callable(queue.dequeue)
        assert callable(queue.size)
        assert callable(queue.clear)
        assert callable(queue.block_worker)
        assert callable(queue.unblock_next_worker)
        assert callable(queue.unblock_workers)


if __name__ == "__main__":
    pytest.main([__file__])