"""
Benchmark runner for TT-Buda Kernels.

This module provides benchmarking functionality for models and operations.
"""

import time
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import statistics

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkRunner:
    """Benchmark runner for TT-Buda models and operations."""
    
    def __init__(
        self,
        backend: str = "tenstorrent",
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        **kwargs: Any
    ):
        self.backend = backend
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        self.kwargs = kwargs
        self.logger = get_logger(__name__)
    
    def benchmark(
        self,
        model: nn.Module,
        input_shapes: List[tuple],
        batch_sizes: List[int]
    ) -> Dict[str, Any]:
        """
        Benchmark a model's performance.
        
        Args:
            model: The model to benchmark
            input_shapes: List of input tensor shapes
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary containing benchmark results
        """
        self.logger.info(f"Starting benchmark with {self.num_iterations} iterations")
        
        results = {
            "backend": self.backend,
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "batch_results": {},
            "summary": {}
        }
        
        for batch_size in batch_sizes:
            self.logger.info(f"Benchmarking with batch size: {batch_size}")
            
            # Create input tensors
            inputs = []
            for shape in input_shapes:
                full_shape = (batch_size,) + shape
                inputs.append(torch.randn(full_shape))
            
            # Warm up
            self._warmup(model, inputs)
            
            # Benchmark
            times = self._run_benchmark(model, inputs)
            
            # Calculate statistics
            stats = self._calculate_stats(times, batch_size)
            results["batch_results"][batch_size] = stats
        
        # Calculate summary statistics
        results["summary"] = self._calculate_summary(results["batch_results"])
        
        self.logger.info("Benchmark completed")
        return results
    
    def _warmup(self, model: nn.Module, inputs: List[torch.Tensor]) -> None:
        """Warm up the model before benchmarking."""
        self.logger.debug(f"Warming up for {self.warmup_iterations} iterations")
        
        model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                if len(inputs) == 1:
                    _ = model(inputs[0])
                else:
                    _ = model(*inputs)
    
    def _run_benchmark(self, model: nn.Module, inputs: List[torch.Tensor]) -> List[float]:
        """Run the actual benchmark and collect timing data."""
        times = []
        
        model.eval()
        with torch.no_grad():
            for i in range(self.num_iterations):
                start_time = time.perf_counter()
                
                if len(inputs) == 1:
                    _ = model(inputs[0])
                else:
                    _ = model(*inputs)
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                
                if (i + 1) % 20 == 0:
                    self.logger.debug(f"Completed {i + 1}/{self.num_iterations} iterations")
        
        return times
    
    def _calculate_stats(self, times: List[float], batch_size: int) -> Dict[str, Any]:
        """Calculate statistics from timing data."""
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        
        # Calculate throughput (samples per second)
        throughput = batch_size / mean_time
        
        return {
            "batch_size": batch_size,
            "mean_time": mean_time,
            "median_time": median_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_dev": std_dev,
            "throughput": throughput,
            "times": times
        }
    
    def _calculate_summary(self, batch_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics across all batch sizes."""
        if not batch_results:
            return {}
        
        all_throughputs = [result["throughput"] for result in batch_results.values()]
        all_mean_times = [result["mean_time"] for result in batch_results.values()]
        
        return {
            "best_throughput": max(all_throughputs),
            "worst_throughput": min(all_throughputs),
            "avg_throughput": statistics.mean(all_throughputs),
            "fastest_mean_time": min(all_mean_times),
            "slowest_mean_time": max(all_mean_times),
            "avg_mean_time": statistics.mean(all_mean_times)
        } 