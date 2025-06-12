"""
Performance data collector for TT-Buda dashboard.
"""

from typing import Any, Dict, List, Optional
import time
import threading
from collections import defaultdict


class PerformanceCollector:
    """
    Collects performance metrics for TT-Buda operations.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self._lock = threading.Lock()
        self._start_times = {}
    
    def start_timing(self, operation_name: str) -> str:
        """Start timing an operation."""
        timing_id = f"{operation_name}_{time.time()}"
        with self._lock:
            self._start_times[timing_id] = time.perf_counter()
        return timing_id
    
    def end_timing(self, timing_id: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """End timing an operation and record the result."""
        end_time = time.perf_counter()
        with self._lock:
            if timing_id in self._start_times:
                duration = end_time - self._start_times[timing_id]
                del self._start_times[timing_id]
                
                metric_data = {
                    'duration': duration,
                    'timestamp': time.time(),
                    'metadata': metadata or {}
                }
                
                operation_name = timing_id.split('_')[0]
                self.metrics[operation_name].append(metric_data)
                return duration
        return 0.0
    
    def record_metric(self, metric_name: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Record a custom metric."""
        with self._lock:
            metric_data = {
                'value': value,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            self.metrics[metric_name].append(metric_data)
    
    def get_metrics(self, metric_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get collected metrics."""
        with self._lock:
            if metric_name:
                return {metric_name: self.metrics.get(metric_name, [])}
            return dict(self.metrics)
    
    def clear_metrics(self, metric_name: Optional[str] = None):
        """Clear collected metrics."""
        with self._lock:
            if metric_name:
                if metric_name in self.metrics:
                    del self.metrics[metric_name]
            else:
                self.metrics.clear() 