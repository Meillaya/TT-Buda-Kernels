"""
Metrics registry for TT-Buda dashboard.
"""

from typing import Any, Dict, List, Optional, Callable
import threading
from collections import defaultdict


class MetricsRegistry:
    """
    Registry for managing metric definitions and handlers.
    """
    
    def __init__(self):
        self._metrics = {}
        self._handlers = defaultdict(list)
        self._lock = threading.Lock()
    
    def register_metric(self, 
                       name: str, 
                       description: str, 
                       unit: str = "", 
                       tags: Optional[List[str]] = None):
        """Register a new metric."""
        with self._lock:
            self._metrics[name] = {
                'name': name,
                'description': description,
                'unit': unit,
                'tags': tags or [],
                'created_at': None
            }
    
    def register_handler(self, metric_name: str, handler: Callable[[str, Any], None]):
        """Register a handler for a specific metric."""
        with self._lock:
            self._handlers[metric_name].append(handler)
    
    def emit_metric(self, name: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Emit a metric value to all registered handlers."""
        with self._lock:
            handlers = self._handlers.get(name, [])
            for handler in handlers:
                try:
                    handler(name, value, metadata or {})
                except Exception as e:
                    # Log error but don't fail
                    print(f"Error in metric handler for {name}: {e}")
    
    def get_metric_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered metric."""
        with self._lock:
            return self._metrics.get(name)
    
    def list_metrics(self) -> List[str]:
        """List all registered metric names."""
        with self._lock:
            return list(self._metrics.keys())
    
    def remove_metric(self, name: str):
        """Remove a metric and its handlers."""
        with self._lock:
            if name in self._metrics:
                del self._metrics[name]
            if name in self._handlers:
                del self._handlers[name]


# Standard metrics
STANDARD_METRICS = [
    ("compilation_time", "Time taken to compile a model", "seconds"),
    ("inference_time", "Time taken for model inference", "seconds"),
    ("memory_usage", "Memory usage during operation", "bytes"),
    ("throughput", "Operations per second", "ops/sec"),
    ("latency", "Operation latency", "milliseconds"),
    ("accuracy", "Model accuracy", "percentage"),
    ("loss", "Training/validation loss", ""),
]


def create_default_registry() -> MetricsRegistry:
    """Create a registry with standard metrics."""
    registry = MetricsRegistry()
    
    for name, description, unit in STANDARD_METRICS:
        registry.register_metric(name, description, unit)
    
    return registry 