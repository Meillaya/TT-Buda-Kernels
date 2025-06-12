"""
TensorCore operation dispatcher for TT-Buda Kernels.

This module implements the dispatch layer that routes PyTorch operations
to the appropriate TT-Buda backend implementations based on hardware
availability, tensor properties, and performance characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from functools import wraps
import threading
from contextlib import contextmanager

from ..utils.logging import get_logger
from ..core.backend import Backend, BackendConfig
from .autograd import (
    tensorcore_matmul_autograd,
    tensorcore_conv2d_autograd,
    fast_gelu_autograd,
    fast_softmax_autograd,
    fast_layernorm_autograd,
    fused_attention_autograd,
)

logger = get_logger(__name__)


class DispatchConfig:
    """Configuration for the TensorCore dispatcher."""
    
    def __init__(
        self,
        preferred_backend: str = "tenstorrent",
        fallback_backends: List[str] = None,
        min_tensor_size: int = 1024,
        precision_mode: str = "fp16",
        enable_fusion: bool = True,
        enable_caching: bool = True,
        performance_threshold: float = 0.8
    ):
        self.preferred_backend = preferred_backend
        self.fallback_backends = fallback_backends or ["cuda", "cpu"]
        self.min_tensor_size = min_tensor_size
        self.precision_mode = precision_mode
        self.enable_fusion = enable_fusion
        self.enable_caching = enable_caching
        self.performance_threshold = performance_threshold


class TensorCoreDispatcher:
    """
    Main dispatcher for TensorCore operations.
    
    Routes operations to the most appropriate backend based on:
    - Hardware availability
    - Tensor size and properties
    - Operation type and complexity
    - Performance characteristics
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self.config = DispatchConfig()
        self.backends: Dict[str, Backend] = {}
        self.operation_cache: Dict[str, Callable] = {}
        self.performance_stats: Dict[str, Dict[str, float]] = {}
        
        # Initialize available backends
        self._initialize_backends()
        
        # Register operation dispatchers
        self._register_dispatchers()
        
        self._initialized = True
        logger.info("TensorCore dispatcher initialized")
    
    def _initialize_backends(self) -> None:
        """Initialize available backends."""
        available_backends = Backend.get_available_backends()
        
        for backend_name in available_backends:
            try:
                config = BackendConfig(backend=backend_name)
                backend = Backend.create(backend_name, config)
                self.backends[backend_name] = backend
                logger.info(f"Initialized {backend_name} backend")
            except Exception as e:
                logger.warning(f"Failed to initialize {backend_name} backend: {e}")
    
    def _register_dispatchers(self) -> None:
        """Register operation dispatchers."""
        self.dispatchers = {
            'matmul': self._dispatch_matmul,
            'conv2d': self._dispatch_conv2d,
            'gelu': self._dispatch_gelu,
            'softmax': self._dispatch_softmax,
            'layernorm': self._dispatch_layernorm,
            'attention': self._dispatch_attention,
        }
    
    def get_best_backend(
        self,
        operation: str,
        *tensors: torch.Tensor,
        **kwargs: Any
    ) -> str:
        """
        Determine the best backend for a given operation and tensors.
        
        Args:
            operation: Operation name
            *tensors: Input tensors
            **kwargs: Additional operation parameters
            
        Returns:
            Name of the best backend to use
        """
        # Check if tensors are too small for acceleration
        total_elements = sum(t.numel() for t in tensors)
        if total_elements < self.config.min_tensor_size:
            return "cpu"
        
        # Check backend availability in preferred order
        backends_to_try = [self.config.preferred_backend] + self.config.fallback_backends
        
        for backend_name in backends_to_try:
            if backend_name in self.backends:
                backend = self.backends[backend_name]
                
                # Check if backend supports this operation
                if self._backend_supports_operation(backend, operation):
                    # Check if tensors are compatible
                    if self._tensors_compatible(backend, tensors):
                        return backend_name
        
        # Fallback to CPU
        return "cpu"
    
    def _backend_supports_operation(self, backend: Backend, operation: str) -> bool:
        """Check if backend supports the given operation."""
        operation_methods = {
            'matmul': 'tensorcore_matmul',
            'conv2d': 'tensorcore_conv2d',
            'gelu': 'fast_gelu',
            'softmax': 'fast_softmax',
            'layernorm': 'fast_layernorm',
            'attention': 'fused_attention',
        }
        
        method_name = operation_methods.get(operation)
        return method_name and hasattr(backend, method_name)
    
    def _tensors_compatible(self, backend: Backend, tensors: Tuple[torch.Tensor, ...]) -> bool:
        """Check if tensors are compatible with the backend."""
        for tensor in tensors:
            # Check device compatibility
            if hasattr(backend, 'device_type'):
                if tensor.device.type != backend.device_type and tensor.device.type != 'cpu':
                    return False
            
            # Check dtype support
            if hasattr(backend, 'supported_dtypes'):
                if tensor.dtype not in backend.supported_dtypes:
                    return False
        
        return True
    
    def _dispatch_matmul(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Dispatch matrix multiplication operation."""
        backend_name = self.get_best_backend('matmul', a, b, *([bias] if bias else []))
        
        if backend_name == "cpu":
            # Use standard PyTorch
            result = torch.matmul(a, b)
            if bias is not None:
                result = result + bias
            return result
        else:
            # Use TensorCore implementation
            return tensorcore_matmul_autograd(
                a, b, bias,
                precision=kwargs.get('precision', self.config.precision_mode),
                backend=backend_name
            )
    
    def _dispatch_conv2d(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        **kwargs: Any
    ) -> torch.Tensor:
        """Dispatch 2D convolution operation."""
        tensors = [input, weight] + ([bias] if bias else [])
        backend_name = self.get_best_backend('conv2d', *tensors)
        
        if backend_name == "cpu":
            return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        else:
            return tensorcore_conv2d_autograd(
                input, weight, bias, stride, padding, dilation, groups,
                precision=kwargs.get('precision', self.config.precision_mode),
                backend=backend_name
            )
    
    def _dispatch_gelu(
        self,
        input: torch.Tensor,
        approximate: bool = True,
        **kwargs: Any
    ) -> torch.Tensor:
        """Dispatch GELU activation."""
        backend_name = self.get_best_backend('gelu', input)
        
        if backend_name == "cpu":
            return F.gelu(input, approximate='tanh' if approximate else 'none')
        else:
            return fast_gelu_autograd(input, approximate, backend_name)
    
    def _dispatch_softmax(
        self,
        input: torch.Tensor,
        dim: int = -1,
        **kwargs: Any
    ) -> torch.Tensor:
        """Dispatch softmax operation."""
        backend_name = self.get_best_backend('softmax', input)
        
        if backend_name == "cpu":
            return F.softmax(input, dim=dim)
        else:
            return fast_softmax_autograd(
                input, dim,
                precision=kwargs.get('precision', self.config.precision_mode),
                backend=backend_name
            )
    
    def _dispatch_layernorm(
        self,
        input: torch.Tensor,
        normalized_shape: List[int],
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
        **kwargs: Any
    ) -> torch.Tensor:
        """Dispatch layer normalization."""
        tensors = [input] + ([weight] if weight else []) + ([bias] if bias else [])
        backend_name = self.get_best_backend('layernorm', *tensors)
        
        if backend_name == "cpu":
            return F.layer_norm(input, normalized_shape, weight, bias, eps)
        else:
            return fast_layernorm_autograd(
                input, normalized_shape, weight, bias, eps, backend_name
            )
    
    def _dispatch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Dispatch attention operation."""
        tensors = [query, key, value] + ([attn_mask] if attn_mask else [])
        backend_name = self.get_best_backend('attention', *tensors)
        
        if backend_name == "cpu":
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask, dropout_p, is_causal, scale
            )
        else:
            return fused_attention_autograd(
                query, key, value, attn_mask, dropout_p, is_causal, scale, backend_name
            )
    
    def dispatch_operation(
        self,
        operation: str,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Dispatch any operation to the appropriate backend.
        
        Args:
            operation: Operation name
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
        """
        if operation not in self.dispatchers:
            raise ValueError(f"Unknown operation: {operation}")
        
        dispatcher = self.dispatchers[operation]
        return dispatcher(*args, **kwargs)
    
    def configure(self, config: DispatchConfig) -> None:
        """Update dispatcher configuration."""
        self.config = config
        logger.info(f"Dispatcher configured with preferred backend: {config.preferred_backend}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for different backends."""
        return self.performance_stats.copy()
    
    def clear_cache(self) -> None:
        """Clear operation cache."""
        self.operation_cache.clear()
        logger.info("Dispatcher cache cleared")


# Global dispatcher instance
_dispatcher = TensorCoreDispatcher()


def get_dispatcher() -> TensorCoreDispatcher:
    """Get the global TensorCore dispatcher instance."""
    return _dispatcher


@contextmanager
def backend_context(backend: str):
    """Context manager for temporarily switching backends."""
    dispatcher = get_dispatcher()
    old_backend = dispatcher.config.preferred_backend
    
    try:
        dispatcher.config.preferred_backend = backend
        yield
    finally:
        dispatcher.config.preferred_backend = old_backend


def dispatch_to_tensorcore(operation: str):
    """Decorator to automatically dispatch operations to TensorCore."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                dispatcher = get_dispatcher()
                return dispatcher.dispatch_operation(operation, *args, **kwargs)
            except Exception as e:
                logger.warning(f"TensorCore dispatch failed for {operation}: {e}")
                # Fallback to original function
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience functions for common operations
def tensorcore_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    **kwargs: Any
) -> torch.Tensor:
    """Dispatch-enabled matrix multiplication."""
    return _dispatcher.dispatch_operation('matmul', a, b, bias, **kwargs)


def tensorcore_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    **kwargs: Any
) -> torch.Tensor:
    """Dispatch-enabled 2D convolution."""
    return _dispatcher.dispatch_operation(
        'conv2d', input, weight, bias, stride, padding, dilation, groups, **kwargs
    )


def tensorcore_gelu(
    input: torch.Tensor,
    approximate: bool = True,
    **kwargs: Any
) -> torch.Tensor:
    """Dispatch-enabled GELU activation."""
    return _dispatcher.dispatch_operation('gelu', input, approximate, **kwargs)


def tensorcore_softmax(
    input: torch.Tensor,
    dim: int = -1,
    **kwargs: Any
) -> torch.Tensor:
    """Dispatch-enabled softmax."""
    return _dispatcher.dispatch_operation('softmax', input, dim, **kwargs)


def tensorcore_layernorm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    **kwargs: Any
) -> torch.Tensor:
    """Dispatch-enabled layer normalization."""
    return _dispatcher.dispatch_operation(
        'layernorm', input, normalized_shape, weight, bias, eps, **kwargs
    )


def tensorcore_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    **kwargs: Any
) -> torch.Tensor:
    """Dispatch-enabled attention operation."""
    return _dispatcher.dispatch_operation(
        'attention', query, key, value, attn_mask, dropout_p, is_causal, scale, **kwargs
    ) 