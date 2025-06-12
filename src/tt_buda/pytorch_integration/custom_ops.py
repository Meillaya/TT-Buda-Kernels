"""
Custom PyTorch operators for TT-Buda Kernels.

This module implements custom PyTorch operators that provide the integration
layer between PyTorch and TT-Buda's optimized kernels for Tenstorrent hardware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.logging import get_logger
from ..core.backend import Backend

logger = get_logger(__name__)

# Global operator registry
_CUSTOM_OPS_REGISTERED = False


def register_custom_ops() -> None:
    """Register custom operators with PyTorch."""
    global _CUSTOM_OPS_REGISTERED
    if _CUSTOM_OPS_REGISTERED:
        return
    
    logger.info("Registering TT-Buda custom operators with PyTorch")
    
    try:
        # Register TensorCore operations
        _register_tensorcore_ops()
        
        # Register memory operations
        _register_memory_ops()
        
        # Register activation operations
        _register_activation_ops()
        
        # Register reduction operations
        _register_reduction_ops()
        
        _CUSTOM_OPS_REGISTERED = True
        logger.info("Successfully registered all custom operators")
        
    except Exception as e:
        logger.error(f"Failed to register custom operators: {e}")
        raise


def _register_tensorcore_ops() -> None:
    """Register TensorCore operations (MatMul, Conv2D, etc.)."""
    
    # TensorCore MatMul
    @torch.library.custom_op("tt_buda::tensorcore_matmul", mutates_args=())
    def tensorcore_matmul(
        a: torch.Tensor,
        b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        precision: str = "fp16",
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """
        Optimized matrix multiplication using TensorCore operations.
        
        Args:
            a: Input tensor A [batch, ..., k]
            b: Input tensor B [k, n] or [batch, k, n]
            bias: Optional bias tensor [n]
            precision: Precision mode ("fp32", "fp16", "bf16")
            backend: Target backend ("tenstorrent", "cuda", "cpu")
            
        Returns:
            Result tensor [batch, ..., n]
        """
        # Get the appropriate backend
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'tensorcore_matmul'):
            return backend_impl.tensorcore_matmul(a, b, bias, precision)
        else:
            # Fallback to standard PyTorch implementation
            result = torch.matmul(a, b)
            if bias is not None:
                result = result + bias
            return result
    
    @tensorcore_matmul.register_fake
    def _(a, b, bias=None, precision="fp16", backend="tenstorrent"):
        # Shape inference for fake tensors
        if bias is not None:
            return torch.matmul(a, b) + bias
        return torch.matmul(a, b)
    
    # TensorCore Conv2D
    @torch.library.custom_op("tt_buda::tensorcore_conv2d", mutates_args=())
    def tensorcore_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
        dilation: Optional[List[int]] = None,
        groups: int = 1,
        precision: str = "fp16",
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """
        Optimized 2D convolution using TensorCore operations.
        
        Args:
            input: Input tensor [N, C_in, H_in, W_in]
            weight: Weight tensor [C_out, C_in/groups, K_h, K_w]
            bias: Optional bias tensor [C_out]
            stride: Convolution stride (h, w)
            padding: Convolution padding (h, w)
            dilation: Convolution dilation (h, w)
            groups: Number of groups for grouped convolution
            precision: Precision mode ("fp32", "fp16", "bf16")
            backend: Target backend ("tenstorrent", "cuda", "cpu")
            
        Returns:
            Output tensor [N, C_out, H_out, W_out]
        """
        # Handle default values
        if stride is None:
            stride = [1, 1]
        if padding is None:
            padding = [0, 0]
        if dilation is None:
            dilation = [1, 1]
            
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'tensorcore_conv2d'):
            return backend_impl.tensorcore_conv2d(
                input, weight, bias, stride, padding, dilation, groups, precision
            )
        else:
            # Fallback to standard PyTorch implementation
            return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    @tensorcore_conv2d.register_fake
    def _(input, weight, bias=None, stride=None, padding=None, 
          dilation=None, groups=1, precision="fp16", backend="tenstorrent"):
        # Handle default values
        if stride is None:
            stride = [1, 1]
        if padding is None:
            padding = [0, 0]
        if dilation is None:
            dilation = [1, 1]
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


def _register_memory_ops() -> None:
    """Register memory operations (copy, reshape, etc.)."""
    
    @torch.library.custom_op("tt_buda::optimized_copy", mutates_args=())
    def optimized_copy(
        input: torch.Tensor,
        device: str = "tenstorrent",
        non_blocking: bool = False
    ) -> torch.Tensor:
        """Optimized tensor copy operation."""
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'optimized_copy'):
            return backend_impl.optimized_copy(input, device, non_blocking)
        else:
            return input.clone()
    
    @optimized_copy.register_fake
    def _(input, device="tenstorrent", non_blocking=False):
        return input.clone()
    
    @torch.library.custom_op("tt_buda::optimized_reshape", mutates_args=())
    def optimized_reshape(
        input: torch.Tensor,
        shape: List[int],
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """Optimized tensor reshape operation."""
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'optimized_reshape'):
            return backend_impl.optimized_reshape(input, shape)
        else:
            return input.reshape(shape)
    
    @optimized_reshape.register_fake
    def _(input, shape, backend="tenstorrent"):
        return input.reshape(shape)


def _register_activation_ops() -> None:
    """Register activation operations (GELU, Swish, etc.)."""
    
    @torch.library.custom_op("tt_buda::fast_gelu", mutates_args=())
    def fast_gelu(
        input: torch.Tensor,
        approximate: bool = True,
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """Fast GELU activation function."""
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'fast_gelu'):
            return backend_impl.fast_gelu(input, approximate)
        else:
            if approximate:
                return F.gelu(input, approximate='tanh')
            else:
                return F.gelu(input)
    
    @fast_gelu.register_fake
    def _(input, approximate=True, backend="tenstorrent"):
        if approximate:
            return F.gelu(input, approximate='tanh')
        else:
            return F.gelu(input)
    
    @torch.library.custom_op("tt_buda::fast_swish", mutates_args=())
    def fast_swish(
        input: torch.Tensor,
        beta: float = 1.0,
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """Fast Swish activation function."""
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'fast_swish'):
            return backend_impl.fast_swish(input, beta)
        else:
            return input * torch.sigmoid(beta * input)
    
    @fast_swish.register_fake
    def _(input, beta=1.0, backend="tenstorrent"):
        return input * torch.sigmoid(beta * input)


def _register_reduction_ops() -> None:
    """Register reduction operations (softmax, layernorm, etc.)."""
    
    @torch.library.custom_op("tt_buda::fast_softmax", mutates_args=())
    def fast_softmax(
        input: torch.Tensor,
        dim: int = -1,
        precision: str = "fp16",
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """Fast softmax operation."""
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'fast_softmax'):
            return backend_impl.fast_softmax(input, dim, precision)
        else:
            return F.softmax(input, dim=dim)
    
    @fast_softmax.register_fake
    def _(input, dim=-1, precision="fp16", backend="tenstorrent"):
        return F.softmax(input, dim=dim)
    
    @torch.library.custom_op("tt_buda::fast_layernorm", mutates_args=())
    def fast_layernorm(
        input: torch.Tensor,
        normalized_shape: List[int],
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """Fast layer normalization."""
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'fast_layernorm'):
            return backend_impl.fast_layernorm(input, normalized_shape, weight, bias, eps)
        else:
            return F.layer_norm(input, normalized_shape, weight, bias, eps)
    
    @fast_layernorm.register_fake
    def _(input, normalized_shape, weight=None, bias=None, eps=1e-5, backend="tenstorrent"):
        return F.layer_norm(input, normalized_shape, weight, bias, eps)


# Attention operations for transformer models
def _register_attention_ops() -> None:
    """Register attention operations for transformer models."""
    
    @torch.library.custom_op("tt_buda::fused_attention", mutates_args=())
    def fused_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """Fused multi-head attention operation."""
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'fused_attention'):
            return backend_impl.fused_attention(
                query, key, value, attn_mask, dropout_p, is_causal, scale
            )
        else:
            # Fallback to PyTorch's scaled_dot_product_attention
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask, dropout_p, is_causal, scale
            )
    
    @fused_attention.register_fake
    def _(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, 
          scale=None, backend="tenstorrent"):
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale
        )


# Utility functions
def get_registered_ops() -> List[str]:
    """Get list of all registered custom operators."""
    ops = [
        "tt_buda::tensorcore_matmul",
        "tt_buda::tensorcore_conv2d",
        "tt_buda::optimized_copy",
        "tt_buda::optimized_reshape",
        "tt_buda::fast_gelu",
        "tt_buda::fast_swish",
        "tt_buda::fast_softmax",
        "tt_buda::fast_layernorm",
        "tt_buda::fused_attention",
    ]
    return ops


def is_custom_op_available(op_name: str) -> bool:
    """Check if a custom operator is available."""
    try:
        return hasattr(torch.ops.tt_buda, op_name.split("::")[-1])
    except:
        return False


# Initialize attention ops with the rest
_register_attention_ops() 