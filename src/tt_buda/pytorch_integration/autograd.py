"""
Autograd functions for TT-Buda custom operators.

This module implements PyTorch autograd functions that provide automatic
differentiation support for TT-Buda's custom operators.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.logging import get_logger
from ..core.backend import Backend

logger = get_logger(__name__)


class TensorCoreMatMulFunction(Function):
    """Autograd function for TensorCore matrix multiplication."""
    
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        precision: str = "fp16",
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """Forward pass for TensorCore MatMul."""
        # Save tensors for backward pass
        ctx.save_for_backward(a, b, bias)
        ctx.precision = precision
        ctx.backend = backend
        
        # Get backend implementation
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'tensorcore_matmul'):
            result = backend_impl.tensorcore_matmul(a, b, bias, precision)
        else:
            # Fallback implementation
            result = torch.matmul(a, b)
            if bias is not None:
                result = result + bias
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass for TensorCore MatMul."""
        a, b, bias = ctx.saved_tensors
        
        grad_a = grad_b = grad_bias = None
        
        if ctx.needs_input_grad[0]:  # grad_a
            grad_a = torch.matmul(grad_output, b.transpose(-2, -1))
        
        if ctx.needs_input_grad[1]:  # grad_b
            grad_b = torch.matmul(a.transpose(-2, -1), grad_output)
        
        if bias is not None and ctx.needs_input_grad[2]:  # grad_bias
            # Sum over all dimensions except the last one
            dims_to_sum = list(range(grad_output.ndim - 1))
            grad_bias = torch.sum(grad_output, dim=dims_to_sum)
        
        return grad_a, grad_b, grad_bias, None, None


class TensorCoreConv2DFunction(Function):
    """Autograd function for TensorCore 2D convolution."""
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        precision: str = "fp16",
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """Forward pass for TensorCore Conv2D."""
        # Save for backward pass
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.precision = precision
        ctx.backend = backend
        
        # Get backend implementation
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'tensorcore_conv2d'):
            result = backend_impl.tensorcore_conv2d(
                input, weight, bias, stride, padding, dilation, groups, precision
            )
        else:
            # Fallback implementation
            result = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass for TensorCore Conv2D."""
        input, weight, bias = ctx.saved_tensors
        
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:  # grad_input
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups
            )
        
        if ctx.needs_input_grad[1]:  # grad_weight  
            grad_weight = torch.nn.grad.conv2d_weight(
                input, weight.shape, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups
            )
        
        if bias is not None and ctx.needs_input_grad[2]:  # grad_bias
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3))
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class FastGELUFunction(Function):
    """Autograd function for fast GELU activation."""
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        approximate: bool = True,
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """Forward pass for fast GELU."""
        ctx.save_for_backward(input)
        ctx.approximate = approximate
        ctx.backend = backend
        
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'fast_gelu'):
            result = backend_impl.fast_gelu(input, approximate)
        else:
            if approximate:
                result = F.gelu(input, approximate='tanh')
            else:
                result = F.gelu(input)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass for fast GELU."""
        input, = ctx.saved_tensors
        
        if ctx.approximate:
            # Approximate GELU derivative
            tanh_arg = 0.7978845608 * (input + 0.044715 * input.pow(3))
            tanh_val = torch.tanh(tanh_arg)
            sech2_val = 1 - tanh_val.pow(2)
            
            grad_input = 0.5 * (1 + tanh_val + input * sech2_val * 
                               0.7978845608 * (1 + 3 * 0.044715 * input.pow(2)))
        else:
            # Exact GELU derivative
            sqrt_2_pi = (2.0 / torch.pi).sqrt()
            grad_input = 0.5 * (1 + torch.erf(input / 1.4142135623730951) + 
                               input * sqrt_2_pi * torch.exp(-0.5 * input.pow(2)))
        
        return grad_output * grad_input, None, None


class FastSoftmaxFunction(Function):
    """Autograd function for fast softmax."""
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        dim: int = -1,
        precision: str = "fp16",
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """Forward pass for fast softmax."""
        ctx.dim = dim
        ctx.precision = precision
        ctx.backend = backend
        
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'fast_softmax'):
            result = backend_impl.fast_softmax(input, dim, precision)
        else:
            result = F.softmax(input, dim=dim)
        
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass for fast softmax."""
        output, = ctx.saved_tensors
        
        # Softmax gradient: output * (grad_output - (output * grad_output).sum(dim, keepdim=True))
        grad_input = output * (grad_output - 
                              torch.sum(output * grad_output, dim=ctx.dim, keepdim=True))
        
        return grad_input, None, None, None


class FastLayerNormFunction(Function):
    """Autograd function for fast layer normalization."""
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        normalized_shape: List[int],
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """Forward pass for fast layer norm."""
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.backend = backend
        
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'fast_layernorm'):
            result = backend_impl.fast_layernorm(input, normalized_shape, weight, bias, eps)
        else:
            result = F.layer_norm(input, normalized_shape, weight, bias, eps)
        
        # Save tensors needed for backward pass
        ctx.save_for_backward(input, weight, bias)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass for fast layer norm."""
        input, weight, bias = ctx.saved_tensors
        
        # Use PyTorch's native layer norm backward
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            grad_input, grad_weight, grad_bias = torch.native_layer_norm_backward(
                grad_output, input, ctx.normalized_shape, weight, bias,
                ctx.eps, [ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]]
            )
        
        return grad_input, None, grad_weight, grad_bias, None, None


class FusedAttentionFunction(Function):
    """Autograd function for fused multi-head attention."""
    
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        backend: str = "tenstorrent"
    ) -> torch.Tensor:
        """Forward pass for fused attention."""
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.backend = backend
        
        backend_impl = Backend.get_current_backend()
        
        if backend_impl and hasattr(backend_impl, 'fused_attention'):
            result = backend_impl.fused_attention(
                query, key, value, attn_mask, dropout_p, is_causal, scale
            )
        else:
            result = F.scaled_dot_product_attention(
                query, key, value, attn_mask, dropout_p, is_causal, scale
            )
        
        # Save for backward (simplified - in practice would need attention weights)
        ctx.save_for_backward(query, key, value, attn_mask)
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass for fused attention."""
        query, key, value, attn_mask = ctx.saved_tensors
        
        # For simplicity, use PyTorch's backward implementation
        # In a real implementation, we'd compute gradients more efficiently
        with torch.enable_grad():
            query_grad = query.detach().requires_grad_(True) if ctx.needs_input_grad[0] else None
            key_grad = key.detach().requires_grad_(True) if ctx.needs_input_grad[1] else None
            value_grad = value.detach().requires_grad_(True) if ctx.needs_input_grad[2] else None
            
            if query_grad is not None or key_grad is not None or value_grad is not None:
                inputs = [query_grad or query, key_grad or key, value_grad or value]
                result = F.scaled_dot_product_attention(
                    inputs[0], inputs[1], inputs[2], attn_mask, 
                    ctx.dropout_p, ctx.is_causal, ctx.scale
                )
                
                grads = torch.autograd.grad(
                    result, [inp for inp in inputs if inp.requires_grad],
                    grad_output, retain_graph=False
                )
                
                grad_idx = 0
                grad_query = grads[grad_idx] if query_grad is not None else None
                grad_idx += 1 if query_grad is not None else 0
                grad_key = grads[grad_idx] if key_grad is not None else None
                grad_idx += 1 if key_grad is not None else 0
                grad_value = grads[grad_idx] if value_grad is not None else None
            else:
                grad_query = grad_key = grad_value = None
        
        return grad_query, grad_key, grad_value, None, None, None, None, None


# Wrapper functions that use the autograd functions
def tensorcore_matmul_autograd(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    precision: str = "fp16",
    backend: str = "tenstorrent"
) -> torch.Tensor:
    """TensorCore MatMul with autograd support."""
    return TensorCoreMatMulFunction.apply(a, b, bias, precision, backend)


def tensorcore_conv2d_autograd(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    precision: str = "fp16",
    backend: str = "tenstorrent"
) -> torch.Tensor:
    """TensorCore Conv2D with autograd support."""
    return TensorCoreConv2DFunction.apply(
        input, weight, bias, stride, padding, dilation, groups, precision, backend
    )


def fast_gelu_autograd(
    input: torch.Tensor,
    approximate: bool = True,
    backend: str = "tenstorrent"
) -> torch.Tensor:
    """Fast GELU with autograd support."""
    return FastGELUFunction.apply(input, approximate, backend)


def fast_softmax_autograd(
    input: torch.Tensor,
    dim: int = -1,
    precision: str = "fp16",
    backend: str = "tenstorrent"
) -> torch.Tensor:
    """Fast softmax with autograd support."""
    return FastSoftmaxFunction.apply(input, dim, precision, backend)


def fast_layernorm_autograd(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    backend: str = "tenstorrent"
) -> torch.Tensor:
    """Fast layer norm with autograd support."""
    return FastLayerNormFunction.apply(input, normalized_shape, weight, bias, eps, backend)


def fused_attention_autograd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    backend: str = "tenstorrent"
) -> torch.Tensor:
    """Fused attention with autograd support."""
    return FusedAttentionFunction.apply(
        query, key, value, attn_mask, dropout_p, is_causal, scale, backend
    )


# Export the main autograd function class for external use
TensorCoreFunction = TensorCoreMatMulFunction 