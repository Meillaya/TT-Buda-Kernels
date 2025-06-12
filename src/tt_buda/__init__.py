"""
TT-Buda Kernels: Open-Source AI Software Stack

A comprehensive AI software stack inspired by Tenstorrent's TT-Buda framework,
featuring PyTorch backend integration, optimized model zoo, and performance monitoring.
"""

__version__ = "0.1.0"
__author__ = "TT-Buda Kernels Contributors"
__email__ = "contributors@tt-buda-kernels.org"

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

# Core compiler and runtime imports
from .core.compiler import Compiler, CompilerConfig
from .core.runtime import Runtime, RuntimeConfig
from .core.backend import Backend, BackendConfig
from .core.operators import CustomOperator, OperatorRegistry

# PyTorch integration imports
from .pytorch_integration.custom_ops import register_custom_ops
from .pytorch_integration.dispatcher import TensorCoreDispatcher
from .pytorch_integration.autograd import TensorCoreFunction

# Model zoo imports
from .model_zoo import ModelZoo, OptimizedModel

# Performance monitoring imports
from .dashboard.collector import PerformanceCollector
from .dashboard.metrics import MetricsRegistry

# Utility imports
from .utils.config import load_config, save_config
from .utils.logging import get_logger

# Initialize the package
_logger = get_logger(__name__)
_registry = OperatorRegistry()
_model_zoo = ModelZoo()

# Global configuration
DEFAULT_CONFIG = {
    "backend": "tenstorrent",
    "optimization_level": "O2",
    "precision": "fp16",
    "enable_fusion": True,
    "enable_profiling": False,
    "cache_kernels": True,
    "enable_parallel_compilation": True,
}


def compile(
    model: nn.Module,
    *,
    backend: str = "tenstorrent",
    optimization_level: str = "O2",
    precision: str = "fp16",
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Compile a PyTorch model for execution on Tenstorrent hardware.
    
    Args:
        model: The PyTorch model to compile
        backend: Target backend ("tenstorrent", "cuda", "cpu")
        optimization_level: Optimization level ("O0", "O1", "O2", "O3")
        precision: Precision mode ("fp32", "fp16", "bf16", "int8")
        config: Additional configuration options
        **kwargs: Additional keyword arguments
        
    Returns:
        Compiled model ready for execution
        
    Example:
        >>> model = torch.nn.Linear(128, 64)
        >>> compiled_model = tt_buda.compile(model, backend="tenstorrent")
        >>> output = compiled_model(torch.randn(1, 128))
    """
    _logger.info(f"Compiling model for {backend} backend with {optimization_level} optimization")
    
    # Merge configuration
    compile_config = DEFAULT_CONFIG.copy()
    compile_config.update({
        "backend": backend,
        "optimization_level": optimization_level,
        "precision": precision,
    })
    if config:
        compile_config.update(config)
    compile_config.update(kwargs)
    
    # Initialize compiler
    compiler = Compiler(CompilerConfig(**compile_config))
    
    # Compile the model
    compiled_model = compiler.compile(model)
    
    _logger.info("Model compilation completed successfully")
    return compiled_model


def benchmark(
    model: nn.Module,
    input_shapes: List[tuple],
    *,
    batch_sizes: Optional[List[int]] = None,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    backend: str = "tenstorrent",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Benchmark a model's performance.
    
    Args:
        model: The model to benchmark
        input_shapes: List of input tensor shapes
        batch_sizes: List of batch sizes to test
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        backend: Target backend for benchmarking
        **kwargs: Additional benchmark options
        
    Returns:
        Dictionary containing benchmark results
    """
    from .benchmarks.runner import BenchmarkRunner
    
    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 32]
    
    runner = BenchmarkRunner(
        backend=backend,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
        **kwargs
    )
    
    return runner.benchmark(model, input_shapes, batch_sizes)


def profile(
    model: nn.Module,
    inputs: Union[torch.Tensor, List[torch.Tensor], tuple],
    *,
    backend: str = "tenstorrent",
    output_dir: str = "./profiles",
    **kwargs: Any,
) -> str:
    """
    Profile a model's execution.
    
    Args:
        model: The model to profile
        inputs: Input tensors for profiling
        backend: Target backend for profiling
        output_dir: Directory to save profile results
        **kwargs: Additional profiling options
        
    Returns:
        Path to the generated profile report
    """
    from .tools.profiler import ModelProfiler
    
    profiler = ModelProfiler(backend=backend, output_dir=output_dir, **kwargs)
    return profiler.profile(model, inputs)


def get_available_backends() -> List[str]:
    """Get list of available backends."""
    return Backend.get_available_backends()


def get_backend_info(backend_name: str) -> Dict[str, Any]:
    """Get information about a specific backend."""
    return Backend.get_backend_info(backend_name)


def list_optimized_models() -> List[str]:
    """List all available optimized models in the model zoo."""
    return _model_zoo.list_models()


def load_optimized_model(model_name: str, **kwargs: Any) -> OptimizedModel:
    """Load an optimized model from the model zoo."""
    return _model_zoo.load_model(model_name, **kwargs)


# Initialize custom operators on import
register_custom_ops()

# Export main components
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Main functions
    "compile",
    "benchmark", 
    "profile",
    
    # Backend utilities
    "get_available_backends",
    "get_backend_info",
    
    # Model zoo
    "list_optimized_models",
    "load_optimized_model",
    
    # Core classes
    "Compiler",
    "CompilerConfig", 
    "Runtime",
    "RuntimeConfig",
    "Backend",
    "BackendConfig",
    "CustomOperator",
    "OperatorRegistry",
    "ModelZoo",
    "OptimizedModel",
    
    # Performance monitoring
    "PerformanceCollector",
    "MetricsRegistry",
    
    # Utilities
    "load_config",
    "save_config",
    "get_logger",
] 