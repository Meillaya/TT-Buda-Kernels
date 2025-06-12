"""
Model profiler for TT-Buda Kernels.

This module provides profiling functionality to analyze model performance.
"""

import os
import time
import json
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import contextlib

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ModelProfiler:
    """Model profiler for TT-Buda models."""
    
    def __init__(
        self,
        backend: str = "tenstorrent",
        output_dir: str = "./profiles",
        enable_memory_profiling: bool = True,
        enable_timing_profiling: bool = True,
        **kwargs: Any
    ):
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_timing_profiling = enable_timing_profiling
        self.kwargs = kwargs
        self.logger = get_logger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def profile(
        self,
        model: nn.Module,
        inputs: Union[torch.Tensor, List[torch.Tensor], tuple]
    ) -> str:
        """
        Profile a model's execution.
        
        Args:
            model: The model to profile
            inputs: Input tensors for profiling
            
        Returns:
            Path to the generated profile report
        """
        self.logger.info(f"Starting model profiling with {self.backend} backend")
        
        # Ensure inputs is a list/tuple
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        elif not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        
        # Generate profile timestamp
        timestamp = int(time.time())
        profile_name = f"profile_{timestamp}"
        
        # Collect profiling data
        profile_data = {
            "timestamp": timestamp,
            "backend": self.backend,
            "model_info": self._get_model_info(model),
            "input_info": self._get_input_info(inputs),
            "timing_profile": None,
            "memory_profile": None,
            "layer_profile": None
        }
        
        # Run timing profiling
        if self.enable_timing_profiling:
            self.logger.info("Running timing profiling")
            profile_data["timing_profile"] = self._profile_timing(model, inputs)
        
        # Run memory profiling
        if self.enable_memory_profiling:
            self.logger.info("Running memory profiling")
            profile_data["memory_profile"] = self._profile_memory(model, inputs)
        
        # Run layer-wise profiling
        self.logger.info("Running layer-wise profiling")
        profile_data["layer_profile"] = self._profile_layers(model, inputs)
        
        # Save profile report
        profile_path = self._save_profile(profile_name, profile_data)
        
        self.logger.info(f"Profiling completed. Report saved to: {profile_path}")
        return str(profile_path)
    
    def _get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get information about the model."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_class": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming fp32
            "modules": [name for name, _ in model.named_modules() if name]
        }
    
    def _get_input_info(self, inputs: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Get information about input tensors."""
        input_info = []
        
        for i, tensor in enumerate(inputs):
            info = {
                "index": i,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "requires_grad": tensor.requires_grad,
                "memory_mb": tensor.numel() * tensor.element_size() / (1024 * 1024)
            }
            input_info.append(info)
        
        return input_info
    
    def _profile_timing(self, model: nn.Module, inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """Profile model timing."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                if len(inputs) == 1:
                    _ = model(inputs[0])
                else:
                    _ = model(*inputs)
        
        # Timing runs
        times = []
        with torch.no_grad():
            for _ in range(100):
                start_time = time.perf_counter()
                
                if len(inputs) == 1:
                    _ = model(inputs[0])
                else:
                    _ = model(*inputs)
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        import statistics
        return {
            "mean_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "num_runs": len(times)
        }
    
    def _profile_memory(self, model: nn.Module, inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """Profile memory usage."""
        memory_info = {
            "initial_memory": 0,
            "peak_memory": 0,
            "final_memory": 0
        }
        
        # Get initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_info["initial_memory"] = torch.cuda.memory_allocated() / (1024 * 1024)
        
        # Run model and track peak memory
        model.eval()
        with torch.no_grad():
            if len(inputs) == 1:
                _ = model(inputs[0])
            else:
                _ = model(*inputs)
            
            if torch.cuda.is_available():
                memory_info["peak_memory"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
                memory_info["final_memory"] = torch.cuda.memory_allocated() / (1024 * 1024)
        
        return memory_info
    
    def _profile_layers(self, model: nn.Module, inputs: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Profile individual layers."""
        layer_profiles = []
        
        # Hook to capture layer execution times
        layer_times = {}
        
        def make_hook(name):
            def hook(module, input, output):
                start_time = getattr(module, '_start_time', None)
                if start_time is not None:
                    layer_times[name] = time.perf_counter() - start_time
            return hook
        
        def make_pre_hook(name):
            def pre_hook(module, input):
                module._start_time = time.perf_counter()
            return pre_hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if name and len(list(module.children())) == 0:  # Leaf modules only
                hooks.append(module.register_forward_pre_hook(make_pre_hook(name)))
                hooks.append(module.register_forward_hook(make_hook(name)))
        
        # Run model
        model.eval()
        with torch.no_grad():
            if len(inputs) == 1:
                _ = model(inputs[0])
            else:
                _ = model(*inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Create layer profiles
        for name, exec_time in layer_times.items():
            module = dict(model.named_modules())[name]
            layer_profiles.append({
                "layer_name": name,
                "layer_type": module.__class__.__name__,
                "execution_time": exec_time,
                "parameters": sum(p.numel() for p in module.parameters()),
            })
        
        # Sort by execution time
        layer_profiles.sort(key=lambda x: x["execution_time"], reverse=True)
        
        return layer_profiles
    
    def _save_profile(self, profile_name: str, profile_data: Dict[str, Any]) -> Path:
        """Save profile data to file."""
        profile_path = self.output_dir / f"{profile_name}.json"
        
        # Convert any non-serializable objects
        serializable_data = self._make_serializable(profile_data)
        
        with open(profile_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        return profile_path
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj) 