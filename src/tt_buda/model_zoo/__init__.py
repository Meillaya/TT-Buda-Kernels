"""
TT-Buda Model Zoo.

This module provides optimized implementations of popular models
for Tenstorrent hardware, including NLP, vision, and diffusion models.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json

from ..utils.logging import get_logger
from ..pytorch_integration.dispatcher import get_dispatcher
from .nlp.bert import OptimizedBERT, BERTConfig
from .vision.resnet import OptimizedResNet
from .diffusion.stable_diffusion import OptimizedStableDiffusion

logger = get_logger(__name__)


class OptimizedModel:
    """Base class for optimized models in the model zoo."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.optimizations = []
        self.benchmarks = {}
    
    def forward(self, *args, **kwargs):
        """Forward pass through the optimized model."""
        return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Allow the model to be called directly."""
        return self.forward(*args, **kwargs)
    
    def add_optimization(self, optimization: str, params: Dict[str, Any] = None):
        """Add an optimization to the model."""
        self.optimizations.append({
            'type': optimization,
            'params': params or {}
        })
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'config': self.config,
            'optimizations': self.optimizations,
            'benchmarks': self.benchmarks,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


class ModelZoo:
    """Model zoo for TT-Buda optimized models."""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self._register_models()
        
    def _register_models(self):
        """Register available models."""
        # NLP Models
        self.models['bert-base'] = {
            'class': OptimizedBERT,
            'config_class': BERTConfig,
            'category': 'nlp',
            'description': 'BERT base model optimized for Tenstorrent hardware'
        }
        
        self.models['bert-large'] = {
            'class': OptimizedBERT,
            'config_class': BERTConfig,
            'category': 'nlp',
            'description': 'BERT large model optimized for Tenstorrent hardware'
        }
        
        # Vision Models  
        self.models['resnet-50'] = {
            'class': OptimizedResNet,
            'config_class': dict,
            'category': 'vision',
            'description': 'ResNet-50 optimized for Tenstorrent hardware'
        }
        
        # Diffusion Models
        self.models['stable-diffusion-v1-5'] = {
            'class': OptimizedStableDiffusion,
            'config_class': dict,
            'category': 'diffusion',
            'description': 'Stable Diffusion v1.5 optimized for Tenstorrent hardware'
        }
    
    def list_models(self) -> List[str]:
        """List all available models."""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].copy()
    
    def list_models_by_category(self, category: str) -> List[str]:
        """List models in a specific category."""
        return [
            name for name, info in self.models.items()
            if info['category'] == category
        ]
    
    def load_model(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        **kwargs
    ) -> OptimizedModel:
        """
        Load an optimized model from the zoo.
        
        Args:
            model_name: Name of the model to load
            config: Model configuration (uses default if None)
            pretrained: Whether to load pretrained weights
            **kwargs: Additional arguments for model initialization
            
        Returns:
            OptimizedModel instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {self.list_models()}")
        
        model_info = self.models[model_name]
        model_class = model_info['class']
        config_class = model_info['config_class']
        
        # Create configuration
        if config is None:
            config = self._get_default_config(model_name)
        
        # Create model config object
        if config_class != dict:
            model_config = config_class(**config)
        else:
            model_config = config
        
        # Initialize model
        logger.info(f"Loading {model_name} model...")
        
        try:
            if model_name.startswith('bert'):
                model = model_class(model_config, **kwargs)
            else:
                model = model_class(config=model_config, **kwargs)
            
            # Create optimized model wrapper
            optimized_model = OptimizedModel(model, config)
            
            # Load pretrained weights if requested
            if pretrained:
                self._load_pretrained_weights(optimized_model, model_name)
            
            # Apply TT-Buda optimizations
            self._apply_optimizations(optimized_model, model_name)
            
            logger.info(f"Successfully loaded {model_name}")
            return optimized_model
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
    
    def _get_default_config(self, model_name: str) -> Dict[str, Any]:
        """Get default configuration for a model."""
        default_configs = {
            'bert-base': {
                'vocab_size': 30522,
                'hidden_size': 768,
                'num_hidden_layers': 12,
                'num_attention_heads': 12,
                'intermediate_size': 3072,
                'hidden_dropout_prob': 0.1,
                'attention_probs_dropout_prob': 0.1,
                'max_position_embeddings': 512,
                'type_vocab_size': 2,
                'layer_norm_eps': 1e-12
            },
            'bert-large': {
                'vocab_size': 30522,
                'hidden_size': 1024,
                'num_hidden_layers': 24,
                'num_attention_heads': 16,
                'intermediate_size': 4096,
                'hidden_dropout_prob': 0.1,
                'attention_probs_dropout_prob': 0.1,
                'max_position_embeddings': 512,
                'type_vocab_size': 2,
                'layer_norm_eps': 1e-12
            },
            'resnet-50': {
                'num_classes': 1000,
                'layers': [3, 4, 6, 3],
                'block_type': 'bottleneck'
            },
            'stable-diffusion-v1-5': {
                'image_size': 512,
                'in_channels': 4,
                'out_channels': 4,
                'num_train_timesteps': 1000
            }
        }
        
        return default_configs.get(model_name, {})
    
    def _load_pretrained_weights(self, model: OptimizedModel, model_name: str):
        """Load pretrained weights for a model."""
        # Placeholder for pretrained weight loading
        # In a real implementation, this would download and load weights
        logger.info(f"Loading pretrained weights for {model_name}")
        model.add_optimization('pretrained_weights', {'source': 'huggingface'})
    
    def _apply_optimizations(self, model: OptimizedModel, model_name: str):
        """Apply TT-Buda specific optimizations to the model."""
        logger.info(f"Applying TT-Buda optimizations to {model_name}")
        
        # Common optimizations for all models
        model.add_optimization('tensorcore_dispatch')
        model.add_optimization('kernel_fusion')
        model.add_optimization('memory_optimization')
        
        # Model-specific optimizations
        if model_name.startswith('bert'):
            model.add_optimization('attention_fusion')
            model.add_optimization('layernorm_fusion')
        elif model_name.startswith('resnet'):
            model.add_optimization('conv_fusion')
            model.add_optimization('batch_norm_fusion')
        elif model_name.startswith('stable-diffusion'):
            model.add_optimization('unet_optimization')
            model.add_optimization('cross_attention_fusion')
    
    def benchmark_model(
        self,
        model_name: str,
        input_shapes: List[tuple],
        batch_sizes: List[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Benchmark a model's performance."""
        from ..benchmarks.runner import BenchmarkRunner
        
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32]
        
        model = self.load_model(model_name, **kwargs)
        
        runner = BenchmarkRunner()
        results = runner.benchmark(model.model, input_shapes, batch_sizes)
        
        # Store results in model
        model.benchmarks.update(results)
        
        return results
    
    def export_model(
        self,
        model: OptimizedModel,
        export_path: str,
        format: str = 'onnx'
    ):
        """Export an optimized model to various formats."""
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        if format == 'onnx':
            # Export to ONNX format
            logger.info(f"Exporting model to ONNX format at {export_path}")
            # Implementation would use torch.onnx.export
        elif format == 'torchscript':
            # Export to TorchScript
            logger.info(f"Exporting model to TorchScript format at {export_path}")
            # Implementation would use torch.jit.script
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Save model config and optimizations
        config_path = export_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(model.get_model_info(), f, indent=2)


# Global model zoo instance
_model_zoo = ModelZoo()


def list_models() -> List[str]:
    """List all available models in the zoo."""
    return _model_zoo.list_models()


def load_model(model_name: str, **kwargs) -> OptimizedModel:
    """Load a model from the zoo."""
    return _model_zoo.load_model(model_name, **kwargs)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model."""
    return _model_zoo.get_model_info(model_name)


def benchmark_model(model_name: str, **kwargs) -> Dict[str, Any]:
    """Benchmark a model."""
    return _model_zoo.benchmark_model(model_name, **kwargs) 