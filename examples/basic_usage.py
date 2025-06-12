#!/usr/bin/env python3
"""
Basic usage example for TT-Buda Kernels.

This example demonstrates how to compile and run a simple PyTorch model
using the TT-Buda compiler stack.
"""

import torch
import torch.nn as nn
import tt_buda

# Example 1: Simple Linear Model
def example_linear_model():
    """Example using a simple linear model."""
    print("=== Example 1: Linear Model ===")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    print(f"Original model: {model}")
    
    # Compile the model for Tenstorrent backend
    try:
        compiled_model = tt_buda.compile(
            model,
            backend="tenstorrent",
            optimization_level="O2",
            precision="fp16"
        )
        print("✓ Model compiled successfully")
        
        # Test inference
        input_tensor = torch.randn(1, 128)
        output = compiled_model(input_tensor)
        print(f"✓ Inference successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ Compilation failed: {e}")


# Example 2: CNN Model
def example_cnn_model():
    """Example using a CNN model."""
    print("\n=== Example 2: CNN Model ===")
    
    # Create a simple CNN
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    print(f"CNN model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Compile with different settings
    try:
        compiled_model = tt_buda.compile(
            model,
            backend="tenstorrent",
            optimization_level="O3",
            precision="fp16",
            enable_fusion=True
        )
        print("✓ CNN model compiled successfully")
        
        # Test with image-like input
        input_tensor = torch.randn(1, 3, 32, 32)
        output = compiled_model(input_tensor)
        print(f"✓ CNN inference successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ CNN compilation failed: {e}")


# Example 3: Backend Information
def example_backend_info():
    """Example showing backend information."""
    print("\n=== Example 3: Backend Information ===")
    
    # List available backends
    backends = tt_buda.get_available_backends()
    print(f"Available backends: {backends}")
    
    # Get backend information
    for backend in backends:
        info = tt_buda.get_backend_info(backend)
        print(f"\n{backend.upper()} Backend:")
        print(f"  Name: {info.get('name', 'N/A')}")
        print(f"  Description: {info.get('description', 'N/A')}")
        print(f"  Precision Support: {info.get('precision_support', [])}")
        print(f"  Features: {info.get('features', [])}")


# Example 4: Model Benchmarking
def example_benchmarking():
    """Example showing model benchmarking."""
    print("\n=== Example 4: Model Benchmarking ===")
    
    # Create a simple model for benchmarking
    model = nn.Linear(512, 512)
    
    try:
        # Benchmark the model
        results = tt_buda.benchmark(
            model,
            input_shapes=[(512,)],
            batch_sizes=[1, 8, 16, 32],
            num_iterations=50,
            warmup_iterations=10,
            backend="tenstorrent"
        )
        
        print("✓ Benchmarking completed")
        print(f"Benchmark results: {results}")
        
    except Exception as e:
        print(f"✗ Benchmarking failed: {e}")


# Example 5: Model Profiling
def example_profiling():
    """Example showing model profiling."""
    print("\n=== Example 5: Model Profiling ===")
    
    # Create a model for profiling
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    try:
        # Profile the model
        input_tensor = torch.randn(16, 256)
        profile_path = tt_buda.profile(
            model,
            input_tensor,
            backend="tenstorrent",
            output_dir="./profiles"
        )
        
        print("✓ Profiling completed")
        print(f"Profile saved to: {profile_path}")
        
    except Exception as e:
        print(f"✗ Profiling failed: {e}")


def main():
    """Run all examples."""
    print("TT-Buda Kernels - Basic Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_linear_model()
    example_cnn_model()
    example_backend_info()
    example_benchmarking()
    example_profiling()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main() 