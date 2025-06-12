# TT-Buda Kernels Development Guide

This guide provides comprehensive instructions for developing and contributing to the TT-Buda Kernels project.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Architecture](#project-architecture)
3. [Development Workflow](#development-workflow)
4. [Implementation Phases](#implementation-phases)
5. [Testing Guidelines](#testing-guidelines)
6. [Performance Optimization](#performance-optimization)
7. [Contributing Guidelines](#contributing-guidelines)

## Development Environment Setup

### Prerequisites

- Python 3.8+ (recommended: 3.10+)
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)
- C++17 compatible compiler
- CMake 3.20+
- Git

### Quick Setup with uv

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/TT-Buda-Kernels.git
cd TT-Buda-Kernels

# Create virtual environment
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev,benchmark,dashboard]"

# Install pre-commit hooks
pre-commit install
```

### Docker Development Environment

```bash
# Build development container
docker build -t tt-buda-kernels:dev .

# Run development container
docker run -it --gpus all -v $(pwd):/workspace tt-buda-kernels:dev

# Or use docker-compose for full development stack
docker-compose up -d
```

### IDE Setup

#### VS Code
1. Install Python extension
2. Install C/C++ extension
3. Copy `.vscode/settings.json.example` to `.vscode/settings.json`
4. Set Python interpreter to `.venv/bin/python`

#### PyCharm
1. Set Python interpreter to `.venv/bin/python`
2. Enable PyTorch support
3. Configure code style to match Black settings

## Project Architecture

### Core Components

#### 1. Compiler (`src/tt_buda/core/compiler.py`)
- **Frontend**: PyTorch model ingestion and graph extraction
- **Middle-end**: Optimization passes and transformations
- **Backend**: Hardware-specific code generation

#### 2. Runtime (`src/tt_buda/core/runtime.py`)
- **Executor**: Manages model execution on target hardware
- **Memory Manager**: Handles tensor allocation and movement
- **Scheduler**: Coordinates kernel execution

#### 3. Kernels (`src/tt_buda/kernels/`)
- **Matrix Operations**: Optimized GEMM, convolution kernels
- **Element-wise**: Activation functions, arithmetic operations
- **Fusion**: Combines multiple operations for efficiency

#### 4. PyTorch Integration (`src/tt_buda/pytorch_integration/`)
- **Custom Operators**: TT-specific PyTorch operators
- **Autograd**: Automatic differentiation support
- **Dispatcher**: Routes operations to appropriate backends

### Key Design Patterns

#### 1. Factory Pattern
```python
# Backend creation
backend = Backend.create("tenstorrent", config)

# Compiler creation
compiler = Compiler.create(CompilerConfig(...))
```

#### 2. Strategy Pattern
```python
# Different optimization strategies
class OptimizationStrategy(ABC):
    @abstractmethod
    def optimize(self, graph: Graph) -> Graph:
        pass

class TenstorrentStrategy(OptimizationStrategy):
    def optimize(self, graph: Graph) -> Graph:
        # TT-specific optimizations
        pass
```

#### 3. Observer Pattern
```python
# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.observers = []
    
    def add_observer(self, observer):
        self.observers.append(observer)
    
    def notify(self, event):
        for observer in self.observers:
            observer.handle(event)
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/new-optimization-pass

# Make changes
# ... implement feature ...

# Run tests
pytest tests/

# Run linting
black src tests
isort src tests
flake8 src tests
mypy src

# Commit changes
git add .
git commit -m "feat: add new optimization pass for matrix fusion"

# Push and create PR
git push origin feature/new-optimization-pass
```

### 2. Code Quality Standards

#### Formatting
- **Black**: Code formatting (line length: 88)
- **isort**: Import sorting
- **flake8**: Linting

#### Type Checking
- **mypy**: Static type checking
- All public APIs must have type hints

#### Documentation
- **Docstrings**: Google style for all public functions/classes
- **Type hints**: Required for all function signatures
- **Examples**: Include usage examples in docstrings

### 3. Testing Strategy

#### Unit Tests
```python
# test_compiler.py
import pytest
from tt_buda.core.compiler import Compiler, CompilerConfig

class TestCompiler:
    def test_compile_linear_model(self):
        config = CompilerConfig(backend="cpu")
        compiler = Compiler(config)
        
        model = torch.nn.Linear(10, 5)
        compiled_model = compiler.compile(model)
        
        assert compiled_model is not None
        # Test inference
        input_tensor = torch.randn(1, 10)
        output = compiled_model(input_tensor)
        assert output.shape == (1, 5)
```

#### Integration Tests
```python
# test_end_to_end.py
def test_bert_compilation():
    """Test full BERT model compilation and execution."""
    from transformers import BertModel
    
    model = BertModel.from_pretrained("bert-base-uncased")
    compiled_model = tt_buda.compile(model, backend="tenstorrent")
    
    # Test inference
    inputs = torch.randint(0, 1000, (1, 128))
    outputs = compiled_model(inputs)
    assert outputs.last_hidden_state.shape == (1, 128, 768)
```

#### Performance Tests
```python
# test_performance.py
@pytest.mark.benchmark
def test_matmul_performance(benchmark):
    """Benchmark matrix multiplication performance."""
    def matmul_op():
        a = torch.randn(1024, 1024)
        b = torch.randn(1024, 1024)
        return torch.matmul(a, b)
    
    result = benchmark(matmul_op)
    assert result.shape == (1024, 1024)
```

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)

#### Goals
- [ ] Complete core compiler infrastructure
- [ ] Basic backend support (CPU, CUDA)
- [ ] PyTorch integration framework
- [ ] CI/CD pipeline

#### Key Tasks
1. **Compiler Infrastructure**
   - Graph representation and manipulation
   - Pass manager implementation
   - Basic optimization passes

2. **Backend Framework**
   - Abstract backend interface
   - CPU reference implementation
   - CUDA backend stub

3. **Testing Framework**
   - Unit test infrastructure
   - Integration test framework
   - Performance benchmarking setup

#### Success Criteria
- Compile simple PyTorch models (Linear, Conv2d)
- Execute on CPU backend
- 90%+ test coverage

### Phase 2: PyTorch Integration (Weeks 5-8)

#### Goals
- [ ] Custom PyTorch operators
- [ ] Automatic differentiation support
- [ ] Operation dispatch mechanism
- [ ] Model tracing and analysis

#### Key Tasks
1. **Custom Operators**
   ```cpp
   // tt_ops.cpp
   #include <torch/extension.h>
   
   torch::Tensor tt_matmul(torch::Tensor a, torch::Tensor b) {
       // TT-optimized matrix multiplication
       return torch::matmul(a, b);  // Placeholder
   }
   
   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
       m.def("tt_matmul", &tt_matmul, "TT optimized matmul");
   }
   ```

2. **Autograd Integration**
   ```python
   class TTMatmulFunction(torch.autograd.Function):
       @staticmethod
       def forward(ctx, a, b):
           ctx.save_for_backward(a, b)
           return tt_ops.tt_matmul(a, b)
       
       @staticmethod
       def backward(ctx, grad_output):
           a, b = ctx.saved_tensors
           grad_a = torch.matmul(grad_output, b.t())
           grad_b = torch.matmul(a.t(), grad_output)
           return grad_a, grad_b
   ```

#### Success Criteria
- Custom operators work with autograd
- Seamless PyTorch integration
- Support for common model architectures

### Phase 3: Kernel Development (Weeks 9-16)

#### Goals
- [ ] Optimized matrix multiplication kernels
- [ ] Convolution implementations
- [ ] Attention mechanism kernels
- [ ] Kernel fusion capabilities

#### Key Tasks
1. **GEMM Kernels**
   ```cpp
   // matmul_kernel.cpp
   void optimized_gemm(
       const float* A, const float* B, float* C,
       int M, int N, int K,
       const GemmConfig& config
   ) {
       // Optimized implementation with:
       // - Tile-based computation
       // - Memory prefetching
       // - SIMD instructions
       // - Cache optimization
   }
   ```

2. **Convolution Kernels**
   ```cpp
   // conv_kernel.cpp
   void optimized_conv2d(
       const float* input, const float* weight, float* output,
       const ConvConfig& config
   ) {
       // Optimized convolution with:
       // - Im2col transformation
       // - Winograd algorithm
       // - Channel-wise optimization
   }
   ```

#### Success Criteria
- 2x speedup over PyTorch defaults
- Support for multiple precisions (fp32, fp16, int8)
- Comprehensive kernel test suite

### Phase 4: Model Zoo (Weeks 17-24)

#### Goals
- [ ] BERT model optimization
- [ ] Stable Diffusion support
- [ ] Vision model implementations
- [ ] Benchmarking suite

#### Key Tasks
1. **BERT Implementation**
   ```python
   # model_zoo/nlp/bert.py
   class OptimizedBert(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.embeddings = OptimizedEmbeddings(config)
           self.encoder = OptimizedTransformerEncoder(config)
           self.pooler = OptimizedPooler(config)
       
       def forward(self, input_ids, attention_mask=None):
           # Optimized forward pass with fused operations
           pass
   ```

2. **Stable Diffusion Support**
   ```python
   # model_zoo/diffusion/stable_diffusion.py
   class OptimizedUNet(nn.Module):
       def __init__(self, config):
           # Optimized U-Net with attention fusion
           pass
   ```

#### Success Criteria
- 50%+ speedup on BERT inference
- Stable Diffusion image generation in <10s
- Comprehensive model accuracy validation

### Phase 5: Performance Dashboard (Weeks 25-32)

#### Goals
- [ ] Real-time performance monitoring
- [ ] Web-based dashboard interface
- [ ] Power consumption analysis
- [ ] Comparative benchmarking

#### Key Tasks
1. **Data Collection**
   ```python
   # dashboard/collector.py
   class PerformanceCollector:
       def collect_metrics(self, model, inputs):
           return {
               'latency': self.measure_latency(model, inputs),
               'throughput': self.measure_throughput(model, inputs),
               'memory_usage': self.measure_memory(model, inputs),
               'power_consumption': self.measure_power(model, inputs)
           }
   ```

2. **Dashboard Interface**
   ```python
   # dashboard/app.py
   import streamlit as st
   
   st.title("TT-Buda Performance Dashboard")
   
   # Model selection
   model_name = st.selectbox("Select Model", models)
   
   # Performance metrics
   col1, col2, col3 = st.columns(3)
   with col1:
       st.metric("Latency", f"{latency:.2f}ms")
   with col2:
       st.metric("Throughput", f"{throughput:.0f} samples/s")
   with col3:
       st.metric("Memory", f"{memory:.1f}GB")
   ```

#### Success Criteria
- Real-time monitoring with <1s latency
- Interactive performance visualization
- Automated regression detection

## Performance Optimization

### 1. Profiling and Debugging

#### Performance Profiling
```python
# Profile model execution
profile_path = tt_buda.profile(
    model, 
    inputs,
    backend="tenstorrent",
    output_dir="./profiles"
)

# Analyze results
profiler = PerformanceProfiler(profile_path)
bottlenecks = profiler.find_bottlenecks()
```

#### Memory Profiling
```python
# Monitor memory usage
with MemoryProfiler() as profiler:
    output = model(inputs)
    
memory_report = profiler.get_report()
```

### 2. Optimization Techniques

#### Kernel Fusion
```python
# Fuse sequential operations
@tt_buda.fuse_ops
def fused_linear_relu(x, weight, bias):
    """Fused linear + ReLU operation."""
    return torch.relu(torch.linear(x, weight, bias))
```

#### Memory Layout Optimization
```python
# Optimize tensor layouts for hardware
@tt_buda.optimize_layout
class OptimizedAttention(nn.Module):
    def forward(self, query, key, value):
        # Optimized attention with layout transformations
        pass
```

### 3. Hardware-Specific Optimizations

#### Tenstorrent Optimizations
- Tile-based computation patterns
- NOC (Network-on-Chip) optimization
- Multi-core parallelization
- Memory hierarchy utilization

#### CUDA Optimizations
- Tensor Core utilization
- Memory coalescing
- Warp-level optimizations
- Shared memory management

## Contributing Guidelines

### 1. Code Review Process

#### Before Submitting PR
- [ ] All tests pass locally
- [ ] Code formatted with Black
- [ ] Type checking passes
- [ ] Documentation updated
- [ ] Performance impact assessed

#### PR Requirements
- [ ] Clear description of changes
- [ ] Test coverage for new features
- [ ] Backward compatibility maintained
- [ ] Performance benchmarks included

### 2. Documentation Standards

#### Code Documentation
```python
def compile_model(
    model: nn.Module,
    backend: str = "tenstorrent",
    optimization_level: str = "O2",
    **kwargs: Any
) -> nn.Module:
    """
    Compile a PyTorch model for execution on target hardware.
    
    Args:
        model: The PyTorch model to compile
        backend: Target backend ("tenstorrent", "cuda", "cpu")
        optimization_level: Optimization level ("O0", "O1", "O2", "O3")
        **kwargs: Additional compilation options
        
    Returns:
        Compiled model ready for execution
        
    Example:
        >>> model = torch.nn.Linear(128, 64)
        >>> compiled_model = compile_model(model, backend="tenstorrent")
        >>> output = compiled_model(torch.randn(1, 128))
        
    Note:
        This function may take significant time for large models due to
        kernel compilation and optimization processes.
    """
```

#### API Documentation
- Use Sphinx for API documentation
- Include examples for all public APIs
- Maintain changelog for breaking changes

### 3. Performance Benchmarking

#### Benchmark Requirements
- [ ] Baseline measurements established
- [ ] Multiple input sizes tested
- [ ] Memory usage monitored
- [ ] Results reproducible
- [ ] Comparison with existing solutions

#### Regression Testing
```python
@pytest.mark.benchmark(group="matmul")
def test_matmul_performance_regression(benchmark):
    """Ensure matmul performance doesn't regress."""
    def matmul_op():
        return tt_buda.matmul(a, b)
    
    result = benchmark(matmul_op)
    
    # Assert performance within acceptable bounds
    assert benchmark.stats.mean < PERFORMANCE_THRESHOLD
```

### 4. Release Process

#### Version Management
- Semantic versioning (MAJOR.MINOR.PATCH)
- Release notes for each version
- Backward compatibility guarantees

#### Release Checklist
- [ ] All tests pass on CI
- [ ] Performance benchmarks run
- [ ] Documentation updated
- [ ] Version number bumped
- [ ] Release notes written
- [ ] Docker images built
- [ ] PyPI package published

---

This development guide provides the foundation for building a robust, high-performance AI compiler stack. As the project evolves, this guide should be updated to reflect new patterns, practices, and lessons learned. 