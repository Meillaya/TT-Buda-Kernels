# TT-Buda Kernels: Open-Source AI Software Stack

An open-source AI compiler and runtime stack inspired by Tenstorrent's TT-Buda framework, featuring PyTorch backend integration, optimized kernels, and performance monitoring.

## Project Goals

- **PyTorch Integration**: Add Tenstorrent backend support via custom operators
- **Optimized Model Zoo**: Pre-optimized implementations of popular models (BERT, Stable Diffusion, etc.)
- **Performance Dashboard**: Real-time monitoring and benchmarking tools for AI workloads
- **Kernel Optimization**: High-performance implementations for matrix operations, convolutions, and attention

## Quick Start

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Clone and install
git clone https://github.com/yourusername/TT-Buda-Kernels.git
cd TT-Buda-Kernels
uv venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
uv pip install -e .
```

### Running Examples

```bash
# Basic usage
uv run python examples/basic_usage.py

# Test implementation
uv run python test_implementation.py

# Start dashboard
uv run streamlit run src/tt_buda/dashboard/app.py
```

### Basic Usage

```python
import torch
import tt_buda

# Create and compile a model
model = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32)
)

compiled_model = tt_buda.compile(model, backend="cpu")
output = compiled_model(torch.randn(1, 128))
```

## Contributing

Contributions welcome! This project follows a modular architecture with core compiler components, optimized kernels, PyTorch integration, and performance tools.

## References

- [Tenstorrent TT-Buda Framework](https://github.com/tenstorrent/tt-buda)
- [PyTorch Custom Operators](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [uv Package Manager](https://github.com/astral-sh/uv)
