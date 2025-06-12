FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    wget \
    vim \
    htop \
    python3 \
    python3-dev \
    python3-pip \
    libc6-dev \
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY pyproject.toml .
COPY setup.py .
COPY README.md .

# Create virtual environment and install dependencies
RUN uv venv .venv
ENV PATH="/workspace/.venv/bin:${PATH}"

# Install PyTorch with CUDA support
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
RUN uv pip install -e ".[dev,benchmark,dashboard]"

# Copy source code
COPY src/ src/
COPY tests/ tests/
COPY docs/ docs/

# Install the package in development mode
RUN uv pip install -e .

# Create directories for development
RUN mkdir -p /workspace/cache /workspace/output /workspace/logs

# Set up environment for development
EXPOSE 8000 8501 6006

# Default command
CMD ["/bin/bash"] 