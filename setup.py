"""
Simplified setup.py for TT-Buda Kernels (Python-only for development)
"""

from setuptools import setup, find_packages

setup(
    name="tt-buda-kernels",
    version="0.1.0",
    author="TT-Buda Kernels Contributors",
    author_email="contributors@tt-buda-kernels.org",
    description="Open-Source AI Software Stack inspired by TT-Buda",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tt-buda-kernels/tt-buda-kernels",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pybind11>=2.10.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.5.0",
        "psutil>=5.9.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
        "rich>=12.0.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "streamlit>=1.20.0",
        "plotly>=5.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "flake8>=5.0.0",
            "pre-commit>=2.20.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "benchmarks": [
            "transformers>=4.20.0",
            "diffusers>=0.15.0",
            "accelerate>=0.20.0",
            "datasets>=2.10.0",
        ],
        "gpu": [
            "triton>=2.0.0",
            "cupy-cuda11x>=11.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tt-buda=tt_buda.cli:main",
            "tt-buda-bench=tt_buda.benchmarks.cli:main",
            "tt-buda-profile=tt_buda.tools.profiler:main",
            "tt-buda-dashboard=tt_buda.dashboard.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine-learning pytorch tenstorrent compiler optimization",
    include_package_data=True,
    zip_safe=False,
) 