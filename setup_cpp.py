"""
Setup script for TT-Buda Kernels.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages, Extension
from pybind11 import get_cmake_dir, get_include
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Read version from package
sys.path.insert(0, "src")
try:
    from tt_buda import __version__
except ImportError:
    __version__ = "0.1.0"

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# C++ extensions for kernels
ext_modules = [
    Pybind11Extension(
        "tt_buda._C.kernels",
        sources=[
            "src/tt_buda/kernels/cpp/matmul.cpp",
            "src/tt_buda/kernels/cpp/conv.cpp",
            "src/tt_buda/kernels/cpp/attention.cpp",
        ],
        include_dirs=[
            get_include(),
            "src/tt_buda/kernels/cpp/include",
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"{}"'.format(__version__))],
    ),
]

setup(
    name="tt-buda-kernels",
    version=__version__,
    author="TT-Buda Kernels Contributors",
    author_email="contributors@tt-buda-kernels.org",
    description="Open-source AI software stack with Tenstorrent backend support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TT-Buda-Kernels",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/TT-Buda-Kernels/issues",
        "Documentation": "https://tt-buda-kernels.readthedocs.io",
        "Source Code": "https://github.com/yourusername/TT-Buda-Kernels",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pybind11>=2.10.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "benchmark": [
            "memory-profiler>=0.61.0",
            "line-profiler>=4.1.0",
            "torch-tb-profiler>=0.4.0",
        ],
        "dashboard": [
            "streamlit>=1.28.0",
            "plotly>=5.17.0",
            "pandas>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tt-buda=tt_buda.cli:main",
            "tt-compile=tt_buda.compiler.cli:compile_model",
            "tt-benchmark=tt_buda.benchmarks.cli:benchmark",
            "tt-dashboard=tt_buda.dashboard.cli:start_dashboard",
        ],
    },
    zip_safe=False,
    include_package_data=True,
) 