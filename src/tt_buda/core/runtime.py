"""Runtime support for TT-Buda Kernels."""

from typing import Any
from dataclasses import dataclass

@dataclass
class RuntimeConfig:
    """Runtime configuration."""
    backend: str = "tenstorrent"

class Runtime:
    """Runtime execution engine."""
    
    def __init__(self, config: RuntimeConfig):
        self.config = config 