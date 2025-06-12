"""
Custom operators module for TT-Buda Kernels.
"""

from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod

from ..utils.logging import get_logger

logger = get_logger(__name__)


class CustomOperator(ABC):
    """Base class for custom operators."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass implementation."""
        pass
    
    @abstractmethod
    def backward(self, *args: Any, **kwargs: Any) -> Any:
        """Backward pass implementation."""
        pass


class OperatorRegistry:
    """Registry for custom operators."""
    
    def __init__(self):
        self._operators: Dict[str, CustomOperator] = {}
        self.logger = get_logger(__name__)
    
    def register(self, operator: CustomOperator) -> None:
        """Register a custom operator."""
        self._operators[operator.name] = operator
        self.logger.info(f"Registered operator: {operator.name}")
    
    def get(self, name: str) -> Optional[CustomOperator]:
        """Get an operator by name."""
        return self._operators.get(name)
    
    def list_operators(self) -> List[str]:
        """List all registered operators."""
        return list(self._operators.keys()) 