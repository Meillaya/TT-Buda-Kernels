"""Compiler passes for TT-Buda Kernels."""

from typing import List, Any
from torch.fx import GraphModule

class OptimizationPass:
    """Base optimization pass."""
    pass

class PassManager:
    """Manages compilation passes."""
    
    def __init__(self):
        self.passes: List[Any] = []
    
    def add_pass(self, pass_obj: Any) -> None:
        """Add a pass to the manager."""
        self.passes.append(pass_obj)
    
    def run(self, graph: GraphModule) -> GraphModule:
        """Run all passes on the graph."""
        for pass_obj in self.passes:
            if hasattr(pass_obj, 'run'):
                graph = pass_obj.run(graph)
        return graph 