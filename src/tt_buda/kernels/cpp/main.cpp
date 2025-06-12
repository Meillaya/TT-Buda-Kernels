#include <pybind11/pybind11.h>
#include <torch/extension.h>

// Forward declarations for submodules
void init_attention(pybind11::module &);
void init_matmul(pybind11::module &);

PYBIND11_MODULE(kernels, m) {
    m.doc() = "TT-Buda optimized kernels for Tenstorrent hardware";
    
    // Create submodules
    auto attention_module = m.def_submodule("attention", "Attention kernels");
    auto matmul_module = m.def_submodule("matmul", "Matrix multiplication kernels");
    
    // Initialize submodules
    init_attention(attention_module);
    init_matmul(matmul_module);
    
    // Version info
    m.attr("__version__") = "0.1.0";
} 