#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace tt_buda {
namespace kernels {

// High-performance matrix multiplication
torch::Tensor tensorcore_matmul(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias = torch::Tensor()) {
    
    auto output = torch::matmul(input, weight);
    
    if (bias.defined()) {
        output = output + bias;
    }
    
    return output;
}

// Batched matrix multiplication
torch::Tensor tensorcore_bmm(
    torch::Tensor batch1,
    torch::Tensor batch2) {
    
    return torch::bmm(batch1, batch2);
}

// Matrix multiplication with activation fusion
torch::Tensor tensorcore_matmul_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias = torch::Tensor()) {
    
    auto output = tensorcore_matmul(input, weight, bias);
    return torch::relu(output);
}

torch::Tensor tensorcore_matmul_gelu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias = torch::Tensor()) {
    
    auto output = tensorcore_matmul(input, weight, bias);
    return torch::gelu(output);
}

} // namespace kernels
} // namespace tt_buda

PYBIND11_MODULE(matmul, m) {
    m.doc() = "TT-Buda matrix multiplication kernels";
    
    m.def("tensorcore_matmul", &tt_buda::kernels::tensorcore_matmul,
          "TensorCore optimized matrix multiplication",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = torch::Tensor());
          
    m.def("tensorcore_bmm", &tt_buda::kernels::tensorcore_bmm,
          "TensorCore optimized batched matrix multiplication",
          pybind11::arg("batch1"),
          pybind11::arg("batch2"));
          
    m.def("tensorcore_matmul_relu", &tt_buda::kernels::tensorcore_matmul_relu,
          "TensorCore matmul with ReLU fusion",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = torch::Tensor());
          
    m.def("tensorcore_matmul_gelu", &tt_buda::kernels::tensorcore_matmul_gelu,
          "TensorCore matmul with GELU fusion",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = torch::Tensor());
} 