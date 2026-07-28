#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/types.h>
#include <c10/core/ScalarType.h>
#include <tuple>
#include <optional>
#include <stdexcept>
#include <string>
#include <torch/extension.h>
#include "../include/kv_cache.h"

namespace py = pybind11;

// CUDA Kernels Forward Declarations
namespace llama_pure {

std::tuple<at::Tensor, at::Tensor> rope_forward(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &cos,
    const at::Tensor &sin,                     
    const c10::optional<at::Tensor> &position_ids 
);

at::Tensor rmsswiglu_forward(
    const at::Tensor& x, 
    const at::Tensor& rms_weight,
    const at::Tensor& gate_w,
    const at::Tensor& up_w,
    float eps
);

} // namespace llama_pure

namespace llama_pure_compute {

// CUDA PyTorch Dispatcher declared in kv_cache.cu
void update_kv_cache(
    torch::Tensor& key_src,
    torch::Tensor& value_src,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    const c10::optional<torch::Tensor>& slot_mapping
);

} // namespace llama_pure_compute

namespace llama_pure {

// Tensor Validation helper
inline void validate_cuda_tensor(const at::Tensor& tensor, const std::string& name) {
    if(!tensor.defined()){
        throw std::invalid_argument("LlamaPureComputeError: " + name + " Tensor is undefined.");
    }
    if(!tensor.is_cuda()){
        throw std::invalid_argument("LlamaPureComputeError: " + name + " must be on CUDA device.");
    }
    if(!tensor.is_contiguous()){
        throw std::invalid_argument("LlamaPureComputeError: " + name + " must be memory-contiguous.");
    }
}

// Binding Wrapper Implementations
std::tuple<at::Tensor, at::Tensor> rope_forward_binding(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const c10::optional<at::Tensor>& position_ids 
){
    validate_cuda_tensor(q, "q");
    validate_cuda_tensor(k, "k");
    validate_cuda_tensor(cos, "cos");
    validate_cuda_tensor(sin, "sin");

    if(position_ids.has_value()){
        validate_cuda_tensor(position_ids.value(), "position_ids");
    }
    if(q.scalar_type() != k.scalar_type()){
        throw std::invalid_argument("LlamaPureComputeError: Query (q) and Key (k) must have same scalar type.");
    }

    py::gil_scoped_release release;
    return llama_pure::rope_forward(q, k, cos, sin, position_ids);
}

at::Tensor rmsswiglu_forward_binding(
    const at::Tensor& x,
    const at::Tensor& rms_weight,
    const at::Tensor& gate_w,
    const at::Tensor& up_w,
    float eps
){
    validate_cuda_tensor(x, "x");
    validate_cuda_tensor(rms_weight, "rms_weight");
    validate_cuda_tensor(gate_w, "gate_w");
    validate_cuda_tensor(up_w, "up_w");

    if(eps <= 0.0f){
        throw std::invalid_argument("LlamaPureComputeError: eps must be strictly positive.");
    }

    py::gil_scoped_release release;          
    return llama_pure::rmsswiglu_forward(x, rms_weight, gate_w, up_w, eps);
}

class PyKVCacheManager {
private:
    llama_pure_compute::KVCacheManager manager_;

public:
    explicit PyKVCacheManager(const llama_pure_compute::KVCacheConfig& config)
        : manager_(config) {}

    void update(size_t layer_idx, size_t batch_idx, at::Tensor k, at::Tensor v, size_t pos) {
        if (!k.is_contiguous() || !v.is_contiguous()) {
            throw std::invalid_argument("LlamaPureComputeError: Input tensors k and v must be contiguous.");
        }
        if (k.scalar_type() != torch::kFloat32 || v.scalar_type() != torch::kFloat32) {
            throw std::invalid_argument("LlamaPureComputeError: Input tensors k and v must be float32.");
        }
        if (k.is_cuda() || v.is_cuda()) {
            throw std::invalid_argument("LlamaPureComputeError: CPU KVCacheManager expects host (CPU) tensors.");
        }

        const float* k_ptr = k.data_ptr<float>();
        const float* v_ptr = v.data_ptr<float>();

        py::gil_scoped_release release;
        manager_.update(layer_idx, batch_idx, k_ptr, v_ptr, pos);
    }

    at::Tensor get_k_tensor(size_t layer_idx, size_t batch_idx, size_t pos = 0) {
        const float* ptr = manager_.get_k_ptr(layer_idx, batch_idx, pos);
        auto config = manager_.config();

        int64_t seq_len = static_cast<int64_t>(config.max_seq_len - pos);
        int64_t num_heads = static_cast<int64_t>(config.num_kv_heads);
        int64_t head_dim = static_cast<int64_t>(config.head_dim);

        return torch::from_blob(
            const_cast<float*>(ptr),
            {seq_len, num_heads, head_dim},
            torch::kFloat32
        );
    }

    at::Tensor get_v_tensor(size_t layer_idx, size_t batch_idx, size_t pos = 0) {
        const float* ptr = manager_.get_v_ptr(layer_idx, batch_idx, pos);
        auto config = manager_.config();

        int64_t seq_len = static_cast<int64_t>(config.max_seq_len - pos);
        int64_t num_heads = static_cast<int64_t>(config.num_kv_heads);
        int64_t head_dim = static_cast<int64_t>(config.head_dim);

        return torch::from_blob(
            const_cast<float*>(ptr),
            {seq_len, num_heads, head_dim},
            torch::kFloat32
        );
    }

    void reset() {
        py::gil_scoped_release release;
        manager_.reset();
    }

    void reset_batch(size_t batch_idx) {
        py::gil_scoped_release release;
        manager_.reset_batch(batch_idx);
    }

    size_t get_current_seq_len(size_t batch_idx) const {
        return manager_.get_current_seq_len(batch_idx);
    }

    const llama_pure_compute::KVCacheConfig& config() const {
        return manager_.config();
    }
};

} // namespace llama_pure

// Pybind11 Module Declaration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Llama_Pure_Compute High-Performance CUDA Inference Engine";

    // CUDA RoPE Op Binding
    m.def(
        "rope_forward",
        &llama_pure::rope_forward_binding,
        "Apply Rotary Position Embedding (RoPE) forward pass (CUDA)",
        py::arg("q"),
        py::arg("k"),
        py::arg("cos"),
        py::arg("sin"),
        py::arg("position_ids") = py::none()
    );

    // CUDA SwiGLU Op Binding
    m.def(
        "rmsswiglu_forward",
        &llama_pure::rmsswiglu_forward_binding,
        "Fused RMSNorm + SwiGLU Forward Pass (CUDA)",
        py::arg("x"),
        py::arg("rms_weight"),
        py::arg("gate_w"),
        py::arg("up_w"),
        py::arg("eps") = 1e-5f
    );

    // CUDA KV Cache Scatter Op Binding (Exposed to ops.py)
    m.def(
        "update_kv_cache",
        &llama_pure_compute::update_kv_cache,
        "Scatter-updates key and value tensors into static KV cache storage (CUDA)",
        py::arg("key_src"),
        py::arg("value_src"),
        py::arg("key_cache"),
        py::arg("value_cache"),
        py::arg("slot_mapping") = py::none()
    );

    // CPU KVCacheConfig Struct Binding
    py::class_<llama_pure_compute::KVCacheConfig>(m, "KVCacheConfig")
        .def(py::init<size_t, size_t, size_t, size_t, size_t>(),
             py::arg("num_layers") = 32,
             py::arg("num_kv_heads") = 8,
             py::arg("head_dim") = 128,
             py::arg("max_seq_len") = 4096,
             py::arg("max_batch_size") = 1)
        .def_readwrite("num_layers", &llama_pure_compute::KVCacheConfig::num_layers)
        .def_readwrite("num_kv_heads", &llama_pure_compute::KVCacheConfig::num_kv_heads)
        .def_readwrite("head_dim", &llama_pure_compute::KVCacheConfig::head_dim)
        .def_readwrite("max_seq_len", &llama_pure_compute::KVCacheConfig::max_seq_len)
        .def_readwrite("max_batch_size", &llama_pure_compute::KVCacheConfig::max_batch_size);

    // CPU KVCacheManager Binding
    py::class_<llama_pure::PyKVCacheManager>(m, "KVCacheManager")
        .def(py::init<const llama_pure_compute::KVCacheConfig &>(), py::arg("config"))
        .def("update", &llama_pure::PyKVCacheManager::update,
             py::arg("layer_idx"), py::arg("batch_idx"), py::arg("k"), py::arg("v"), py::arg("pos"))
        .def("get_k_tensor", &llama_pure::PyKVCacheManager::get_k_tensor,
             py::arg("layer_idx"), py::arg("batch_idx"), py::arg("pos") = 0)
        .def("get_v_tensor", &llama_pure::PyKVCacheManager::get_v_tensor,
             py::arg("layer_idx"), py::arg("batch_idx"), py::arg("pos") = 0)
        .def("reset", &llama_pure::PyKVCacheManager::reset)
        .def("reset_batch", &llama_pure::PyKVCacheManager::reset_batch, py::arg("batch_idx"))
        .def("get_current_seq_len", &llama_pure::PyKVCacheManager::get_current_seq_len, py::arg("batch_idx"))
        .def("config", &llama_pure::PyKVCacheManager::config);
}
