#ifndef LLAMA_PURE_COMPUTE_ROPE_H_
#define LLAMA_PURE_COMPUTE_ROPE_H_

#include <cuda_runtime.h>
#include <torch/all.h>
#include <tuple>
#include "utils.h"

namespace llama_pure {

/**
    Rotary Position Embedding (RoPE) forward pass for Query and Key tensors.

    Mathematical Formulation (2D Coordinate Rotation):
    For pair (x0, x1) and angle theta:
      x0_out = x0 * cos(theta) - x1 * sin(theta)
     x1_out = x0 * sin(theta) + x1 * cos(theta)
 */
std::tuple<torch::Tensor, torch::Tensor> rope_forward(
    const torch::Tensor &q,
    const torch::Tensor &k,
    const torch::Tensor &cos,
    const torch::Tensor &sin,
    const c10::optional<torch::Tensor> &position_ids = c10::nullopt
);
void launch_rope_cuda(
    void *q_ptr,
    void *k_ptr,
    const void *cos_ptr,
    const void *sin_ptr,
    const int64_t *pos_ids_ptr,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    at::ScalarType dtype,
    cudaStream_t stream
);

} // namespace llama_pure

#endif // LLAMA_PURE_COMPUTE_ROPE_H_