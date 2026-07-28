#include "rope.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>

namespace llama_pure {

// Dedicated Query Rotation Kernel
template <typename scalar_t>
__global__ void rope_q_kernel(
    scalar_t* __restrict__ q,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const int64_t* __restrict__ pos_ids,
    int num_tokens,
    int num_heads,
    int head_dim,
    int half_rotary
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * num_heads * half_rotary;

    if (tid >= total_elements) return;

    int d = tid % half_rotary;
    int head_idx = (tid / half_rotary) % num_heads;
    int token_idx = tid / (num_heads * half_rotary);

    int pos = pos_ids ? static_cast<int>(pos_ids[token_idx]) : token_idx;

    int base_offset = token_idx * (num_heads * head_dim) + head_idx * head_dim;
    int q0_idx = base_offset + d;
    int q1_idx = base_offset + d + half_rotary;

    int freq_offset = pos * half_rotary + d;

    float q0 = static_cast<float>(q[q0_idx]);
    float q1 = static_cast<float>(q[q1_idx]);
    float c  = static_cast<float>(cos[freq_offset]);
    float s  = static_cast<float>(sin[freq_offset]);

    q[q0_idx] = static_cast<scalar_t>(q0 * c - q1 * s);
    q[q1_idx] = static_cast<scalar_t>(q0 * s + q1 * c);
}

// Dedicated Key Rotation Kernel (Handles GQA)
template <typename scalar_t>
__global__ void rope_k_kernel(
    scalar_t* __restrict__ k,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const int64_t* __restrict__ pos_ids,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int half_rotary
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * num_kv_heads * half_rotary;

    if (tid >= total_elements) return;

    int d = tid % half_rotary;
    int kv_head_idx = (tid / half_rotary) % num_kv_heads;
    int token_idx = tid / (num_kv_heads * half_rotary);

    int pos = pos_ids ? static_cast<int>(pos_ids[token_idx]) : token_idx;

    int base_offset = token_idx * (num_kv_heads * head_dim) + kv_head_idx * head_dim;
    int k0_idx = base_offset + d;
    int k1_idx = base_offset + d + half_rotary;

    int freq_offset = pos * half_rotary + d;

    float k0 = static_cast<float>(k[k0_idx]);
    float k1 = static_cast<float>(k[k1_idx]);
    float c  = static_cast<float>(cos[freq_offset]);
    float s  = static_cast<float>(sin[freq_offset]);

    k[k0_idx] = static_cast<scalar_t>(k0 * c - k1 * s);
    k[k1_idx] = static_cast<scalar_t>(k0 * s + k1 * c);
}

// C++ CUDA Kernel Launcher Function
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
) {
    int half_rotary = rotary_dim / 2;
    const int threads_per_block = 256;

    int q_work_items = num_tokens * num_heads * half_rotary;
    int k_work_items = num_tokens * num_kv_heads * half_rotary;

    int q_blocks = (q_work_items + threads_per_block - 1) / threads_per_block;
    int k_blocks = (k_work_items + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "rope_cuda_kernel",
        ([&] {
            rope_q_kernel<scalar_t><<<q_blocks, threads_per_block, 0, stream>>>(
                reinterpret_cast<scalar_t*>(q_ptr),
                reinterpret_cast<const scalar_t*>(cos_ptr),
                reinterpret_cast<const scalar_t*>(sin_ptr),
                pos_ids_ptr,
                num_tokens,
                num_heads,
                head_dim,
                half_rotary
            );

            rope_k_kernel<scalar_t><<<k_blocks, threads_per_block, 0, stream>>>(
                reinterpret_cast<scalar_t*>(k_ptr),
                reinterpret_cast<const scalar_t*>(cos_ptr),
                reinterpret_cast<const scalar_t*>(sin_ptr),
                pos_ids_ptr,
                num_tokens,
                num_kv_heads,
                head_dim,
                half_rotary
            );
        })
    );
    C10_CUDA_CHECK(cudaGetLastError());
}

// PyTorch Tensor API Dispatcher
std::tuple<torch::Tensor, torch::Tensor> rope_forward(
    const torch::Tensor &q,
    const torch::Tensor &k,
    const torch::Tensor &cos,
    const torch::Tensor &sin,
    const std::optional<at::Tensor> &position_ids
) {
    TORCH_CHECK(q.is_cuda(), "Query tensor q must be on CUDA");
    TORCH_CHECK(k.is_cuda(), "Key tensor k must be on CUDA");
    TORCH_CHECK(cos.is_cuda(), "cos tensor must be on CUDA");
    TORCH_CHECK(sin.is_cuda(), "sin tensor must be on CUDA");

    TORCH_CHECK(q.is_contiguous(), "Query tensor q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Key tensor k must be contiguous");

    torch::Tensor q_out = q;
    torch::Tensor k_out = k;

    int head_dim = q.size(-1);
    int num_heads = q.size(-2);
    int num_kv_heads = k.size(-2);
    int num_tokens = q.numel() / (num_heads * head_dim);
    int rotary_dim = head_dim;

    const int64_t *pos_ids_ptr = nullptr;
    if (position_ids.has_value() && position_ids->defined()) {
        const torch::Tensor &pos_tensor = position_ids.value();
        TORCH_CHECK(pos_tensor.is_cuda(), "position_ids must be on CUDA");
        TORCH_CHECK(pos_tensor.is_contiguous(), "position_ids must be contiguous");

        pos_ids_ptr = pos_tensor.data_ptr<int64_t>();
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_rope_cuda(
        q_out.data_ptr(),
        k_out.data_ptr(),
        cos.data_ptr(),
        sin.data_ptr(),
        pos_ids_ptr,
        num_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        q.scalar_type(),
        stream
    );
    return std::make_tuple(q_out, k_out);
}

} // namespace llama_pure