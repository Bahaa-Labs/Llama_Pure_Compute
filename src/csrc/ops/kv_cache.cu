#include "../include/kv_cache.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cstdint>

namespace llama_pure_compute {

// Vectorized 128-bit Scatter Kernel for [B, H, S, D] contiguous cache
template <typename vec_t>
__global__ void update_kv_cache_vectorized_128bit_kernel(
    const vec_t* __restrict__ key_src,
    const vec_t* __restrict__ value_src,
    vec_t* __restrict__ key_cache,
    vec_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_kv_heads,
    int vec_head_dim,
    int max_seq_len
){
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int64_t slot = slot_mapping ? slot_mapping[token_idx] : token_idx;
    if (slot < 0) return;

    // Source offset: [num_tokens, num_kv_heads, vec_head_dim]
    int src_head_offset = token_idx * (num_kv_heads * vec_head_dim) + head_idx * vec_head_dim;

    // Target 4D Cache [B, H, S, D] stride offset:
    int batch_idx = slot / max_seq_len;
    int pos = slot % max_seq_len;
    int cache_head_offset = (batch_idx * num_kv_heads * max_seq_len + head_idx * max_seq_len + pos) * vec_head_dim;

    for (int tid = threadIdx.x; tid < vec_head_dim; tid += blockDim.x) {
        key_cache[cache_head_offset + tid] = key_src[src_head_offset + tid];
        value_cache[cache_head_offset + tid] = value_src[src_head_offset + tid];
    }
}

// Fallback Scalar Kernel for [B, H, S, D] contiguous cache
template <typename scalar_t>
__global__ void update_kv_cache_scalar_kernel(
    const scalar_t* __restrict__ key_src,
    const scalar_t* __restrict__ value_src,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_kv_heads,
    int head_dim,
    int max_seq_len
){
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int64_t slot = slot_mapping ? slot_mapping[token_idx] : token_idx;
    if (slot < 0) return;

    int src_head_offset = token_idx * (num_kv_heads * head_dim) + head_idx * head_dim;

    int batch_idx = slot / max_seq_len;
    int pos = slot % max_seq_len;
    int cache_head_offset = (batch_idx * num_kv_heads * max_seq_len + head_idx * max_seq_len + pos) * head_dim;

    for (int tid = threadIdx.x; tid < head_dim; tid += blockDim.x) {
        key_cache[cache_head_offset + tid] = key_src[src_head_offset + tid];
        value_cache[cache_head_offset + tid] = value_src[src_head_offset + tid];
    }
}

void launch_update_kv_cache_cuda(
    const void* key_src,
    const void* value_src,
    void* key_cache,
    void* value_cache,
    const int64_t* slot_mapping_ptr,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    at::ScalarType dtype,
    cudaStream_t stream
){
    if (num_tokens <= 0) return;

    dim3 grid(num_tokens, num_kv_heads);

    bool is_aligned = (reinterpret_cast<uintptr_t>(key_src) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(value_src) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(key_cache) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(value_cache) % 16 == 0);

    size_t element_size = c10::elementSize(dtype);
    size_t row_bytes = head_dim * element_size;

    if ((row_bytes % 16 == 0) && is_aligned) {
        int vec_head_dim = row_bytes / 16;
        int block_size = std::min(1024, ((vec_head_dim + 31) / 32) * 32);

        update_kv_cache_vectorized_128bit_kernel<uint4><<<grid, block_size, 0, stream>>>(
            reinterpret_cast<const uint4*>(key_src),
            reinterpret_cast<const uint4*>(value_src),
            reinterpret_cast<uint4*>(key_cache),
            reinterpret_cast<uint4*>(value_cache),
            slot_mapping_ptr,
            num_kv_heads,
            vec_head_dim,
            max_seq_len
        );
    } else {
        int block_size = std::min(1024, ((head_dim + 31) / 32) * 32);

        switch (dtype) {
            case at::ScalarType::Float:
                update_kv_cache_scalar_kernel<float><<<grid, block_size, 0, stream>>>(
                    reinterpret_cast<const float*>(key_src),
                    reinterpret_cast<const float*>(value_src),
                    reinterpret_cast<float*>(key_cache),
                    reinterpret_cast<float*>(value_cache),
                    slot_mapping_ptr, num_kv_heads, head_dim, max_seq_len);
                break;

            case at::ScalarType::Half:
                update_kv_cache_scalar_kernel<__half><<<grid, block_size, 0, stream>>>(
                    reinterpret_cast<const __half*>(key_src),
                    reinterpret_cast<const __half*>(value_src),
                    reinterpret_cast<__half*>(key_cache),
                    reinterpret_cast<__half*>(value_cache),
                    slot_mapping_ptr, num_kv_heads, head_dim, max_seq_len);
                break;

            case at::ScalarType::BFloat16:
                update_kv_cache_scalar_kernel<__nv_bfloat16><<<grid, block_size, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(key_src),
                    reinterpret_cast<const __nv_bfloat16*>(value_src),
                    reinterpret_cast<__nv_bfloat16*>(key_cache),
                    reinterpret_cast<__nv_bfloat16*>(value_cache),
                    slot_mapping_ptr, num_kv_heads, head_dim, max_seq_len);
                break;

            default:
                TORCH_CHECK(false, "LlamaPureComputeError: Unsupported dtype for update_kv_cache");
        }
    }
    C10_CUDA_CHECK(cudaGetLastError());
}

void update_kv_cache(
    torch::Tensor& key_src,
    torch::Tensor& value_src,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    const c10::optional<torch::Tensor>& slot_mapping
){
    TORCH_CHECK(key_src.is_cuda(), "key_src must be on CUDA");
    TORCH_CHECK(value_src.is_cuda(), "value_src must be on CUDA");
    TORCH_CHECK(key_cache.is_cuda(), "key_cache must be on CUDA");
    TORCH_CHECK(value_cache.is_cuda(), "value_cache must be on CUDA");

    TORCH_CHECK(key_src.is_contiguous(), "key_src must be contiguous");
    TORCH_CHECK(value_src.is_contiguous(), "value_src must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");

    int head_dim = key_src.size(-1);
    int num_kv_heads = key_src.size(-2);
    int num_tokens = key_src.numel() / (num_kv_heads * head_dim);

    int max_seq_len = key_cache.dim() == 4 ? key_cache.size(2) : key_cache.size(1);

    const int64_t* slot_mapping_ptr = nullptr;
    if (slot_mapping.has_value() && slot_mapping->defined()) {
        const torch::Tensor& slot_tensor = slot_mapping.value();
        TORCH_CHECK(slot_tensor.is_cuda(), "slot_mapping must be on CUDA");
        TORCH_CHECK(slot_tensor.is_contiguous(), "slot_mapping must be contiguous");
        slot_mapping_ptr = slot_tensor.data_ptr<int64_t>();
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_update_kv_cache_cuda(
        key_src.data_ptr(),
        value_src.data_ptr(),
        key_cache.data_ptr(),
        value_cache.data_ptr(),
        slot_mapping_ptr,
        num_tokens,
        num_kv_heads,
        head_dim,
        max_seq_len,
        key_src.scalar_type(),
        stream
    );
}

} // namespace llama_pure_compute