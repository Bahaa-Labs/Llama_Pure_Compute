#include "rmsswiglu.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace llama_pure {

namespace {

cublasLtHandle_t get_cublaslt_handle() {
    static cublasLtHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasLtCreate(&handle);
    }
    return handle;
}

void* get_cublaslt_workspace() {
    static void* workspace = nullptr;
    if (workspace == nullptr) {
        cudaMalloc(&workspace, 32 * 1024 * 1024); // 32MB Workspace for cuBLASLt heuristics
    }
    return workspace;
}

// Fast Warp Reduction using Register Shuffles
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fast Silu Activation
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

} // anonymous namespace

// 1. Optimized Vectorized RMSNorm Kernel using Warp Shuffles
template <typename scalar_t>
__global__ void rmsnorm_vectorized_128bit_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t*       __restrict__ output,
    const int hidden_dim,
    const float eps
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = block_size / 32;

    const scalar_t* in_row  = input  + blockIdx.x * hidden_dim;
    scalar_t*       out_row = output + blockIdx.x * hidden_dim;

    float sq_sum = 0.0f;
    const int vec_count = hidden_dim / 8; // 8 elements per 128-bit load (FP16/BF16)
    const uint4* in_vec = reinterpret_cast<const uint4*>(in_row);

    #pragma unroll
    for (int i = tid; i < vec_count; i += block_size) {
        uint4 raw = in_vec[i];
        const scalar_t* vals = reinterpret_cast<const scalar_t*>(&raw);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = static_cast<float>(vals[j]);
            sq_sum += v * v;
        }
    }

    // Warp level reduction
    sq_sum = warp_reduce_sum(sq_sum);

    extern __shared__ float sdata[];
    if (lane_id == 0) {
        sdata[warp_id] = sq_sum;
    }
    __syncthreads();

    // Block final reduction in warp 0
    float block_sq_sum = 0.0f;
    if (warp_id == 0) {
        block_sq_sum = (tid < num_warps) ? sdata[tid] : 0.0f;
        block_sq_sum = warp_reduce_sum(block_sq_sum);
    }

    if (tid == 0) {
        sdata[0] = block_sq_sum;
    }
    __syncthreads();

    const float inv_rms = rsqrtf(sdata[0] / static_cast<float>(hidden_dim) + eps);

    // Vectorized 128-bit write-back
    uint4* out_vec = reinterpret_cast<uint4*>(out_row);
    const uint4* w_vec = reinterpret_cast<const uint4*>(weight);

    #pragma unroll
    for (int i = tid; i < vec_count; i += block_size) {
        uint4 raw_in = in_vec[i];
        uint4 raw_w  = w_vec[i];
        uint4 raw_out;

        const scalar_t* in_arr = reinterpret_cast<const scalar_t*>(&raw_in);
        const scalar_t* w_arr  = reinterpret_cast<const scalar_t*>(&raw_w);
        scalar_t* out_arr      = reinterpret_cast<scalar_t*>(&raw_out);

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_arr[j] = static_cast<scalar_t>(
                static_cast<float>(in_arr[j]) * inv_rms * static_cast<float>(w_arr[j])
            );
        }
        out_vec[i] = raw_out;
    }
}

// 2. High-Speed Fully Coalesced SwiGLU Kernel
template <typename scalar_t>
__global__ void silu_mul_vectorized_kernel(
    const scalar_t* __restrict__ fused_proj,
    scalar_t*       __restrict__ output,
    const int num_tokens,
    const int inter_dim
) {
    // 2D grid strategy: x -> inter_dim (coalesced), y -> tokens
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col >= inter_dim || row >= num_tokens) return;

    const int stride_fused = 2 * inter_dim;
    const scalar_t* row_fused = fused_proj + row * stride_fused;
    scalar_t* row_out = output + row * inter_dim;

    float gate_val = static_cast<float>(row_fused[col]);
    float up_val   = static_cast<float>(row_fused[inter_dim + col]);

    float res = silu(gate_val) * up_val;
    row_out[col] = static_cast<scalar_t>(res);
}

torch::Tensor rmsswiglu_forward(
    const torch::Tensor &x,
    const torch::Tensor &rms_weight,
    const torch::Tensor &gate_w,
    const torch::Tensor &up_w,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    const int hidden_dim = x.size(-1);
    const int num_tokens = x.numel() / hidden_dim;

    torch::Tensor fused_w;
    int inter_dim = 0;

    if (up_w.defined() && up_w.numel() > 0) {
        inter_dim = gate_w.size(0);
        fused_w = torch::cat({gate_w, up_w}, 0).contiguous();
    } else {
        fused_w = gate_w;
        inter_dim = fused_w.size(0) / 2;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const at::ScalarType dtype = x.scalar_type();

    // Step 1: RMSNorm execution
    torch::Tensor x_normed = torch::empty_like(x);
    {
        constexpr int threads = 256;
        const dim3 grid(num_tokens);
        const dim3 block(threads);
        const size_t smem = (threads / 32) * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, dtype, "rmsnorm_vectorized_128bit", ([&] {
                rmsnorm_vectorized_128bit_kernel<scalar_t><<<grid, block, smem, stream>>>(
                    static_cast<const scalar_t*>(x.data_ptr()),
                    static_cast<const scalar_t*>(rms_weight.data_ptr()),
                    static_cast<scalar_t*>(x_normed.data_ptr()),
                    hidden_dim, eps
                );
            })
        );
    }

    // Step 2: High-speed cuBLASLt GEMM
    torch::Tensor fused_proj = torch::empty({num_tokens, 2 * inter_dim}, x.options());

    cublasLtHandle_t lt_handle = get_cublaslt_handle();
    
    cublasLtMatrixLayout_t layout_a, layout_b, layout_c;
    cublasLtMatmulDesc_t operation_desc;

    cudaDataType_t cuda_dtype = (dtype == at::ScalarType::Half) ? CUDA_R_16F : CUDA_R_16BF;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    cublasLtMatmulDescCreate(&operation_desc, compute_type, CUDA_R_32F);
    
    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
    cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));

    int m = 2 * inter_dim;
    int n = num_tokens;
    int k = hidden_dim;

    cublasLtMatrixLayoutCreate(&layout_a, cuda_dtype, k, m, k);
    cublasLtMatrixLayoutCreate(&layout_b, cuda_dtype, k, n, k);
    cublasLtMatrixLayoutCreate(&layout_c, cuda_dtype, m, n, m);

    float alpha = 1.0f, beta = 0.0f;

    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    size_t workspace_size = 32 * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size));

    cublasLtMatmulHeuristicResult_t heuristic_result[1];
    int returned_results = 0;
    cublasLtMatmulAlgoGetHeuristic(
        lt_handle, operation_desc, layout_a, layout_b, layout_c, layout_c, preference, 1, heuristic_result, &returned_results
    );

    if (returned_results > 0) {
        cublasLtMatmul(
            lt_handle, operation_desc, &alpha,
            fused_w.data_ptr(), layout_a,
            x_normed.data_ptr(), layout_b, &beta,
            fused_proj.data_ptr(), layout_c,
            fused_proj.data_ptr(), layout_c,
            &heuristic_result[0].algo,
            get_cublaslt_workspace(), workspace_size, stream
        );
    } else {
        cublasLtMatmul(
            lt_handle, operation_desc, &alpha,
            fused_w.data_ptr(), layout_a,
            x_normed.data_ptr(), layout_b, &beta,
            fused_proj.data_ptr(), layout_c,
            fused_proj.data_ptr(), layout_c,
            nullptr, get_cublaslt_workspace(), workspace_size, stream
        );
    }

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatmulDescDestroy(operation_desc);

    // Step 3: Vectorized Coalesced SwiGLU Pass
    torch::Tensor output = torch::empty({num_tokens, inter_dim}, x.options());
    constexpr int block_threads = 256;
    const dim3 grid_dim((inter_dim + block_threads - 1) / block_threads, num_tokens);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, dtype, "silu_mul_vectorized", ([&] {
            silu_mul_vectorized_kernel<scalar_t><<<grid_dim, block_threads, 0, stream>>>(
                static_cast<const scalar_t*>(fused_proj.data_ptr()),
                static_cast<scalar_t*>(output.data_ptr()),
                num_tokens, inter_dim
            );
        })
    );

    auto out_shape = x.sizes().vec();
    out_shape.back() = inter_dim;
    return output.view(out_shape);
}

} // namespace llama_pure