#ifndef LLAMA_PURE_COMPUTE_RMSSWIGLU_H_
#define LLAMA_PURE_COMPUTE_RMSSWIGLU_H_

#include <cuda_runtime.h>
#include <torch/all.h>

namespace llama_pure {

torch::Tensor rmsswiglu_forward(
    const torch::Tensor &x,
    const torch::Tensor &rms_weight,
    const torch::Tensor &gate_w,
    const torch::Tensor &up_w,
    float eps = 1e-5f
);

} // namespace llama_pure

#endif // LLAMA_PURE_COMPUTE_RMSSWIGLU_H_