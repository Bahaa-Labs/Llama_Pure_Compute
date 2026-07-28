#ifndef LLAMA_PURE_COMPUTE_UTILS_H_
#define LLAMA_PURE_COMPUTE_UTILS_H_

#include <chrono>
#include <concepts>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
#include <cuda_bf16.h>
#endif
#endif

namespace llama_pure {
// 1- Target Exectuon & Inline Qualifiers
#if defined(__CUDACC__) || defined(__HIPCC__)
    #define HOST_DEVICE __host__ __device__
    #define DEVICE __device__
    #define HOST __host__
    #define FORCE_INLINE __forceinline__
#else
    #define HOST_DEVICE
    #define DEVICE
    #define HOST
    #define FORCE_INLINE inline
#endif

// 2- Strict CUDA Error Handling
#ifdef __CUDACC__
inline void checkCuda(cudaError_t result, const char* const func, const char* const file, const int line){
    if(result != cudaSuccess){
        std::ostringstream ss;
        ss << "[Llama_Pure Error] CUDA API failed at: " << file 
            << ":" << line << " in " << func << 
                " -> (" << static_cast<int>(result) << ") " 
                    << cudaGetErrorString(result);
        throw std::runtime_error(ss.str());
    }
}
inline void checkCudaKernel(const char* const file, const int line){
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::ostringstream ss;
        ss << "[Llama-Pure Error] Kernel Launch error at: " << file
            << ":" << line << " -> (" << static_cast<int>(err) << ") " 
                << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}

#define CUDA_CHECK(val) ::llama_pure::checkCuda((val), #val, __FILE__, __LINE__)
#define CUDA_CHECK_KERNEL() ::llama_pure::checkCudaKernel(__FILE__, __LINE__)
inline void syncAndCheckDevice(const char* const file, const int line){
    CUDA_CHECK(cudaDeviceSynchronize());
    checkCudaKernel(file, line);
}
#define CUDA_SYNC_CHECK() ::llama_pure::syncAndCheckDevice(__FILE__, __LINE__)
#endif  // __CUDACC__

// 3- Tensor & Grid Arithmetic
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
HOST_DEVICE FORCE_INLINE constexpr T cdiv(T a, T b){
    return (a + b - static_cast<T>(1)) / b;
}
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
HOST_DEVICE FORCE_INLINE constexpr T align_up(T value, T alignment){
    return (value + alignment - static_cast<T>(1)) & ~(alignment - static_cast<T>(1));
}

// Benchmarking & Profiling Timer(Host Side)
class Timer{
    private:
        std::chrono::high_resolution_clock::time_point start_time_;

    public:
        Timer() { start(); }
        void start(){
            start_time_ = std::chrono::high_resolution_clock::now();
        }
        double elapsed_ms() const {
            auto end_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end_time - start_time_).count();
        }
        double elapsed_us() const {
            auto end_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::micro>(end_time - start_time_).count();
        }
};

// CUDA Event-base GPU Timer
#ifdef __CUDACC__
class GpuTimer{
    private:
        cudaEvent_t start_{};
        cudaEvent_t stop_{};
    public:
        GpuTimer(){
            CUDA_CHECK(cudaEventCreate(&start_));
            CUDA_CHECK(cudaEventCreate(&stop_));
        }
        ~GpuTimer(){
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);

        }
        void start(cudaStream_t stream = 0){
            CUDA_CHECK(cudaEventRecord(start_, stream));
        }
        void stop(cudaStream_t stream = 0){
            CUDA_CHECK(cudaEventRecord(stop_, stream));
        }
        float elapsed_ms(){
            CUDA_CHECK(cudaEventSynchronize(stop_));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
            return ms;
        }
};
#endif // __CUDACC__
}
#endif // Llama_Pure_Compute_Utils_H_
