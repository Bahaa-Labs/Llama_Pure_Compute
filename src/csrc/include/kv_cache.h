#ifndef KV_CACHE_H_
#define KV_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <mutex>
#include <algorithm>
#include <new>
#include <utility>

namespace llama_pure_compute {

// Memory Alignment Vector Allocator for SIMD optimization.
template <typename T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;
    
    template <typename U> 
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
        std::size_t bytes = n * sizeof(T);
#if defined(_MSC_VER) || defined(__MINGW32__)
        ptr = _aligned_malloc(bytes, Alignment);
        if (!ptr) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, Alignment, bytes) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
#if defined(_MSC_VER) || defined(__MINGW32__)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <typename U>
    void destroy(U* p) noexcept {
        p->~U();
    }
};

template <typename T, typename U, std::size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return true; }

template <typename T, typename U, std::size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return false; }

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

// Configuration parameters for KV Cache
struct KVCacheConfig {
    size_t num_layers = 32;
    size_t num_kv_heads = 8;
    size_t head_dim = 128;
    size_t max_seq_len = 4096;
    size_t max_batch_size = 1;
};

// Key-Value Cache Manager for Transformer Inference.
class KVCacheManager {
private:
    [[nodiscard]] inline size_t get_buffer_offset(size_t layer_idx, size_t batch_idx, size_t pos) const noexcept {
        return (layer_idx * layer_stride_) + (batch_idx * batch_stride_) + (pos * seq_stride_);
    }

    inline void validate_indices(size_t layer_idx, size_t batch_idx, size_t pos) const {
        if (layer_idx >= config_.num_layers) {
            throw std::out_of_range("KV Cache: Layer index out of bounds");
        }
        if (batch_idx >= config_.max_batch_size) {
            throw std::out_of_range("KV Cache: Batch index out of bounds");
        }
        if (pos >= config_.max_seq_len) {
            throw std::out_of_range("KV Cache: Sequence position exceeds max_seq_len");
        }
    }

    KVCacheConfig config_;
    size_t layer_stride_;
    size_t seq_stride_;
    size_t batch_stride_;

    // Contiguous 64-byte aligned vector storage for cache tensors
    AlignedVector<float> k_cache_;
    AlignedVector<float> v_cache_;

    // Sequence tracking per batch item
    std::vector<size_t> current_seq_lens_;
    mutable std::mutex seq_mutex_;

public:
    explicit KVCacheManager(const KVCacheConfig& config)
        : config_(config),
          layer_stride_(config.max_batch_size * config.max_seq_len * config.num_kv_heads * config.head_dim),
          seq_stride_(config.num_kv_heads * config.head_dim),
          batch_stride_(config.max_seq_len * seq_stride_)
    {
        if (config_.max_seq_len == 0 || config_.num_layers == 0 || config_.num_kv_heads == 0 || config_.head_dim == 0) {
            throw std::invalid_argument("KVCacheConfig parameters must be greater than 0");
        }
        size_t total_elements_per_cache = config_.num_layers * layer_stride_;
        
        k_cache_.resize(total_elements_per_cache, 0.0f);
        v_cache_.resize(total_elements_per_cache, 0.0f);
        current_seq_lens_.resize(config_.max_batch_size, 0);
    }

    ~KVCacheManager() = default;

    // Non-copyable for performance & safety
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;

    // Move semantics allowed
    KVCacheManager(KVCacheManager&&) noexcept = default;
    KVCacheManager& operator=(KVCacheManager&&) noexcept = default;

    void update(size_t layer_idx, size_t batch_idx, const float* __restrict k_src, const float* __restrict v_src, size_t pos) {
        validate_indices(layer_idx, batch_idx, pos);
        size_t offset = get_buffer_offset(layer_idx, batch_idx, pos);

        float* __restrict k_dst = k_cache_.data() + offset;
        float* __restrict v_dst = v_cache_.data() + offset;

        // Vectorized block copy
        std::memcpy(k_dst, k_src, seq_stride_ * sizeof(float));
        std::memcpy(v_dst, v_src, seq_stride_ * sizeof(float));

        // Thread-safe progress tracking
        {
            std::lock_guard<std::mutex> lock(seq_mutex_);
            if (pos >= current_seq_lens_[batch_idx]) {
                current_seq_lens_[batch_idx] = pos + 1;
            }
        }
    }

    [[nodiscard]] const float* get_k_ptr(size_t layer_idx, size_t batch_idx, size_t pos = 0) const {
        validate_indices(layer_idx, batch_idx, pos);
        return k_cache_.data() + get_buffer_offset(layer_idx, batch_idx, pos);
    }

    [[nodiscard]] const float* get_v_ptr(size_t layer_idx, size_t batch_idx, size_t pos = 0) const {
        validate_indices(layer_idx, batch_idx, pos);
        return v_cache_.data() + get_buffer_offset(layer_idx, batch_idx, pos);
    }

    void reset() noexcept {
        std::lock_guard<std::mutex> lock(seq_mutex_);
        std::fill(current_seq_lens_.begin(), current_seq_lens_.end(), 0);
    }

    void reset_batch(size_t batch_idx) {
        if (batch_idx >= config_.max_batch_size) {
            throw std::out_of_range("Batch index out of bounds");
        }
        std::lock_guard<std::mutex> lock(seq_mutex_);
        current_seq_lens_[batch_idx] = 0;
    }

    [[nodiscard]] size_t get_current_seq_len(size_t batch_idx) const {
        if (batch_idx >= config_.max_batch_size) {
            throw std::out_of_range("Batch index out of bounds");
        }
        return current_seq_lens_[batch_idx];
    }

    [[nodiscard]] const KVCacheConfig& config() const noexcept { return config_; }
};

} // namespace llama_pure_compute

#endif // KV_CACHE_H_