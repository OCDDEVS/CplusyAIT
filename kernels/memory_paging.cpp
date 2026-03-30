#include <cstdint>
#include <cstddef>
#include <cstring>

extern "C" {

    /// Page KV cache blocks from a memory-mapped region into a working buffer.
    /// `block_indices`: array of block slot indices to fetch (length = num_blocks).
    /// `num_blocks`: number of blocks to page in.
    /// `mapped_memory_pool`: base pointer to the mmap'd file region.
    /// `block_size_bytes`: size of each KV block in bytes (keys + values).
    /// `output_buffer`: destination buffer (must be at least num_blocks * block_size_bytes).
    void page_kv_cache(
        int32_t* block_indices,
        size_t num_blocks,
        void* mapped_memory_pool,
        size_t block_size_bytes,
        void* output_buffer
    ) {
        const char* src_base = static_cast<const char*>(mapped_memory_pool);
        char* dst_base = static_cast<char*>(output_buffer);

        for (size_t i = 0; i < num_blocks; ++i) {
            int32_t idx = block_indices[i];
            if (idx >= 0) {
                const char* src = src_base + (static_cast<size_t>(idx) * block_size_bytes);
                char* dst = dst_base + (i * block_size_bytes);
                std::memcpy(dst, src, block_size_bytes);
            } else {
                // Invalid index: zero-fill the output slot
                char* dst = dst_base + (i * block_size_bytes);
                std::memset(dst, 0, block_size_bytes);
            }
        }
    }

    /// Gathers Top-K MemScene centroids into a contiguous working memory buffer.
    /// Used by the MSA router to collect selected memory vectors for attention.
    void gather_working_memory(
        const int32_t* top_k_indices,
        size_t k,
        const float* long_term_memory_pool,
        size_t vector_dim,
        float* working_memory_out
    ) {
        for (size_t i = 0; i < k; ++i) {
            int32_t idx = top_k_indices[i];

            if (idx >= 0) {
                const float* src = &long_term_memory_pool[idx * vector_dim];
                float* dst = &working_memory_out[i * vector_dim];

                // Fast contiguous memory copy (SIMD optimized by compiler with -O3)
                for (size_t j = 0; j < vector_dim; ++j) {
                    dst[j] = src[j];
                }
            } else {
                float* dst = &working_memory_out[i * vector_dim];
                for (size_t j = 0; j < vector_dim; ++j) {
                    dst[j] = 0.0f;
                }
            }
        }
    }

}
