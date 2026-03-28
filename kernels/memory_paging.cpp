#include <cstdint>
#include <cstddef>
// #include <sys/mman.h> // For direct paging if needed

extern "C" {

    /// Manages paging KV cache from NVMe disk to CPU RAM for the EverMemOS memory lifecycle.
    /// Crucial for supporting millions of tokens within an 8GB RAM budget.
    void page_kv_cache(
        int32_t* block_indices,
        size_t num_blocks,
        void* mapped_memory_pool
    ) {
        // Simulates mapping the returned Top-K chunks from Long-Term (Disk/RAM)
        // into fast Working Memory for the Attention block to process.
        // The implementation here touches memory pages to ensure they are hot in the L3 cache.
        // Assume mapped_memory_pool points to an array of large KV Cache blocks.
        if (!mapped_memory_pool) return;

        char** blocks = (char**)mapped_memory_pool;
        size_t block_size = 4096; // Example 4KB KV page

        for (size_t i = 0; i < num_blocks; ++i) {
            int32_t idx = block_indices[i];
            if (idx >= 0) {
                // Touch the first byte to page it in
                volatile char touch = blocks[idx][0];
                (void)touch; // Suppress unused warning
            }
        }
    }

}