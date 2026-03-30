use libc::{int32_t, int8_t, size_t, c_float, c_void};

extern "C" {
    /// Full Precision (FP32) General Matrix Multiply (GEMM)
    pub fn fp32_gemm(
        weights: *const c_float,
        activations: *const c_float,
        output: *mut c_float,
        m: size_t,
        n: size_t,
        k: size_t,
    );

    /// AVX2 highly-optimized Ternary GEMM Kernel.
    /// Operates on 2-bit packed weights (uint8_t arrays).
    pub fn ternary_gemm_avx2_packed(
        packed_weights: *const u8,
        activations: *const int8_t,
        output: *mut int32_t,
        m: size_t,
        n: size_t,
        k: size_t,
    );

    /// C FFI Binding for the 1.58-bit Ternary Matrix Multiply Kernel
    /// This operates on {-1,0,1} weights and quantized activations.
    pub fn ternary_gemm(
        weights: *const int8_t,
        activations: *const int8_t,
        output: *mut int32_t,
        m: size_t,
        n: size_t,
        k: size_t,
    );

    /// C FFI Binding for the MSA (Memory Sparse Attention) Router.
    /// Finds the top K memory blocks for context retrieval.
    pub fn msa_route_top_k(
        query_vector: *const c_float,
        routing_keys: *const c_float,
        top_k_indices: *mut int32_t,
        num_keys: size_t,
        vector_dim: size_t,
        k: size_t,
    );

    /// C FFI Binding for KV Cache Disk Paging.
    pub fn page_kv_cache(
        block_indices: *mut int32_t,
        num_blocks: size_t,
        mapped_memory_pool: *mut c_void,
    );

    /// Gathers Top-K MemScene centroids into a continuous working memory buffer
    pub fn gather_working_memory(
        top_k_indices: *const int32_t,
        k: size_t,
        long_term_memory_pool: *const c_float,
        vector_dim: size_t,
        working_memory_out: *mut c_float,
    );

    /// (CUDA) GPU-Native Flash-MSA Routing Kernel (in Shared Memory)
    #[cfg(feature = "cuda")]
    pub fn flash_msa_route_kernel(
        query_vectors: *const c_float,
        routing_keys: *const c_float,
        top_k_indices_out: *mut int32_t,
        num_queries: int32_t,
        num_keys: int32_t,
        vector_dim: int32_t,
        k: int32_t,
    );

    /// (CUDA) 1.58-bit Ternary GPU Kernels (__dp4a)
    #[cfg(feature = "cuda")]
    pub fn ternary_gemm_dp4a_kernel(
        packed_weights: *const u8,
        activations: *const int8_t,
        output: *mut int32_t,
        m: int32_t,
        n: int32_t,
        k: int32_t,
    );

    /// ARM NEON Ternary GEMM Kernel for aarch64.
    /// Same interface as AVX2: 2-bit packed weights, int8 activations.
    #[cfg(target_arch = "aarch64")]
    pub fn ternary_gemm_neon_packed(
        packed_weights: *const u8,
        activations: *const i8,
        output: *mut i32,
        m: usize,
        n: usize,
        k: usize,
    );
}
