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
}
