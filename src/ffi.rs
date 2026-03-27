use libc::{int32_t, int8_t, size_t, c_float, c_void};

extern "C" {
    /// C FFI Binding for the 1.58-bit Ternary Matrix Multiply Kernel
    /// This operates on packed weights and quantized activations.
    pub fn ternary_gemm(
        packed_weights: *const int8_t,
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
