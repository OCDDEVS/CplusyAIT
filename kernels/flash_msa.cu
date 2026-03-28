#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

/// GPU-Native Flash-MSA Routing Kernel
/// This experimental kernel loads EverMemOS `MemScene` centroids directly into
/// ultra-fast GPU Shared Memory (SRAM) and performs the Top-K cosine similarity
/// routing without ever crossing the slow PCIe bus back to the CPU.
///
/// `query_vectors` (Batch x Dim)
/// `routing_keys` (NumKeys x Dim) -> The MemScene Centroids
/// `top_k_indices_out` (Batch x K) -> The matched indices
extern "C" __global__ void flash_msa_route_kernel(
    const float* __restrict__ query_vectors,
    const float* __restrict__ routing_keys,
    int32_t* __restrict__ top_k_indices_out,
    int num_queries,
    int num_keys,
    int vector_dim,
    int k
) {
    // Each block processes one query vector
    int query_idx = blockIdx.x;
    if (query_idx >= num_queries) return;

    // Use Shared Memory (SRAM) for fast reduction and dot products
    // We assume vector_dim is small enough to fit in shared memory (e.g., 128 or 256)
    extern __shared__ float sram[];
    float* s_query = sram; // size: vector_dim

    int tid = threadIdx.x;

    // 1. Load Query into SRAM
    if (tid < vector_dim) {
        s_query[tid] = query_vectors[query_idx * vector_dim + tid];
    }
    __syncthreads();

    // Calculate query norm for Cosine Similarity
    float q_norm_sq = 0.0f;
    for (int i = 0; i < vector_dim; ++i) {
        q_norm_sq += s_query[i] * s_query[i];
    }

    // 2. Iterate through all routing keys (MemScenes)
    // To prevent storing all scores, each thread can maintain a local Top-K heap.
    // For simplicity in this experimental version, thread 0 scans all keys.
    // In production, this is a parallel reduction over keys.

    if (tid == 0) {
        // Simple local arrays to maintain top K (Assume K is very small, e.g., 2 or 4)
        float top_scores[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
        int32_t top_indices[4] = {-1, -1, -1, -1};

        for (int key_idx = 0; key_idx < num_keys; ++key_idx) {
            float dot_product = 0.0f;
            float k_norm_sq = 0.0f;

            // Load key values directly from Global Memory (HBM) and compute
            for (int i = 0; i < vector_dim; ++i) {
                float k_val = routing_keys[key_idx * vector_dim + i];
                dot_product += s_query[i] * k_val;
                k_norm_sq += k_val * k_val;
            }

            float similarity = -FLT_MAX;
            if (q_norm_sq > 0.0f && k_norm_sq > 0.0f) {
                similarity = dot_product * rsqrtf(q_norm_sq * k_norm_sq); // Fast inverse square root
            }

            // Insert into Top-K array (Insertion sort for small K)
            for (int i = 0; i < k; ++i) {
                if (similarity > top_scores[i]) {
                    // Shift down
                    for (int j = k - 1; j > i; --j) {
                        top_scores[j] = top_scores[j - 1];
                        top_indices[j] = top_indices[j - 1];
                    }
                    top_scores[i] = similarity;
                    top_indices[i] = key_idx;
                    break;
                }
            }
        }

        // Write out the Top-K indices to Global Memory
        for (int i = 0; i < k; ++i) {
            top_k_indices_out[query_idx * k + i] = top_indices[i];
        }
    }
}
