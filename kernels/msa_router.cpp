#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cmath>

extern "C" {

    /// Memory Sparse Attention (MSA) Router Execution
    /// Executes fast Top-K vector dot products to select relevant KV blocks from RAM
    ///
    /// This is an O(N) linear time memory routing mechanism, bypassing full attention O(N^2).
    void msa_route_top_k(
        const float* query_vector,
        const float* routing_keys, // Highly compressed feature vectors in VRAM/Cache
        int32_t* top_k_indices,
        size_t num_keys,
        size_t vector_dim,
        size_t k
    ) {
        // Actual Implementation: Calculate cosine similarity to map query to MemScenes
        // Returns the top `k` indices representing relevant Memory Blocks.

        struct ScoreIndex {
            float score;
            int32_t index;

            bool operator<(const ScoreIndex& other) const {
                return score > other.score; // Max heap logic (descending)
            }
        };

        std::vector<ScoreIndex> scores;
        scores.reserve(num_keys);

        // Precalculate query norm
        float query_norm_sq = 0.0f;
        for (size_t j = 0; j < vector_dim; ++j) {
            query_norm_sq += query_vector[j] * query_vector[j];
        }

        for (size_t i = 0; i < num_keys; ++i) {
            float dot_product = 0.0f;
            float key_norm_sq = 0.0f;
            for (size_t j = 0; j < vector_dim; ++j) {
                float k_val = routing_keys[i * vector_dim + j];
                dot_product += query_vector[j] * k_val;
                key_norm_sq += k_val * k_val;
            }

            float similarity = 0.0f;
            if (query_norm_sq > 0.0f && key_norm_sq > 0.0f) {
                // Approximate sqrt for speed, or std::sqrt if precision is needed.
                // using standard sqrt here for correctness in cosine similarity.
                similarity = dot_product / std::sqrt(query_norm_sq * key_norm_sq);
            }

            scores.push_back({similarity, static_cast<int32_t>(i)});
        }

        // Sort descending
        std::sort(scores.begin(), scores.end());

        // Assign top k (fill rest with -1 if not enough keys)
        for (size_t i = 0; i < k; ++i) {
            if (i < num_keys) {
                top_k_indices[i] = scores[i].index;
            } else {
                top_k_indices[i] = -1; // Invalid sentinel
            }
        }
    }
}
