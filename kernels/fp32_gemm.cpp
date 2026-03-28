#include <cstdint>
#include <cstddef>

extern "C" {

    /// Standard Full Precision (FP32) General Matrix Multiply (GEMM)
    /// This serves as our baseline to compare against the 1.58-bit Ternary kernel.
    ///
    /// Computes: output = weights * activations
    /// Dimensions:
    ///   weights: (m x k)
    ///   activations: (k x n)
    ///   output: (m x n)
    void fp32_gemm(
        const float* weights,
        const float* activations,
        float* output,
        size_t m,
        size_t n,
        size_t k
    ) {
        // Simple O(m*n*k) implementation
        // For a true baseline, we unroll and reorder loops to cache-friendly forms

        for (size_t row = 0; row < m; ++row) {
            for (size_t col = 0; col < n; ++col) {
                float acc = 0.0f;
                // The innermost loop - the MAC (Multiply-Accumulate)
                // Note the correct memory indexing for standard GEMM.
                // Weights: m x k. Activations: k x n. Output: m x n
                for (size_t i = 0; i < k; ++i) {
                    // weights[row][i] * activations[i][col]
                    acc += weights[row * k + i] * activations[i * n + col];
                }
                output[row * n + col] = acc;
            }
        }
    }

}
