#include <cstdint>
#include <cstddef>
// For AVX-512 intrinsics, normally one would include <immintrin.h>
// #include <immintrin.h>

extern "C" {

    /// Performs a ternary matrix-vector multiplication (General Matrix Multiply).
    /// Replaces floating-point MACs with pure integer additions/subtractions
    /// for 1.58-bit networks like BitNet b1.58.
    ///
    /// `weights` are directly represented as int8_t {-1, 0, 1} for this benchmark phase.
    /// (Packing 2-bit values is complex to write purely natively without SIMD intrinsics yet)
    /// `activations` are 8-bit quantized integers (int8_t).
    /// `output` is accumulated into a 32-bit integer (int32_t) to prevent overflow.
    ///
    /// Computes: output = weights * activations
    /// Dimensions:
    ///   weights: (m x k)
    ///   activations: (k x n)
    ///   output: (m x n)
    void ternary_gemm(
        const int8_t* weights,
        const int8_t* activations,
        int32_t* output,
        size_t m,
        size_t n,
        size_t k
    ) {
        // Cache-friendly loop order (i, j, k) or block tiling would be best,
        // but for a pure algorithm latency test on CPU:

        for (size_t row = 0; row < m; ++row) {
            for (size_t col = 0; col < n; ++col) {
                int32_t acc = 0;
                // The innermost loop - pure addition/subtraction
                for (size_t i = 0; i < k; ++i) {
                    int8_t weight = weights[row * k + i];
                    int8_t act = activations[i * n + col];

                    // Pure addition/subtraction (no multiplication!)
                    // A branch prediction here is expensive, so a branchless approach:
                    // acc += weight * act; // Wait, we can't multiply!

                    // Branchless addition:
                    // This is mathematically: acc += act * weight (where weight is -1, 0, 1)
                    // If weight == 1, add act. If -1, sub act. If 0, add 0.
                    // The simplest branchless way is a bitwise mask, but standard C++:

                    if (weight == 1) {
                        acc += act;
                    } else if (weight == -1) {
                        acc -= act;
                    }
                    // if 0, do nothing (saves cycles)
                }
                output[row * n + col] = acc;
            }
        }
    }

}