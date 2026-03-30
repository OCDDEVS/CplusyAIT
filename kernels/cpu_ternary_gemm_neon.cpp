#include <cstdint>
#include <cstddef>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

extern "C" {

/// Ternary GEMM using ARM NEON SIMD for Apple Silicon / ARM64.
/// Same interface as the AVX2 version: 2-bit packed weights, int8 activations.
///
/// Weights Encoding (2 bits per weight):
/// 00 = 0, 01 = +1, 10 = -1, 11 = 0
///
/// Output(M, N) = PackedWeights(M, K) * Activations(K, N)
void ternary_gemm_neon_packed(
    const uint8_t* packed_weights,
    const int8_t* activations,
    int32_t* output,
    size_t m,
    size_t n,
    size_t k
) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Process N columns in chunks of 4 (NEON has 128-bit registers = 4x32-bit)
    size_t num_chunks = n / 4;
    size_t remainder = n % 4;

    for (size_t row = 0; row < m; ++row) {
        // Initialize NEON accumulators
        int32x4_t* acc = nullptr;
        if (num_chunks > 0) {
            acc = new int32x4_t[num_chunks];
            for (size_t c = 0; c < num_chunks; ++c) {
                acc[c] = vdupq_n_s32(0);
            }
        }

        int32_t* acc_rem = nullptr;
        if (remainder > 0) {
            acc_rem = new int32_t[remainder];
            for (size_t r = 0; r < remainder; ++r) acc_rem[r] = 0;
        }

        for (size_t i = 0; i < k; ++i) {
            size_t weight_idx = row * k + i;
            size_t byte_idx = weight_idx / 4;
            size_t bit_offset = (weight_idx % 4) * 2;

            uint8_t byte = packed_weights[byte_idx];
            uint8_t weight_val = (byte >> bit_offset) & 0x03;

            if (weight_val == 0) continue;

            const int8_t* act_row = &activations[i * n];

            // NEON path: process 4 activations at a time
            for (size_t c = 0; c < num_chunks; ++c) {
                // Load 4 int8 activations
                int8_t vals[4];
                for (int j = 0; j < 4; ++j) {
                    vals[j] = act_row[c * 4 + j];
                }

                // Widen to int32
                int32x4_t act_32 = {vals[0], vals[1], vals[2], vals[3]};

                if (weight_val == 1) {
                    acc[c] = vaddq_s32(acc[c], act_32);
                } else if (weight_val == 2) {
                    acc[c] = vsubq_s32(acc[c], act_32);
                }
            }

            // Scalar remainder
            for (size_t r = 0; r < remainder; ++r) {
                int8_t act = act_row[num_chunks * 4 + r];
                if (weight_val == 1) acc_rem[r] += act;
                else if (weight_val == 2) acc_rem[r] -= act;
            }
        }

        // Store results
        if (num_chunks > 0) {
            for (size_t c = 0; c < num_chunks; ++c) {
                vst1q_s32(&output[row * n + c * 4], acc[c]);
            }
            delete[] acc;
        }

        if (remainder > 0) {
            for (size_t r = 0; r < remainder; ++r) {
                output[row * n + num_chunks * 4 + r] = acc_rem[r];
            }
            delete[] acc_rem;
        }
    }

#else
    // Fallback: scalar implementation for non-NEON platforms
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            int32_t acc = 0;
            for (size_t i = 0; i < k; ++i) {
                size_t weight_idx = row * k + i;
                size_t byte_idx = weight_idx / 4;
                size_t bit_offset = (weight_idx % 4) * 2;
                uint8_t w = (packed_weights[byte_idx] >> bit_offset) & 0x03;
                int8_t a = activations[i * n + col];
                if (w == 1) acc += a;
                else if (w == 2) acc -= a;
            }
            output[row * n + col] = acc;
        }
    }
#endif
}

} // extern "C"
