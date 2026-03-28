#include <cuda_runtime.h>
#include <cstdint>

/// 1.58-bit Ternary GPU Kernels (`__dp4a`)
/// This experimental compute shader replaces standard FP16 Matrix Multiplications
/// (GEMMs) on the GPU with highly specialized integer operations.
/// It unpacks 2-bit compressed weights and computes dot products in 4-byte
/// chunks using the NVIDIA `__dp4a` instruction (Dot Product 4 Add).
/// This enables a 70B parameter model to run efficiently on local RTX 4090s.
///
/// `packed_weights` (M x K / 4): 2-bit weights in uint8_t array.
/// `activations` (K x N): Quantized Int8 activations.
/// `output` (M x N): Accumulated Int32 outputs.
extern "C" __global__ void ternary_gemm_dp4a_kernel(
    const uint8_t* __restrict__ packed_weights,
    const int8_t* __restrict__ activations,
    int32_t* __restrict__ output,
    int m,
    int n,
    int k
) {
    // Thread block identifies the row (m) and col (n)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int32_t acc = 0;

        // Process K dimension in chunks of 4 (since dp4a takes 4 x 8-bit ints)
        for (int i = 0; i < k; i += 4) {
            // 1. Fetch 4 packed 2-bit weights (1 byte)
            int weight_idx = row * k + i;
            uint8_t packed_byte = packed_weights[weight_idx / 4];

            // Unpack 4 weights into a 32-bit register (each byte holds an 8-bit weight)
            int8_t w0 = (packed_byte & 0x03);
            int8_t w1 = ((packed_byte >> 2) & 0x03);
            int8_t w2 = ((packed_byte >> 4) & 0x03);
            int8_t w3 = ((packed_byte >> 6) & 0x03);

            // Map (0->0, 1->1, 2->-1)
            w0 = (w0 == 2) ? -1 : w0;
            w1 = (w1 == 2) ? -1 : w1;
            w2 = (w2 == 2) ? -1 : w2;
            w3 = (w3 == 2) ? -1 : w3;

            // Pack the four unpacked 8-bit weights into a single 32-bit integer for __dp4a
            uint32_t w_reg = ((uint32_t)(uint8_t)w3 << 24) |
                             ((uint32_t)(uint8_t)w2 << 16) |
                             ((uint32_t)(uint8_t)w1 << 8)  |
                             ((uint32_t)(uint8_t)w0);

            // 2. Fetch 4 corresponding 8-bit activations from Global Memory
            // For a naive layout (Activations are K x N, so they are not contiguous along K)
            // In a highly optimized tensor core layout, we would transpose Activations to N x K
            // so we could load a 32-bit integer directly `*(uint32_t*)&activations[...]`
            int8_t a0 = activations[(i + 0) * n + col];
            int8_t a1 = activations[(i + 1) * n + col];
            int8_t a2 = activations[(i + 2) * n + col];
            int8_t a3 = activations[(i + 3) * n + col];

            uint32_t a_reg = ((uint32_t)(uint8_t)a3 << 24) |
                             ((uint32_t)(uint8_t)a2 << 16) |
                             ((uint32_t)(uint8_t)a1 << 8)  |
                             ((uint32_t)(uint8_t)a0);

            // 3. Execute `__dp4a` (NVIDIA Int8 Dot Product)
            // acc = __dp4a(a_reg, w_reg, acc)
            // Computes: acc += (a0*w0 + a1*w1 + a2*w2 + a3*w3)
#if __CUDA_ARCH__ >= 610
            acc = __dp4a(a_reg, w_reg, acc);
#else
            // Fallback for extremely old GPUs
            acc += (a0*w0 + a1*w1 + a2*w2 + a3*w3);
#endif
        }

        // Store Int32 result
        output[row * n + col] = acc;
    }
}
