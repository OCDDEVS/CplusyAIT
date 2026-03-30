//! A ternary linear layer that uses the AVX2 packed GEMM kernel for inference.
//! Replaces standard FP32 matmul with 1.58-bit ternary computation.

use crate::ffi;

/// A pre-quantized linear layer: stores 2-bit packed weights and the gamma scale.
/// Computes: output = (packed_weights @ quantized_activations) * gamma * act_scale
pub struct TernaryLinear {
    /// 2-bit packed weights, shape conceptually (out_features, in_features),
    /// stored as (out_features * in_features / 4) bytes.
    pub packed_weights: Vec<u8>,
    pub in_features: usize,
    pub out_features: usize,
    /// BitNet absolute-mean scale factor.
    pub gamma: f32,
}

impl TernaryLinear {
    pub fn new(packed_weights: Vec<u8>, in_features: usize, out_features: usize, gamma: f32) -> Self {
        Self { packed_weights, in_features, out_features, gamma }
    }

    /// Forward pass: input is f32 slice of length `batch * in_features`.
    /// Output is f32 slice of length `batch * out_features`.
    /// For autoregressive decoding, batch is typically 1.
    pub fn forward(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let k = self.in_features;
        let n = batch; // columns of the activation matrix
        let m = self.out_features; // rows of the weight matrix

        // 1. Quantize activations to int8
        let mut max_abs = 0.0f32;
        for &v in input {
            let a = v.abs();
            if a > max_abs { max_abs = a; }
        }
        let act_scale = max_abs / 127.0 + 1e-8;

        let mut acts_i8 = vec![0i8; input.len()];
        for (i, &v) in input.iter().enumerate() {
            acts_i8[i] = (v / act_scale).round().clamp(-127.0, 127.0) as i8;
        }

        // 2. Call AVX2 kernel: Output(M, N) = Weights(M, K) * Acts(K, N)
        let mut output_i32 = vec![0i32; m * n];

        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                unsafe {
                    ffi::ternary_gemm_avx2_packed(
                        self.packed_weights.as_ptr(),
                        acts_i8.as_ptr(),
                        output_i32.as_mut_ptr(),
                        m, n, k,
                    );
                }
            } else {
                self.scalar_fallback(&acts_i8, &mut output_i32, m, n, k);
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        self.scalar_fallback(&acts_i8, &mut output_i32, m, n, k);

        // 3. Dequantize: out_f32 = out_i32 * gamma * act_scale
        let dequant = self.gamma * act_scale;
        let mut output_f32 = vec![0.0f32; m * n];
        for i in 0..output_f32.len() {
            output_f32[i] = output_i32[i] as f32 * dequant;
        }

        output_f32
    }

    fn scalar_fallback(&self, acts: &[i8], out: &mut [i32], m: usize, n: usize, k: usize) {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0i32;
                for i in 0..k {
                    let idx = row * k + i;
                    let byte_idx = idx / 4;
                    let bit_offset = (idx % 4) * 2;
                    let w = (self.packed_weights[byte_idx] >> bit_offset) & 0x03;
                    let a = acts[i * n + col] as i32;
                    if w == 1 { acc += a; }
                    else if w == 2 { acc -= a; }
                }
                out[row * n + col] = acc;
            }
        }
    }
}
