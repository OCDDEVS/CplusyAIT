//! A ternary linear layer that uses the AVX2 packed GEMM kernel for inference.
//! Replaces standard FP32 matmul with 1.58-bit ternary computation.
//! When compiled with `--features cuda`, dispatches to the GPU __dp4a kernel.

use crate::ffi;

/// A pre-quantized linear layer: stores 2-bit packed weights and the gamma scale.
/// Computes: output = (packed_weights @ quantized_activations) * gamma * act_scale
///
/// Supports two modes:
/// - Scalar gamma: a single scale factor for the entire layer (simple PTQ path).
/// - Per-channel gammas: one scale factor per output row (GPTQ path).
///   When per-channel gammas are present, each output row `i` is scaled by
///   `channel_gammas[i]` instead of the scalar `gamma`.
pub struct TernaryLinear {
    /// 2-bit packed weights, shape conceptually (out_features, in_features),
    /// stored as (out_features * in_features / 4) bytes.
    pub packed_weights: Vec<u8>,
    pub in_features: usize,
    pub out_features: usize,
    /// BitNet absolute-mean scale factor (scalar, used when channel_gammas is empty).
    pub gamma: f32,
    /// Per-channel (per-row) gamma values from GPTQ quantization.
    /// When non-empty, `channel_gammas[i]` is used instead of `gamma` for row `i`.
    pub channel_gammas: Vec<f32>,
    /// Variance correction factor: compensates for the variance loss when
    /// quantizing FP16 weights to ternary {-1,0,+1}. Ternary weights have
    /// lower variance than the original because they can't represent the tails
    /// of the weight distribution. Without this correction, each layer's output
    /// is ~1.5-3x too small, compounding to ~10000x over 22 layers.
    pub variance_correction: f32,
}

impl TernaryLinear {
    pub fn new(packed_weights: Vec<u8>, in_features: usize, out_features: usize, gamma: f32) -> Self {
        let vc = Self::compute_variance_correction(&packed_weights, in_features, out_features);
        Self { packed_weights, in_features, out_features, gamma, channel_gammas: Vec::new(), variance_correction: vc }
    }

    pub fn with_channel_gammas(packed_weights: Vec<u8>, in_features: usize, out_features: usize, gamma: f32, channel_gammas: Vec<f32>) -> Self {
        let vc = Self::compute_variance_correction(&packed_weights, in_features, out_features);
        Self { packed_weights, in_features, out_features, gamma, channel_gammas, variance_correction: vc }
    }

    /// Compute variance correction factor from packed weights.
    /// Ternary weights lose variance because they can't represent the tails
    /// of the weight distribution. We estimate the correction as:
    ///   correction = expected_rms / ternary_rms
    /// where expected_rms is estimated from gamma using the relationship
    /// between mean-absolute-value and RMS for typical weight distributions.
    /// For neural network weights (approximately Gaussian):
    ///   rms ≈ gamma * sqrt(pi/2) * sqrt(1 / fraction_nonzero)
    /// But ternary rms = gamma * sqrt(fraction_nonzero)
    /// So correction = sqrt(pi/2) / fraction_nonzero
    fn compute_variance_correction(packed: &[u8], in_features: usize, out_features: usize) -> f32 {
        let total_weights = out_features * in_features;
        let mut nonzero = 0usize;
        for i in 0..total_weights {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            if byte_idx < packed.len() {
                let w = (packed[byte_idx] >> bit_offset) & 0x03;
                if w != 0 { nonzero += 1; }
            }
        }
        let p = nonzero as f32 / total_weights as f32;
        if p < 0.01 { return 1.0; }
        // Empirical correction for ternary quantization variance loss.
        // For neural network weights, rms(W) / mean_abs(W) ≈ 1.8 (heavier tails
        // than Gaussian where the ratio would be sqrt(pi/2) ≈ 1.25).
        // Ternary rms = gamma * sqrt(p), original rms ≈ gamma * 1.8
        // correction = 1.8 / sqrt(p)
        1.8 / p.sqrt()
    }

    /// Forward pass: input is f32 slice of length `batch * in_features`.
    /// Output is f32 slice of length `batch * out_features`.
    /// For autoregressive decoding, batch is typically 1.
    ///
    /// Dispatch order:
    /// 1. `cuda` feature → GPU __dp4a kernel
    /// 2. x86_64 + AVX2 → CPU AVX2 SIMD kernel
    /// 3. aarch64 → ARM NEON kernel
    /// 4. Scalar fallback
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
        let act_scale = if max_abs < 1e-8 { 1e-8 } else { max_abs / 127.0 };

        let mut acts_i8 = vec![0i8; input.len()];
        for (i, &v) in input.iter().enumerate() {
            acts_i8[i] = (v / act_scale).round().clamp(-127.0, 127.0) as i8;
        }

        // 2. Call compute kernel: Output(M, N) = Weights(M, K) * Acts(K, N)
        let mut output_i32 = vec![0i32; m * n];

        #[cfg(feature = "cuda")]
        {
            unsafe {
                ffi::ternary_gemm_dp4a_kernel(
                    self.packed_weights.as_ptr(),
                    acts_i8.as_ptr(),
                    output_i32.as_mut_ptr(),
                    m as i32, n as i32, k as i32,
                );
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
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

            #[cfg(target_arch = "aarch64")]
            {
                unsafe {
                    ffi::ternary_gemm_neon_packed(
                        self.packed_weights.as_ptr(),
                        acts_i8.as_ptr(),
                        output_i32.as_mut_ptr(),
                        m, n, k,
                    );
                }
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            self.scalar_fallback(&acts_i8, &mut output_i32, m, n, k);
        }

        // 3. Dequantize: out_f32 = out_i32 * gamma * act_scale * variance_correction
        // The variance_correction compensates for the fact that ternary weights
        // have lower variance than the original FP16 weights. Without it, each
        // layer's output is ~1.5-3x too small, compounding catastrophically.
        let mut output_f32 = vec![0.0f32; m * n];
        if !self.channel_gammas.is_empty() {
            for row in 0..m {
                let row_gamma = self.channel_gammas[row];
                let dequant = row_gamma * act_scale * self.variance_correction;
                for col in 0..n {
                    output_f32[row * n + col] = output_i32[row * n + col] as f32 * dequant;
                }
            }
        } else {
            let dequant = self.gamma * act_scale * self.variance_correction;
            for i in 0..output_f32.len() {
                output_f32[i] = output_i32[i] as f32 * dequant;
            }
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
