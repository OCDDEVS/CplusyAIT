use rand::Rng;
use crate::ffi;

/// A simple single-layer Multi-Layer Perceptron (MLP) trained with 1.58-bit ternary weights
/// using the Straight-Through Estimator (STE) method.
pub struct TernarySTEModel {
    pub m: usize,
    pub n: usize,
    pub k: usize,

    /// The FP32 "Master" Weights. These hold the high-precision gradients and get updated during backprop.
    pub master_weights: Vec<f32>,

    /// The quantized 1.58-bit {-1, 0, 1} weights used strictly during the forward pass.
    pub quantized_weights: Vec<i8>,
}

impl TernarySTEModel {
    pub fn new(m: usize, k: usize, n: usize) -> Self {
        let mut rng = rand::thread_rng();
        // Initialize Master Weights using Xavier-like small uniform
        let master_weights: Vec<f32> = (0..k * n)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        let quantized_weights = vec![0; k * n];

        Self { m, n, k, master_weights, quantized_weights }
    }

    /// Quantize the Master FP32 Weights down to 1.58-bit (-1, 0, 1) using BitNet absolute mean scaling.
    pub fn quantize(&mut self) {
        // Find the absolute mean of the master weights
        let mut sum_abs = 0.0;
        for w in &self.master_weights {
            sum_abs += w.abs();
        }
        let gamma = sum_abs / (self.master_weights.len() as f32 + 1e-8);

        // Quantize: W_quant = Round(Clamp(W / gamma, -1.0, 1.0))
        for i in 0..self.master_weights.len() {
            let scaled = self.master_weights[i] / gamma;
            let clamped = scaled.clamp(-1.0, 1.0);
            self.quantized_weights[i] = clamped.round() as i8;
        }
    }

    /// Forward pass using the standard FP32 Kernel for the toy training loop
    /// Note: Since our fast Ternary C++ kernel is strictly written as (Weights * Activations)
    /// where Weights MUST be {-1, 0, 1}, we cannot swap the arguments (A * B) if A is activations
    /// containing arbitrary integers.
    /// For this toy STE training loop, we will use a naive Rust implementation of the forward pass
    /// to correctly apply the quantized ternary weights to the activations.
    pub fn forward_naive(&mut self, activations: &[f32], output: &mut [f32]) {
        self.quantize(); // Ensure quantized weights match current master weights before forward pass

        // Output (M x N) = Activations (M x K) * Weights (K x N)
        for row in 0..self.m {
            for col in 0..self.n {
                let mut acc = 0.0;
                for i in 0..self.k {
                    let act = activations[row * self.k + i];
                    let weight = self.quantized_weights[i * self.n + col];

                    // Ternary math (using the quantized -1, 0, 1 weights)
                    if weight == 1 {
                        acc += act;
                    } else if weight == -1 {
                        acc -= act;
                    }
                }
                output[row * self.n + col] = acc;
            }
        }
    }

    /// Fake backward pass / toy update for the STE demonstration.
    /// In a real scenario, this calculates the gradients wrt to the FP32 output
    /// and applies them DIRECTLY to the FP32 master weights.
    pub fn backward_ste_update(&mut self, loss_gradient: &[f32], learning_rate: f32) {
        // Simplified gradient descent step directly applying fake gradients to master weights
        for i in 0..self.master_weights.len() {
            // Apply straight-through update
            self.master_weights[i] -= learning_rate * loss_gradient[i % loss_gradient.len()];
        }
    }
}
