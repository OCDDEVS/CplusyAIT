//! Lightweight Mamba Selective State-Space Model for the inference pipeline.
//!
//! This operates on raw f32 slices (matching the inference pipeline's convention)
//! rather than candle Tensors. It maintains a recurrent hidden state per layer,
//! enabling O(N) linear-time sequence processing as described in the research docs.
//!
//! The "selective" aspect: B, C, and dt (Delta) are dynamically computed from the
//! input, allowing the model to selectively remember or forget information.

use crate::inference::ternary_linear::TernaryLinear;

/// Per-layer recurrent state for the Mamba SSM.
/// Shape: (hidden_dim, state_dim) stored as a flat Vec<f32>.
pub struct MambaState {
    pub state: Vec<f32>,
    pub hidden_dim: usize,
    pub state_dim: usize,
}

impl MambaState {
    pub fn new(hidden_dim: usize, state_dim: usize) -> Self {
        Self {
            state: vec![0.0; hidden_dim * state_dim],
            hidden_dim,
            state_dim,
        }
    }

    pub fn clear(&mut self) {
        self.state.fill(0.0);
    }
}

/// A Mamba SSM block for inference, using TernaryLinear projections.
/// Replaces quadratic self-attention with O(1) per-token recurrent updates.
///
/// Architecture (per token):
///   x -> x_proj -> [u, gate]
///   u -> dt_proj -> dt (step size, softplus activated)
///   u -> b_proj  -> B  (input-to-state)
///   u -> c_proj  -> C  (state-to-output)
///   h_t = exp(-softplus(A) * dt) * h_{t-1} + dt * (u outer B)
///   y_t = sum_over_state(C * h_t)
///   output = o_proj(y_t * silu(gate))
pub struct MambaLayer {
    pub x_proj: TernaryLinear,   // hidden_dim -> hidden_dim * 2 (u + gate)
    pub dt_proj: TernaryLinear,  // hidden_dim -> hidden_dim (step size)
    pub b_proj: TernaryLinear,   // hidden_dim -> state_dim
    pub c_proj: TernaryLinear,   // hidden_dim -> state_dim
    pub out_proj: TernaryLinear, // hidden_dim -> hidden_dim

    /// Diagonal A parameter (log-space), length = hidden_dim.
    /// Initialized to negative values so exp(-softplus(A)*dt) < 1 (stable decay).
    pub a_log: Vec<f32>,

    /// D skip-connection parameter, length = hidden_dim.
    pub d_param: Vec<f32>,

    pub hidden_dim: usize,
    pub state_dim: usize,
}

impl MambaLayer {
    /// Single-token forward pass with recurrent state update.
    /// `x` is the input hidden state (length = hidden_dim).
    /// `mamba_state` is the persistent recurrent state for this layer.
    /// Returns the output hidden state (length = hidden_dim).
    pub fn forward(&self, x: &[f32], mamba_state: &mut MambaState) -> Vec<f32> {
        let h = self.hidden_dim;
        let s = self.state_dim;

        // 1. Expand input: x_proj(x) -> [u, gate] each of size hidden_dim
        let x_expanded = self.x_proj.forward(x, 1);
        let u = &x_expanded[..h];
        let gate = &x_expanded[h..h * 2];

        // 2. Compute dynamic SSM parameters from u (the "selective" part)
        let dt_raw = self.dt_proj.forward(u, 1);
        let b = self.b_proj.forward(u, 1);
        let c = self.c_proj.forward(u, 1);

        // dt = softplus(dt_raw) to ensure positive step size
        let dt: Vec<f32> = dt_raw.iter().map(|&v| softplus(v)).collect();

        // 3. Recurrent state update: h_t = decay * h_{t-1} + dt * (u outer B)
        // decay_i = exp(-softplus(a_log_i) * dt_i)
        // For each (i, j) in (hidden_dim, state_dim):
        //   state[i,j] = decay_i * state[i,j] + dt_i * u_i * B_j
        for i in 0..h {
            let decay = (-softplus(self.a_log[i]) * dt[i]).exp();
            let dt_u = dt[i] * u[i];
            for j in 0..s {
                let idx = i * s + j;
                mamba_state.state[idx] = decay * mamba_state.state[idx] + dt_u * b[j];
            }
        }

        // 4. Output: y_i = sum_j(C_j * state[i,j]) + D_i * x_i
        let mut y = vec![0.0f32; h];
        for i in 0..h {
            let mut sum = 0.0f32;
            for j in 0..s {
                sum += c[j] * mamba_state.state[i * s + j];
            }
            y[i] = sum + self.d_param[i] * x[i];
        }

        // 5. Gating: output = out_proj(y * silu(gate))
        let mut gated = vec![0.0f32; h];
        for i in 0..h {
            gated[i] = y[i] * silu(gate[i]);
        }

        self.out_proj.forward(&gated, 1)
    }
}

/// Collection of Mamba states across all layers.
pub struct MambaCache {
    pub layers: Vec<MambaState>,
}

impl MambaCache {
    pub fn new(num_layers: usize, hidden_dim: usize, state_dim: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| MambaState::new(hidden_dim, state_dim))
                .collect(),
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // avoid overflow
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
