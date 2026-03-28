use candle_core::{Tensor, Result};
use candle_nn::{Linear, linear, VarBuilder, Module};

/// A standard Multi-Head Attention layer integrating standard RoPE (Rotary Positional Embeddings)
/// with Document-wise resets for 100M token contexts as theorized by MSA.
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
}

impl MultiHeadAttention {
    pub fn new(vb: VarBuilder, hidden_dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = hidden_dim / num_heads;
        let q_proj = linear(hidden_dim, hidden_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_dim, hidden_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_dim, hidden_dim, vb.pp("v_proj"))?;
        let o_proj = linear(hidden_dim, hidden_dim, vb.pp("o_proj"))?;

        Ok(Self {
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    /// Forward pass including standard RoPE positional encodings.
    /// `seq_positions` allows us to pass custom positions. By resetting this counter
    /// at document boundaries, we achieve "Document-wise RoPE", enabling 100M context extrapolation.
    pub fn forward(&self, x: &Tensor, seq_positions: &[usize]) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_dim) = x.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head: (b_sz, seq_len, num_heads, head_dim)
        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let mut k = k.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        // Apply Document-wise RoPE here
        let q_rope = self.apply_rope(&q, seq_positions)?;
        k = self.apply_rope(&k, seq_positions)?;

        // Scaled Dot-Product Attention: Softmax(Q * K^T / sqrt(d)) * V
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let att_scores = q_rope.matmul(&k.transpose(2, 3)?)?;
        let att_scores = (att_scores * scale)?;

        let att_probs = candle_nn::ops::softmax(&att_scores, candle_core::D::Minus1)?;
        let att_output = att_probs.matmul(&v)?;

        // Reshape back to (b_sz, seq_len, hidden_dim)
        let att_output = att_output.transpose(1, 2)?.reshape((b_sz, seq_len, hidden_dim))?;

        // Output projection
        self.o_proj.forward(&att_output)
    }

    /// Real Rotary Positional Embedding (RoPE) implementation.
    /// Supports "Document-wise RoPE" by accepting an arbitrary `positions` array,
    /// which can be reset at document boundaries.
    fn apply_rope(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let (b_sz, seq_len, num_heads, head_dim) = x.dims4()?;

        let mut rotated_elements = vec![0.0f32; b_sz * seq_len * num_heads * head_dim];
        let x_vec = x.flatten_all()?.to_vec1::<f32>()?;

        // Calculate frequencies for RoPE
        for b in 0..b_sz {
            for s in 0..seq_len {
                let pos = positions[s] as f32;
                for h in 0..num_heads {
                    for d in (0..head_dim).step_by(2) {
                        let inv_freq = 1.0 / 10000_f32.powf(d as f32 / head_dim as f32);
                        let freq = pos * inv_freq;
                        let (sin, cos) = freq.sin_cos();

                        let idx = b * (seq_len * num_heads * head_dim) +
                                  s * (num_heads * head_dim) +
                                  h * head_dim + d;

                        let x0 = x_vec[idx];
                        let x1 = x_vec[idx + 1];

                        rotated_elements[idx] = x0 * cos - x1 * sin;
                        rotated_elements[idx + 1] = x1 * cos + x0 * sin;
                    }
                }
            }
        }

        Tensor::from_vec(rotated_elements, (b_sz, seq_len, num_heads, head_dim), x.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    use candle_nn::VarMap;

    #[test]
    fn test_attention_rope_forward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let hidden_dim = 16;
        let num_heads = 4;
        let mha = MultiHeadAttention::new(vb, hidden_dim, num_heads)?;

        // Create dummy input (batch=1, seq_len=4, hidden=16)
        let x_data: Vec<f32> = (0..(1 * 4 * 16)).map(|i| i as f32 * 0.01).collect();
        let x = Tensor::from_vec(x_data, (1, 4, 16), &device)?;

        // Positional IDs for document-wise rope
        // E.g., Document 1: [0, 1], Document 2: [0, 1]
        let positions = vec![0, 1, 0, 1];

        let out = mha.forward(&x, &positions)?;
        assert_eq!(out.dims3()?, (1, 4, 16));
        Ok(())
    }
}
