use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{Embedding, Module, VarBuilder};
use std::sync::Arc;
use crate::ffi;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

impl Config {
    pub fn qwen_1_5b() -> Self {
        Self {
            hidden_size: 1536,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            intermediate_size: 8960,
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
        }
    }
}

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(&variance + self.eps)?.sqrt()?)?;
        x_normed.broadcast_mul(&self.weight)
    }
}

/// A 1.58-bit Linear Layer utilizing AVX2 SIMD optimizations.
/// Weights are stored as packed `u8` (4 weights per byte).
pub struct TernaryLinear {
    packed_weights: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl TernaryLinear {
    pub fn new(packed_weights: Tensor, bias: Option<Tensor>, in_features: usize, out_features: usize) -> Self {
        Self { packed_weights, bias, in_features, out_features }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, in_f) = x.dims3()?;
        if in_f != self.in_features {
            return Err(candle_core::Error::Msg(format!("Input feature size mismatch: expected {}, got {}", self.in_features, in_f)));
        }

        // Quantize inputs to int8. For BitNet 1.58b, activations are usually quantized to int8
        // via absolute maximum scalar.
        let x_f32 = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        // Simple activation quantization (scaling by max abs value to 127)
        let max_val = x_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-5);
        let scale = 127.0 / max_val;

        let mut x_i8 = vec![0i8; x_f32.len()];
        for i in 0..x_f32.len() {
            x_i8[i] = (x_f32[i] * scale).round().clamp(-127.0, 127.0) as i8;
        }

        let m = b_sz * seq_len;
        let k = self.in_features;
        let n = self.out_features;

        let mut out_i32 = vec![0i32; m * n];

        // Ensure packed weights are retrieved properly
        let weights_u8 = self.packed_weights.flatten_all()?.to_vec1::<u8>()?;

        // Dispatch to AVX2 kernel (assuming target arch matches)
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                ffi::ternary_gemm_avx2_packed(
                    weights_u8.as_ptr(),
                    x_i8.as_ptr(),
                    out_i32.as_mut_ptr(),
                    m, n, k
                );
            }
        } else {
            // Fallback to basic ternary C function
            return Err(candle_core::Error::Msg("AVX2 not detected on x86_64, fallback not strictly typed for u8 here".to_string()));
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            return Err(candle_core::Error::Msg("AVX2 Kernel strictly requires x86_64".to_string()));
        }

        // Dequantize: output / scale
        let mut out_f32 = vec![0f32; m * n];
        let inv_scale = 1.0 / scale;
        for i in 0..m * n {
            out_f32[i] = out_i32[i] as f32 * inv_scale;
        }

        let mut out = Tensor::from_vec(out_f32, (b_sz, seq_len, n), x.device())?;

        if let Some(b) = &self.bias {
            out = out.broadcast_add(b)?;
        }

        Ok(out)
    }
}

pub struct Mlp {
    gate_proj: TernaryLinear,
    up_proj: TernaryLinear,
    down_proj: TernaryLinear,
}

impl Mlp {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        // Load the 2-bit packed weights from safetensors (saved as u8)
        let gate_proj = TernaryLinear::new(
            vb.get((config.intermediate_size, config.hidden_size / 4), "gate_proj.weight")?,
            None, config.hidden_size, config.intermediate_size
        );
        let up_proj = TernaryLinear::new(
            vb.get((config.intermediate_size, config.hidden_size / 4), "up_proj.weight")?,
            None, config.hidden_size, config.intermediate_size
        );
        let down_proj = TernaryLinear::new(
            vb.get((config.hidden_size, config.intermediate_size / 4), "down_proj.weight")?,
            None, config.intermediate_size, config.hidden_size
        );

        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        // SiLU activation: x * sigmoid(x)
        let silu = candle_nn::ops::silu(&gate)?;

        let hidden = (silu * up)?;
        self.down_proj.forward(&hidden)
    }
}

pub struct Attention {
    q_proj: TernaryLinear,
    k_proj: TernaryLinear,
    v_proj: TernaryLinear,
    o_proj: TernaryLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let q_proj = TernaryLinear::new(
            vb.get((config.num_attention_heads * head_dim, config.hidden_size / 4), "q_proj.weight")?,
            None, config.hidden_size, config.num_attention_heads * head_dim
        );
        let k_proj = TernaryLinear::new(
            vb.get((config.num_key_value_heads * head_dim, config.hidden_size / 4), "k_proj.weight")?,
            None, config.hidden_size, config.num_key_value_heads * head_dim
        );
        let v_proj = TernaryLinear::new(
            vb.get((config.num_key_value_heads * head_dim, config.hidden_size / 4), "v_proj.weight")?,
            None, config.hidden_size, config.num_key_value_heads * head_dim
        );
        let o_proj = TernaryLinear::new(
            vb.get((config.hidden_size, config.num_attention_heads * head_dim / 4), "o_proj.weight")?,
            None, config.num_attention_heads * head_dim, config.hidden_size
        );

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, _positions: &[usize]) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // Simple repetition for GQA
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // Skipping exact RoPE here for simplicity of AVX demo inference

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let att_scores = q.matmul(&k.transpose(2, 3)?)?;
        let att_scores = (att_scores * scale)?;

        let att_probs = candle_nn::ops::softmax_last_dim(&att_scores)?;
        let att_output = att_probs.matmul(&v)?;

        let att_output = att_output.transpose(1, 2)?.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&att_output)
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b_sz, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b_sz, num_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((b_sz, num_kv_heads * n_rep, seq_len, head_dim))
}

pub struct Block {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Block {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let self_attn = Attention::new(vb.pp("self_attn"), config)?;
        let mlp = Mlp::new(vb.pp("mlp"), config)?;
        let input_layernorm = RmsNorm::new(
            vb.get(config.hidden_size, "input_layernorm.weight")?,
            config.rms_norm_eps
        );
        let post_attention_layernorm = RmsNorm::new(
            vb.get(config.hidden_size, "post_attention_layernorm.weight")?,
            config.rms_norm_eps
        );

        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm })
    }

    pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, positions)?;
        let x = (x + residual)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}

pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<Block>,
    norm: RmsNorm,
    lm_head: TernaryLinear,
}

impl Model {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Block::new(vb.pp(&format!("model.layers.{}", i)), config)?);
        }
        let norm = RmsNorm::new(
            vb.get(config.hidden_size, "model.norm.weight")?,
            config.rms_norm_eps
        );

        let lm_head = TernaryLinear::new(
            vb.get((config.vocab_size, config.hidden_size / 4), "lm_head.weight")?,
            None, config.hidden_size, config.vocab_size
        );

        Ok(Self { embed_tokens, layers, norm, lm_head })
    }

    pub fn forward(&self, tokens: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(tokens)?;
        for layer in &self.layers {
            x = layer.forward(&x, positions)?;
        }
        x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }
}
