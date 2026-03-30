//! Transformer decoder block and full model for ternary inference.
//! Implements RMSNorm, GQA with RoPE, SwiGLU FFN — all using TernaryLinear.

use crate::inference::ternary_linear::TernaryLinear;
use crate::inference::kv_cache::KVCache;
use crate::inference::mamba::{MambaLayer, MambaCache};
use crate::inference::format::{PackedModel, ModelManifest};
use rayon::prelude::*;

// ─── RMSNorm ────────────────────────────────────────────────────────────────

/// RMSNorm: x * weight / sqrt(mean(x^2) + eps)
/// Weight is kept in f32 (tiny — just `hidden_dim` floats).
pub struct RMSNorm {
    pub weight: Vec<f32>,
    pub eps: f64,
}

impl RMSNorm {
    pub fn forward(&self, x: &mut [f32]) {
        let n = x.len();
        let mut ss: f64 = 0.0;
        for &v in x.iter() {
            ss += (v as f64) * (v as f64);
        }
        ss = 1.0 / (ss / n as f64 + self.eps).sqrt();
        for i in 0..n {
            x[i] = (x[i] as f64 * ss * self.weight[i] as f64) as f32;
        }
    }
}

// ─── Transformer Layer ──────────────────────────────────────────────────────

pub struct TransformerLayer {
    // Attention
    pub attn_norm: RMSNorm,
    pub q_proj: TernaryLinear,
    pub k_proj: TernaryLinear,
    pub v_proj: TernaryLinear,
    pub o_proj: TernaryLinear,

    // FFN (SwiGLU)
    pub ffn_norm: RMSNorm,
    pub gate_proj: TernaryLinear, // W1
    pub up_proj: TernaryLinear,   // W3
    pub down_proj: TernaryLinear, // W2

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,

    /// Optional Mamba SSM block for hybrid attention/SSM layers.
    /// When present, enables O(N) linear-time sequence mixing as an
    /// alternative to quadratic self-attention.
    pub mamba: Option<MambaLayer>,
}

impl TransformerLayer {
    /// Forward pass for a single token position (autoregressive decoding).
    /// `x` is the hidden state of length `hidden_dim`.
    /// `pos` is the current sequence position (for RoPE).
    /// `kv` is this layer's KV cache.
    /// `causal_len`: if Some(n), apply causal mask so this token can only attend
    /// to KV positions 0..=n. During decode this is None (we naturally only see past).
    /// During prefill, pass Some(pos) to enforce causality.
    pub fn forward(
        &self,
        x: &mut Vec<f32>,
        pos: usize,
        kv: &mut crate::inference::kv_cache::LayerKVCache,
        causal_len: Option<usize>,
    ) {
        let hidden_dim = x.len();
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;
        let kv_groups = num_heads / num_kv_heads;

        // ── Attention ──
        let mut residual = x.clone();
        self.attn_norm.forward(x);

        let q_out = self.q_proj.forward(x, 1);
        let k_out = self.k_proj.forward(x, 1);
        let v_out = self.v_proj.forward(x, 1);

        let mut q = q_out;
        let mut k = k_out.clone();
        apply_rope(&mut q, num_heads, head_dim, pos, self.rope_theta);
        apply_rope(&mut k, num_kv_heads, head_dim, pos, self.rope_theta);

        kv.append(&k, &v_out);

        // How many KV positions this token is allowed to attend to (causal mask)
        let attend_len = match causal_len {
            Some(cl) => (cl + 1).min(kv.seq_len), // attend to positions 0..=cl
            None => kv.seq_len,                     // attend to everything (decode mode)
        };

        let mut attn_output = vec![0.0f32; num_heads * head_dim];

        // Parallel attention across heads using Rayon
        let head_outputs: Vec<Vec<f32>> = (0..num_heads).into_par_iter().map(|h| {
            let kv_h = h / kv_groups;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = vec![0.0f32; attend_len];
            let scale = 1.0 / (head_dim as f32).sqrt();

            for s in 0..attend_len {
                let k_offset = s * num_kv_heads * head_dim + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_head[d] * kv.keys[k_offset + d];
                }
                scores[s] = dot * scale;
            }

            softmax_inplace(&mut scores);

            let mut out_head = vec![0.0f32; head_dim];
            for s in 0..attend_len {
                let v_offset = s * num_kv_heads * head_dim + kv_h * head_dim;
                let w = scores[s];
                for d in 0..head_dim {
                    out_head[d] += w * kv.values[v_offset + d];
                }
            }
            out_head
        }).collect();

        // Gather parallel results
        for (h, head_out) in head_outputs.into_iter().enumerate() {
            attn_output[h * head_dim..(h + 1) * head_dim].copy_from_slice(&head_out);
        }

        let o_out = self.o_proj.forward(&attn_output, 1);

        // Residual connection
        for i in 0..hidden_dim {
            x[i] = residual[i] + o_out[i];
        }

        // ── FFN (SwiGLU) ──
        residual = x.clone();
        self.ffn_norm.forward(x);

        let gate = self.gate_proj.forward(x, 1);
        let up = self.up_proj.forward(x, 1);

        // SwiGLU: silu(gate) * up
        let mut ffn_hidden = vec![0.0f32; gate.len()];
        for i in 0..gate.len() {
            let silu = gate[i] * sigmoid(gate[i]);
            ffn_hidden[i] = silu * up[i];
        }

        let down = self.down_proj.forward(&ffn_hidden, 1);

        // Residual connection
        for i in 0..hidden_dim {
            x[i] = residual[i] + down[i];
        }
    }

    /// MSA-routed forward: instead of attending to the full KV cache,
    /// use the C++ MSA router to select only the top-k most relevant positions.
    /// This gives O(k) attention instead of O(seq_len) for long contexts.
    pub fn forward_msa(
        &self,
        x: &mut Vec<f32>,
        pos: usize,
        kv: &mut crate::inference::kv_cache::LayerKVCache,
        top_k: usize,
    ) {
        let hidden_dim = x.len();
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;
        let kv_groups = num_heads / num_kv_heads;

        let mut residual = x.clone();
        self.attn_norm.forward(x);

        let q_out = self.q_proj.forward(x, 1);
        let k_out = self.k_proj.forward(x, 1);
        let v_out = self.v_proj.forward(x, 1);

        let mut q = q_out;
        let mut k = k_out.clone();
        apply_rope(&mut q, num_heads, head_dim, pos, self.rope_theta);
        apply_rope(&mut k, num_kv_heads, head_dim, pos, self.rope_theta);

        // Append current token to full cache (we always keep the full history)
        kv.append(&k, &v_out);

        // Use MSA router to select top-k relevant positions from the cache
        // We use the query vector (averaged across heads) as the routing query
        let kv_dim = num_kv_heads * head_dim;
        let (sel_keys, sel_values, _indices) = kv.msa_select_top_k(&q[..kv_dim.min(q.len())], top_k);
        let sel_len = sel_keys.len() / kv_dim;

        let mut attn_output = vec![0.0f32; num_heads * head_dim];

        let head_outputs: Vec<Vec<f32>> = (0..num_heads).into_par_iter().map(|h| {
            let kv_h = h / kv_groups;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = vec![0.0f32; sel_len];
            let scale = 1.0 / (head_dim as f32).sqrt();

            for s in 0..sel_len {
                let k_offset = s * num_kv_heads * head_dim + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_head[d] * sel_keys[k_offset + d];
                }
                scores[s] = dot * scale;
            }

            softmax_inplace(&mut scores);

            let mut out_head = vec![0.0f32; head_dim];
            for s in 0..sel_len {
                let v_offset = s * num_kv_heads * head_dim + kv_h * head_dim;
                let w = scores[s];
                for d in 0..head_dim {
                    out_head[d] += w * sel_values[v_offset + d];
                }
            }
            out_head
        }).collect();

        for (h, head_out) in head_outputs.into_iter().enumerate() {
            attn_output[h * head_dim..(h + 1) * head_dim].copy_from_slice(&head_out);
        }

        let o_out = self.o_proj.forward(&attn_output, 1);
        for i in 0..hidden_dim {
            x[i] = residual[i] + o_out[i];
        }

        // FFN is identical
        residual = x.clone();
        self.ffn_norm.forward(x);
        let gate = self.gate_proj.forward(x, 1);
        let up = self.up_proj.forward(x, 1);
        let mut ffn_hidden = vec![0.0f32; gate.len()];
        for i in 0..gate.len() {
            ffn_hidden[i] = gate[i] * sigmoid(gate[i]) * up[i];
        }
        let down = self.down_proj.forward(&ffn_hidden, 1);
        for i in 0..hidden_dim {
            x[i] = residual[i] + down[i];
        }
    }

    /// Mamba SSM forward: replaces self-attention with O(1)-per-token recurrent
    /// state update. The FFN still runs as normal (SwiGLU with ternary kernels).
    /// This is the "Tri-Hybrid" path from the GPU research doc.
    pub fn forward_mamba(
        &self,
        x: &mut Vec<f32>,
        mamba_state: &mut crate::inference::mamba::MambaState,
    ) {
        let hidden_dim = x.len();

        // ── Mamba SSM (replaces attention) ──
        let mut residual = x.clone();
        self.attn_norm.forward(x);

        let mamba = self.mamba.as_ref().expect("forward_mamba called on layer without MambaLayer");
        let ssm_out = mamba.forward(x, mamba_state);

        // Residual connection
        for i in 0..hidden_dim {
            x[i] = residual[i] + ssm_out[i];
        }

        // ── FFN (SwiGLU) — identical to attention path ──
        residual = x.clone();
        self.ffn_norm.forward(x);
        let gate = self.gate_proj.forward(x, 1);
        let up = self.up_proj.forward(x, 1);
        let mut ffn_hidden = vec![0.0f32; gate.len()];
        for i in 0..gate.len() {
            ffn_hidden[i] = gate[i] * sigmoid(gate[i]) * up[i];
        }
        let down = self.down_proj.forward(&ffn_hidden, 1);
        for i in 0..hidden_dim {
            x[i] = residual[i] + down[i];
        }
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

pub struct TernaryTransformer {
    pub layers: Vec<TransformerLayer>,
    pub final_norm: RMSNorm,
    pub embed_tokens: Vec<f32>,
    pub lm_head: Option<TernaryLinear>,
    pub config: ModelManifest,
    pub kv_cache: KVCache,
    /// When set, use MSA routing to attend to only the top-k most relevant
    /// cached positions instead of the full cache. 0 = disabled (full attention).
    pub msa_top_k: usize,
    /// Mamba recurrent state cache. Present when any layer has a MambaLayer.
    pub mamba_cache: Option<MambaCache>,
    /// Sequence mixing mode for inference.
    /// - `Attention`: standard quadratic self-attention (default)
    /// - `Mamba`: O(1) per-token recurrent SSM (requires mamba weights)
    /// - `Hybrid`: even layers use attention, odd layers use Mamba
    pub mixing_mode: MixingMode,
    /// Document-level RoPE position tracker.
    /// When enabled, the position counter resets at document boundaries,
    /// allowing a model trained on short contexts to extrapolate to massive
    /// multi-document knowledge bases without RoPE frequency overflow.
    pub doc_rope: DocumentRoPE,
}

/// Controls how sequence mixing is performed during inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MixingMode {
    /// Standard quadratic self-attention on all layers.
    Attention,
    /// Mamba SSM on all layers (requires mamba weights loaded).
    Mamba,
    /// Alternating: even layers use attention, odd layers use Mamba.
    Hybrid,
}

/// Document-level RoPE: resets the position counter at document boundaries.
///
/// From the research doc: "The position counter resets to zero for new documents
/// (MemCells). This allows a model trained on short contexts to seamlessly
/// extrapolate to massive personal knowledge bases on the user's local machine."
///
/// When `enabled` is false, positions are monotonically increasing (standard behavior).
/// When `enabled` is true, positions reset at each document boundary.
pub struct DocumentRoPE {
    /// Whether document-level position resets are active.
    pub enabled: bool,
    /// Current position within the current document (resets at boundaries).
    pub local_pos: usize,
    /// Token IDs that mark document boundaries (position resets after these).
    /// Common choices: BOS token, special `<doc>` separator, or EOS token.
    pub boundary_token_ids: Vec<u32>,
}

impl DocumentRoPE {
    pub fn disabled() -> Self {
        Self { enabled: false, local_pos: 0, boundary_token_ids: Vec::new() }
    }

    pub fn new(boundary_token_ids: Vec<u32>) -> Self {
        Self { enabled: true, local_pos: 0, boundary_token_ids }
    }

    /// Compute the RoPE position for a token. If the token is a document
    /// boundary, reset the local position counter.
    /// Returns the position to use for RoPE encoding.
    pub fn next_pos(&mut self, token_id: u32) -> usize {
        if self.boundary_token_ids.contains(&token_id) {
            self.local_pos = 0;
        }
        let pos = self.local_pos;
        self.local_pos += 1;
        pos
    }

    /// Compute RoPE positions for a batch of tokens (prefill).
    /// Returns a Vec of positions, one per token.
    pub fn compute_positions(&self, token_ids: &[u32]) -> Vec<usize> {
        let mut positions = Vec::with_capacity(token_ids.len());
        let mut local = 0usize;
        for &tok in token_ids {
            if self.boundary_token_ids.contains(&tok) {
                local = 0;
            }
            positions.push(local);
            local += 1;
        }
        positions
    }

    pub fn reset(&mut self) {
        self.local_pos = 0;
    }
}

impl TernaryTransformer {
    /// Load a model from a PackedModel (manifest + blob).
    pub fn from_packed(model: &PackedModel) -> Self {
        let cfg = &model.manifest;
        let hidden_dim = cfg.hidden_dim;
        let head_dim = hidden_dim / cfg.num_heads;

        // Detect if this model has Mamba SSM weights
        let mamba_state_dim = cfg.mamba_state_dim.unwrap_or(0);
        let has_mamba = mamba_state_dim > 0
            && resolve_tensor_name(model, "model.layers.0.mamba.x_proj").is_some();

        if has_mamba {
            println!("Detected Mamba SSM weights (state_dim={}). Hybrid mode available.", mamba_state_dim);
        }

        let mut layers = Vec::with_capacity(cfg.num_layers);
        let rope_theta = cfg.rope_theta as f32;
        for l in 0..cfg.num_layers {
            let prefix = format!("model.layers.{}", l);

            // Try to load Mamba weights for this layer
            let mamba = if has_mamba {
                let mp = format!("{}.mamba", prefix);
                let x_proj_name = format!("{}.x_proj", mp);
                if resolve_tensor_name(model, &x_proj_name).is_some() {
                    // Load A_log and D as f32 tensors
                    let a_log = load_f32_tensor(model, &format!("{}.a_log", mp));
                    let d_param = load_f32_tensor(model, &format!("{}.d", mp));
                    Some(MambaLayer {
                        x_proj: load_ternary_linear(model, &x_proj_name),
                        dt_proj: load_ternary_linear(model, &format!("{}.dt_proj", mp)),
                        b_proj: load_ternary_linear(model, &format!("{}.b_proj", mp)),
                        c_proj: load_ternary_linear(model, &format!("{}.c_proj", mp)),
                        out_proj: load_ternary_linear(model, &format!("{}.out_proj", mp)),
                        a_log,
                        d_param,
                        hidden_dim,
                        state_dim: mamba_state_dim,
                    })
                } else {
                    None
                }
            } else {
                None
            };

            layers.push(TransformerLayer {
                attn_norm: load_rms_norm(model, &format!("{}.input_layernorm", prefix), hidden_dim, cfg.rms_norm_eps),
                q_proj: load_ternary_linear(model, &format!("{}.self_attn.q_proj", prefix)),
                k_proj: load_ternary_linear(model, &format!("{}.self_attn.k_proj", prefix)),
                v_proj: load_ternary_linear(model, &format!("{}.self_attn.v_proj", prefix)),
                o_proj: load_ternary_linear(model, &format!("{}.self_attn.o_proj", prefix)),
                ffn_norm: load_rms_norm(model, &format!("{}.post_attention_layernorm", prefix), hidden_dim, cfg.rms_norm_eps),
                gate_proj: load_ternary_linear(model, &format!("{}.mlp.gate_proj", prefix)),
                up_proj: load_ternary_linear(model, &format!("{}.mlp.up_proj", prefix)),
                down_proj: load_ternary_linear(model, &format!("{}.mlp.down_proj", prefix)),
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim,
                rope_theta,
                mamba,
            });
        }

        let final_norm = load_rms_norm(model, "model.norm", hidden_dim, cfg.rms_norm_eps);
        let embed_tokens = load_f32_tensor(model, "model.embed_tokens");

        let lm_head = if resolve_tensor_name(model, "lm_head").is_some() {
            Some(load_ternary_linear(model, "lm_head"))
        } else {
            println!("Note: lm_head not found, tying to embed_tokens.");
            None
        };

        let kv_cache = KVCache::new(cfg.num_layers, cfg.num_kv_heads, head_dim);

        // Create Mamba cache if any layer has Mamba weights
        let mamba_cache = if has_mamba {
            Some(MambaCache::new(cfg.num_layers, hidden_dim, mamba_state_dim))
        } else {
            None
        };

        Self {
            layers,
            final_norm,
            embed_tokens,
            lm_head,
            config: cfg.clone(),
            kv_cache,
            msa_top_k: 0,
            mamba_cache,
            mixing_mode: MixingMode::Attention, // default to attention
            doc_rope: DocumentRoPE::disabled(),
        }
    }

    /// Run a single forward pass for one token (decode mode), returning logits.
    /// When document-level RoPE is enabled, the position is computed from the
    /// document tracker instead of the global sequence position.
    pub fn forward_token(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;

        // Compute RoPE position: document-local or global
        let rope_pos = if self.doc_rope.enabled {
            self.doc_rope.next_pos(token_id)
        } else {
            pos
        };

        let offset = token_id as usize * hidden_dim;
        let mut hidden: Vec<f32> = self.embed_tokens[offset..offset + hidden_dim].to_vec();

        let msa_k = self.msa_top_k;
        let mode = self.mixing_mode;

        for (i, layer) in self.layers.iter().enumerate() {
            let use_mamba = match mode {
                MixingMode::Attention => false,
                MixingMode::Mamba => layer.mamba.is_some(),
                MixingMode::Hybrid => layer.mamba.is_some() && (i % 2 == 1),
            };

            if use_mamba {
                let mamba_cache = self.mamba_cache.as_mut()
                    .expect("MixingMode requires mamba_cache");
                layer.forward_mamba(&mut hidden, &mut mamba_cache.layers[i]);
            } else if msa_k > 0 && self.kv_cache.layers[i].seq_len > msa_k * 2 {
                layer.forward_msa(&mut hidden, rope_pos, &mut self.kv_cache.layers[i], msa_k);
            } else {
                layer.forward(&mut hidden, rope_pos, &mut self.kv_cache.layers[i], None);
            }
        }

        self.final_norm.forward(&mut hidden);
        self.compute_logits(&hidden)
    }

    /// Prefill: process a batch of prompt tokens with causal masking.
    /// Only returns logits for the last token.
    /// In Mamba/Hybrid mode, tokens are processed sequentially through the
    /// recurrent state (O(N) total, O(1) memory per token).
    /// When document-level RoPE is enabled, positions reset at document boundaries.
    pub fn forward_prefill(&mut self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let num_tokens = token_ids.len();

        if num_tokens == 0 {
            return vec![0.0; self.config.vocab_size];
        }

        // Pre-compute all RoPE positions (document-local or sequential)
        let rope_positions = if self.doc_rope.enabled {
            self.doc_rope.compute_positions(token_ids)
        } else {
            (0..num_tokens).collect()
        };

        // Update the doc_rope tracker to the final state after prefill
        if self.doc_rope.enabled {
            if let Some(&last_pos) = rope_positions.last() {
                self.doc_rope.local_pos = last_pos + 1;
            }
        }

        let mode = self.mixing_mode;
        let mut last_hidden = Vec::new();

        for (seq_idx, &tok) in token_ids.iter().enumerate() {
            let offset = tok as usize * hidden_dim;
            let mut hidden: Vec<f32> = self.embed_tokens[offset..offset + hidden_dim].to_vec();
            let rope_pos = rope_positions[seq_idx];

            for (i, layer) in self.layers.iter().enumerate() {
                let use_mamba = match mode {
                    MixingMode::Attention => false,
                    MixingMode::Mamba => layer.mamba.is_some(),
                    MixingMode::Hybrid => layer.mamba.is_some() && (i % 2 == 1),
                };

                if use_mamba {
                    let mamba_cache = self.mamba_cache.as_mut()
                        .expect("MixingMode requires mamba_cache");
                    layer.forward_mamba(&mut hidden, &mut mamba_cache.layers[i]);
                } else {
                    layer.forward(&mut hidden, rope_pos, &mut self.kv_cache.layers[i], Some(seq_idx));
                }
            }

            if seq_idx == num_tokens - 1 {
                last_hidden = hidden;
            }
        }

        self.final_norm.forward(&mut last_hidden);
        self.compute_logits(&last_hidden)
    }


    /// Compute logits from hidden state. Uses lm_head if available, otherwise
    /// ties to embed_tokens (matmul hidden @ embed_tokens^T).
    fn compute_logits(&self, hidden: &[f32]) -> Vec<f32> {
        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(hidden, 1)
        } else {
            // Tied embeddings: logits[v] = dot(hidden, embed_tokens[v])
            let hidden_dim = self.config.hidden_dim;
            let vocab_size = self.config.vocab_size;
            let mut logits = vec![0.0f32; vocab_size];
            for v in 0..vocab_size {
                let offset = v * hidden_dim;
                let mut dot = 0.0f32;
                for d in 0..hidden_dim {
                    dot += hidden[d] * self.embed_tokens[offset + d];
                }
                logits[v] = dot;
            }
            logits
        }
    }
}

// ─── Helper functions ───────────────────────────────────────────────────────

/// Weight name mapping: tries the primary name, then common alternatives
/// used by different model families (Mistral, Phi, Qwen, etc.)
fn resolve_tensor_name<'a>(model: &'a PackedModel, primary: &str) -> Option<(&'a crate::inference::format::TensorMeta, &'a [u8])> {
    // Try primary name first
    if let Some(result) = model.get_tensor_data(primary) {
        return Some(result);
    }

    // Try with .weight suffix (HuggingFace convention for norm/embedding layers)
    let with_weight = format!("{}.weight", primary);
    if let Some(result) = model.get_tensor_data(&with_weight) {
        return Some(result);
    }

    // Common alternative naming patterns
    let alternatives: Vec<String> = vec![
        // Llama uses "model.layers.X" but some use just "layers.X"
        primary.replace("model.layers.", "layers."),
        primary.replace("model.layers.", "transformer.h."),
        // Phi uses "transformer.h.X" with different sub-names
        primary.replace("self_attn.q_proj", "mixer.Wqkv")
            .replace("self_attn.k_proj", "mixer.Wqkv")
            .replace("self_attn.v_proj", "mixer.Wqkv"),
        primary.replace("self_attn.o_proj", "mixer.out_proj"),
        // Qwen uses "transformer.h.X.attn" instead of "self_attn"
        primary.replace("self_attn.", "attn."),
        // Some models use "attention" instead of "self_attn"
        primary.replace("self_attn.", "attention."),
        // Norm naming variants
        primary.replace("input_layernorm", "ln_1")
            .replace("post_attention_layernorm", "ln_2"),
        primary.replace("input_layernorm", "attention_norm")
            .replace("post_attention_layernorm", "ffn_norm"),
        // MLP naming variants
        primary.replace("mlp.gate_proj", "mlp.w1")
            .replace("mlp.up_proj", "mlp.w3")
            .replace("mlp.down_proj", "mlp.w2"),
        // Final norm variants
        primary.replace("model.norm", "model.final_layernorm"),
        primary.replace("model.norm", "transformer.ln_f"),
        // Embedding variants
        primary.replace("model.embed_tokens", "transformer.wte"),
        primary.replace("model.embed_tokens", "model.embed_tokens.weight"),
    ];

    for alt in &alternatives {
        if let Some(result) = model.get_tensor_data(alt) {
            return Some(result);
        }
        // Also try with .weight suffix
        let alt_w = format!("{}.weight", alt);
        if let Some(result) = model.get_tensor_data(&alt_w) {
            return Some(result);
        }
    }

    None
}

fn load_ternary_linear(model: &PackedModel, name: &str) -> TernaryLinear {
    let (meta, data) = resolve_tensor_name(model, name)
        .unwrap_or_else(|| panic!("Missing tensor (tried alternatives): {}", name));
    let out_features = meta.shape[0];
    let in_features = meta.shape[1];

    // Check for per-channel gammas (written by GPTQ quantizer as "name.__gamma__")
    let gamma_name = format!("{}.__gamma__", meta.name);
    let channel_gammas = if let Some((_gm, gdata)) = model.get_tensor_data(&gamma_name) {
        // Per-channel gammas stored as f32 array
        let num_floats = gdata.len() / 4;
        let mut gammas = vec![0.0f32; num_floats];
        unsafe {
            std::ptr::copy_nonoverlapping(
                gdata.as_ptr(),
                gammas.as_mut_ptr() as *mut u8,
                gdata.len(),
            );
        }
        gammas
    } else {
        Vec::new()
    };

    if !channel_gammas.is_empty() {
        TernaryLinear::with_channel_gammas(data.to_vec(), in_features, out_features, meta.gamma, channel_gammas)
    } else {
        TernaryLinear::new(data.to_vec(), in_features, out_features, meta.gamma)
    }
}

fn load_rms_norm(model: &PackedModel, name: &str, dim: usize, eps: f64) -> RMSNorm {
    let weight = load_f32_tensor(model, name);
    assert_eq!(weight.len(), dim, "RMSNorm weight size mismatch for {}", name);
    RMSNorm { weight, eps }
}

fn load_f32_tensor(model: &PackedModel, name: &str) -> Vec<f32> {
    let (_meta, data) = resolve_tensor_name(model, name)
        .unwrap_or_else(|| panic!("Missing tensor (tried alternatives): {}", name));
    let num_floats = data.len() / 4;
    let mut out = vec![0.0f32; num_floats];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            data.len(),
        );
    }
    out
}

fn apply_rope(x: &mut [f32], num_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    for h in 0..num_heads {
        let base = h * head_dim;
        for d in (0..head_dim).step_by(2) {
            let freq = 1.0 / theta.powf(d as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = x[base + d];
            let x1 = x[base + d + 1];
            x[base + d] = x0 * cos - x1 * sin;
            x[base + d + 1] = x1 * cos + x0 * sin;
        }
    }
}

fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() { return; }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

