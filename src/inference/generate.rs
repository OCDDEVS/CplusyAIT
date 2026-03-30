//! Autoregressive text generation loop with streaming support.

use crate::inference::transformer::TernaryTransformer;
use crate::inference::sampler::{SamplingStrategy, sample};
use std::time::Instant;

/// Configuration for text generation.
pub struct GenerateConfig {
    pub max_tokens: usize,
    pub strategy: SamplingStrategy,
    /// EOS token ID — generation stops when this is produced.
    pub eos_token_id: u32,
    /// Token IDs that mark document boundaries for Document-level RoPE.
    /// When non-empty, enables document-level position resets during generation.
    /// The RoPE position counter resets to 0 after each boundary token,
    /// allowing multi-document prompts to avoid RoPE frequency overflow.
    pub doc_boundary_tokens: Vec<u32>,
}

/// Result of a generation run.
pub struct GenerateResult {
    pub token_ids: Vec<u32>,
    pub total_time_ms: f64,
    pub tokens_per_second: f64,
    pub prefill_time_ms: f64,
}

/// Run autoregressive generation given a prompt as token IDs.
/// Non-streaming version — collects all tokens then returns.
pub fn generate(
    model: &mut TernaryTransformer,
    prompt_tokens: &[u32],
    config: &GenerateConfig,
) -> GenerateResult {
    generate_streaming(model, prompt_tokens, config, |_| {})
}

/// Run autoregressive generation with a callback fired on each new token.
/// `on_token` receives each generated token ID as it's produced,
/// enabling real-time streaming output.
pub fn generate_streaming<F>(
    model: &mut TernaryTransformer,
    prompt_tokens: &[u32],
    config: &GenerateConfig,
    mut on_token: F,
) -> GenerateResult
where
    F: FnMut(u32),
{
    model.kv_cache.clear();
    if let Some(ref mut mc) = model.mamba_cache {
        mc.clear();
    }

    // Configure document-level RoPE if boundary tokens are specified
    if !config.doc_boundary_tokens.is_empty() {
        model.doc_rope = crate::inference::transformer::DocumentRoPE::new(
            config.doc_boundary_tokens.clone(),
        );
    } else {
        model.doc_rope.reset();
    }

    let mut output_tokens: Vec<u32> = Vec::new();

    let start = Instant::now();

    // Prefill: process all prompt tokens in one batch (skips LM head for intermediate tokens)
    let last_logits = model.forward_prefill(prompt_tokens);

    let prefill_time = start.elapsed();
    let prefill_ms = prefill_time.as_secs_f64() * 1000.0;

    // Decode: generate new tokens one at a time
    let mut pos = prompt_tokens.len();
    let mut next_token = sample(&last_logits, &config.strategy);

    for _ in 0..config.max_tokens {
        if next_token == config.eos_token_id {
            break;
        }
        output_tokens.push(next_token);
        on_token(next_token);

        let logits = model.forward_token(next_token, pos);
        pos += 1;

        // Enforce memory budget — evict oldest KV entries if over limit
        model.kv_cache.enforce_budget();

        next_token = sample(&logits, &config.strategy);
    }

    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let decode_tokens = output_tokens.len() as f64;
    let tps = if total_ms > 0.0 { decode_tokens / (total_ms / 1000.0) } else { 0.0 };

    GenerateResult {
        token_ids: output_tokens,
        total_time_ms: total_ms,
        tokens_per_second: tps,
        prefill_time_ms: prefill_ms,
    }
}
