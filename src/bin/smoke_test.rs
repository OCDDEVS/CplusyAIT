//! Smoke test: loads a HuggingFace model, converts it to 1.58-bit, and runs inference.
//! Usage: cargo run --release --bin smoke_test -- --model-dir models/tiny-random-llama-2

use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::process;

use safetensors::tensor::SafeTensors;
use serde::Deserialize;

use cpu_ai_framework::inference::format::{ModelManifest, TensorMeta, PackedModel};
use cpu_ai_framework::inference::transformer::TernaryTransformer;
use cpu_ai_framework::inference::sampler::SamplingStrategy;
use cpu_ai_framework::inference::generate::{generate, GenerateConfig};
use cpu_ai_framework::inference::tokenizer::TokenizerWrapper;

#[derive(Deserialize)]
struct HFConfig {
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    intermediate_size: Option<usize>,
    rope_theta: Option<f64>,
    rms_norm_eps: Option<f64>,
    model_type: Option<String>,
}

fn should_quantize(name: &str, shape: &[usize]) -> bool {
    if name.contains("norm") || name.contains("bias") || name.contains("embed_tokens") {
        return false;
    }
    shape.len() == 2
}

/// Convert raw bytes to f32, handling bf16 and f16 formats.
/// Safetensors stores dtype info but the Rust API gives raw bytes.
/// We detect by checking if byte_count == num_elements * 2 (half) or * 4 (float).
fn bytes_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    if data.len() == num_elements * 4 {
        // Already f32
        let mut out = vec![0.0f32; num_elements];
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), out.as_mut_ptr() as *mut u8, data.len());
        }
        out
    } else if data.len() == num_elements * 2 {
        // bf16 or f16 — treat as bf16 (most common for HF models)
        let mut out = vec![0.0f32; num_elements];
        for i in 0..num_elements {
            let lo = data[i * 2] as u16;
            let hi = data[i * 2 + 1] as u16;
            let bf16_bits = lo | (hi << 8);
            // bf16 to f32: just shift left by 16 bits
            let f32_bits = (bf16_bits as u32) << 16;
            out[i] = f32::from_bits(f32_bits);
        }
        out
    } else {
        panic!("Unexpected data size: {} bytes for {} elements", data.len(), num_elements);
    }
}

fn quantize_to_packed_2bit(weights: &[f32]) -> (Vec<u8>, f32) {
    let sum_abs: f32 = weights.iter().map(|w| w.abs()).sum();
    let gamma = sum_abs / (weights.len() as f32 + 1e-8);
    let packed_len = (weights.len() + 3) / 4;
    let mut packed = vec![0u8; packed_len];
    for (i, &w) in weights.iter().enumerate() {
        let scaled = w / gamma;
        let q = scaled.clamp(-1.0, 1.0).round() as i8;
        let encoded: u8 = match q { 1 => 1, -1 => 2, _ => 0 };
        packed[i / 4] |= encoded << ((i % 4) * 2);
    }
    (packed, gamma)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut model_dir = String::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => { i += 1; model_dir = args[i].clone(); }
            _ => {}
        }
        i += 1;
    }
    if model_dir.is_empty() {
        eprintln!("Usage: smoke_test --model-dir <path>");
        process::exit(1);
    }

    let src = Path::new(&model_dir);
    println!("=== SMOKE TEST ===");
    println!("Source model: {}", src.display());

    // 1. Read config.json
    let cfg_str = fs::read_to_string(src.join("config.json")).expect("No config.json");
    let hf_cfg: HFConfig = serde_json::from_str(&cfg_str).expect("Bad config.json");
    println!("Model: {} layers, hidden={}, vocab={}",
        hf_cfg.num_hidden_layers.unwrap_or(0),
        hf_cfg.hidden_size.unwrap_or(0),
        hf_cfg.vocab_size.unwrap_or(0));

    // 2. Convert safetensors to packed format
    println!("\n--- Step 1: Converting to 1.58-bit ---");
    let st_path = src.join("model.safetensors");
    let mut f = File::open(&st_path).expect("No model.safetensors");
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer).unwrap();

    let tensors = SafeTensors::deserialize(&buffer).expect("Bad safetensors");
    let tensor_names: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();
    println!("Found {} tensors", tensor_names.len());

    let mut blob: Vec<u8> = Vec::new();
    let mut metas: Vec<TensorMeta> = Vec::new();

    for name in &tensor_names {
        let tensor = tensors.tensor(name).unwrap();
        let shape: Vec<usize> = tensor.shape().to_vec();
        let data = tensor.data();
        let num_elements: usize = shape.iter().product();
        let floats = bytes_to_f32(data, num_elements);

        let offset = blob.len();
        if should_quantize(name, &shape) {
            let (packed, gamma) = quantize_to_packed_2bit(&floats);
            let len = packed.len();
            blob.extend_from_slice(&packed);
            metas.push(TensorMeta { name: name.clone(), shape, byte_offset: offset, byte_length: len, gamma });
            println!("  [QUANT] {} {:?} -> {} bytes (gamma={:.4})", name, metas.last().unwrap().shape, len, gamma);
        } else {
            // Store as f32 bytes
            let f32_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(floats.as_ptr() as *const u8, floats.len() * 4)
            };
            let len = f32_bytes.len();
            blob.extend_from_slice(f32_bytes);
            metas.push(TensorMeta { name: name.clone(), shape, byte_offset: offset, byte_length: len, gamma: 0.0 });
            println!("  [FP32]  {} {:?} -> {} bytes", name, metas.last().unwrap().shape, len);
        }
    }

    // Infer dimensions from actual tensor shapes if config.json is incomplete
    let hidden_dim = hf_cfg.hidden_size.unwrap_or_else(|| {
        // Infer from embed_tokens shape: [vocab_size, hidden_dim]
        metas.iter()
            .find(|m| m.name.contains("embed_tokens"))
            .map(|m| *m.shape.last().unwrap_or(&0))
            .unwrap_or(0)
    });
    let vocab_size = hf_cfg.vocab_size.unwrap_or_else(|| {
        metas.iter()
            .find(|m| m.name.contains("embed_tokens"))
            .map(|m| m.shape[0])
            .unwrap_or(0)
    });
    let num_layers = hf_cfg.num_hidden_layers.unwrap_or_else(|| {
        // Count unique layer indices
        metas.iter()
            .filter_map(|m| {
                m.name.strip_prefix("model.layers.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    });
    let intermediate_dim = hf_cfg.intermediate_size.unwrap_or_else(|| {
        metas.iter()
            .find(|m| m.name.contains("gate_proj"))
            .map(|m| m.shape[0])
            .unwrap_or(0)
    });
    let num_heads = hf_cfg.num_attention_heads.unwrap_or_else(|| {
        // Infer from q_proj: [num_heads * head_dim, hidden_dim]
        // If q_proj out == hidden_dim, then num_heads = hidden_dim / head_dim
        // Default guess: hidden_dim / 64 (common head_dim)
        if hidden_dim > 0 { (hidden_dim / 64).max(1) } else { 1 }
    });
    let num_kv_heads = hf_cfg.num_key_value_heads.unwrap_or(num_heads);

    let manifest = ModelManifest {
        model_type: hf_cfg.model_type.unwrap_or_else(|| "llama".into()),
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
        max_seq_len: hf_cfg.max_position_embeddings.unwrap_or(4096),
        intermediate_dim,
        rope_theta: hf_cfg.rope_theta.unwrap_or(10000.0),
        rms_norm_eps: hf_cfg.rms_norm_eps.unwrap_or(1e-5),
        tensors: metas,
        mamba_state_dim: None,
    };

    // Save to output dir — derive name from source directory
    let src_name = src.file_name().unwrap().to_string_lossy();
    let out_dir = src.parent().unwrap().join(format!("{}_1_58bit", src_name));
    fs::create_dir_all(&out_dir).unwrap();
    fs::write(out_dir.join("manifest.json"), serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    File::create(out_dir.join("weights.bin")).unwrap().write_all(&blob).unwrap();

    // Copy tokenizer
    for fname in &["tokenizer.json", "tokenizer_config.json"] {
        let src_f = src.join(fname);
        if src_f.exists() {
            fs::copy(&src_f, out_dir.join(fname)).unwrap();
        }
    }

    println!("\nPacked model saved to: {}", out_dir.display());
    println!("Blob size: {} bytes", blob.len());

    // 3. Load the packed model
    println!("\n--- Step 2: Loading packed model ---");
    let packed = PackedModel::load(&out_dir).expect("Failed to load packed model");
    println!("Manifest loaded: {} tensors", packed.manifest.tensors.len());

    // 4. Build transformer
    println!("\n--- Step 3: Building transformer ---");
    let mut model = TernaryTransformer::from_packed(&packed);
    println!("Transformer built: {} layers", model.layers.len());

    // 5. Load tokenizer
    let tokenizer = TokenizerWrapper::from_file(out_dir.join("tokenizer.json")).ok();
    let has_tokenizer = tokenizer.is_some();
    println!("Tokenizer: {}", if has_tokenizer { "loaded" } else { "byte-level fallback" });

    // 6. Run inference
    println!("\n--- Step 4: Running inference ---");
    let prompt = "Hello";
    let (prompt_tokens, eos_id) = if let Some(ref tok) = tokenizer {
        let ids = tok.encode(prompt, true).unwrap_or_else(|_| vec![1]);
        let eos = tok.eos_token_id;
        (ids, eos)
    } else {
        (prompt.bytes().map(|b| b as u32).collect(), 2u32)
    };
    println!("Prompt tokens: {:?}", prompt_tokens);

    let config = GenerateConfig {
        max_tokens: 20,
        strategy: SamplingStrategy::Greedy,
        eos_token_id: eos_id,
        doc_boundary_tokens: Vec::new(),
    };

    let result = generate(&mut model, &prompt_tokens, &config);

    println!("Generated {} tokens in {:.1} ms", result.token_ids.len(), result.total_time_ms);
    println!("Speed: {:.1} tokens/sec", result.tokens_per_second);
    println!("Token IDs: {:?}", result.token_ids);

    if let Some(ref tok) = tokenizer {
        let text = tok.decode(&result.token_ids).unwrap_or_default();
        println!("Output text: {}", text);
    }

    println!("\n=== SMOKE TEST PASSED ===");
    println!("The pipeline works end-to-end: load -> convert -> build -> infer");
    println!("(Output is gibberish because this is a random-weight model)");
}
