//! CLI binary for running inference with a quantized 1.58-bit ternary model.
//!
//! Usage: cargo run --release --bin infer -- --model models/my_model_1_58bit --prompt "Hello"

use std::env;
use std::path::Path;
use std::process;

use cpu_ai_framework::inference::format::PackedModel;
use cpu_ai_framework::inference::transformer::{TernaryTransformer, MixingMode};
use cpu_ai_framework::inference::sampler::SamplingStrategy;
use cpu_ai_framework::inference::generate::{generate_streaming, GenerateConfig};
use cpu_ai_framework::inference::tokenizer::TokenizerWrapper;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut model_path = String::new();
    let mut prompt = String::from("Hello");
    let mut max_tokens: usize = 256;
    let mut temperature: f32 = 0.7;
    let mut top_k: usize = 40;
    let mut kv_budget_mb: usize = 0; // 0 = unlimited
    let mut mixing_mode = MixingMode::Attention;
    let mut doc_boundary_token: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_path = args[i].clone(); }
            "--prompt" => { i += 1; prompt = args[i].clone(); }
            "--max-tokens" => { i += 1; max_tokens = args[i].parse().unwrap_or(256); }
            "--temperature" => { i += 1; temperature = args[i].parse().unwrap_or(0.7); }
            "--top-k" => { i += 1; top_k = args[i].parse().unwrap_or(40); }
            "--kv-budget" => { i += 1; kv_budget_mb = args[i].parse().unwrap_or(0); }
            "--mixing-mode" => {
                i += 1;
                mixing_mode = match args[i].as_str() {
                    "attention" => MixingMode::Attention,
                    "mamba" => MixingMode::Mamba,
                    "hybrid" => MixingMode::Hybrid,
                    other => {
                        eprintln!("Unknown mixing mode: '{}'. Use: attention, mamba, hybrid", other);
                        process::exit(1);
                    }
                };
            }
            "--doc-boundary" => {
                i += 1;
                doc_boundary_token = Some(args[i].clone());
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage();
                process::exit(1);
            }
        }
        i += 1;
    }

    if model_path.is_empty() {
        eprintln!("Error: --model path is required.");
        print_usage();
        process::exit(1);
    }

    // Load model
    println!("Loading model from: {}", model_path);
    let packed = match PackedModel::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            process::exit(1);
        }
    };

    println!("Building transformer ({} layers, hidden_dim={}, vocab={})...",
        packed.manifest.num_layers, packed.manifest.hidden_dim, packed.manifest.vocab_size);
    let mut model = TernaryTransformer::from_packed(&packed);

    // Set mixing mode (attention/mamba/hybrid)
    model.mixing_mode = mixing_mode;
    if mixing_mode != MixingMode::Attention {
        if model.mamba_cache.is_none() {
            eprintln!("Warning: --mixing-mode {:?} requested but model has no Mamba weights. Falling back to attention.", mixing_mode);
            model.mixing_mode = MixingMode::Attention;
        } else {
            println!("Mixing mode: {:?}", mixing_mode);
        }
    }

    // Set KV cache memory budget if specified
    if kv_budget_mb > 0 {
        model.kv_cache.max_memory_bytes = kv_budget_mb * 1024 * 1024;
        println!("KV cache budget: {} MB", kv_budget_mb);
    }

    // Load tokenizer — look for tokenizer.json in the model directory
    let model_dir = Path::new(&model_path);
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = match TokenizerWrapper::from_file(&tokenizer_path) {
        Ok(t) => {
            println!("Loaded tokenizer from {}", tokenizer_path.display());
            Some(t)
        }
        Err(e) => {
            eprintln!("Warning: Could not load tokenizer ({}). Falling back to byte-level.", e);
            None
        }
    };

    // Tokenize prompt
    let (prompt_tokens, eos_id) = if let Some(ref tok) = tokenizer {
        let ids = tok.encode(&prompt, true).unwrap_or_else(|e| {
            eprintln!("Tokenization failed: {}", e);
            process::exit(1);
        });
        (ids, tok.eos_token_id)
    } else {
        let ids: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();
        (ids, 2)
    };

    let strategy = if temperature < 0.01 {
        SamplingStrategy::Greedy
    } else {
        SamplingStrategy::TopK { k: top_k, temperature }
    };

    // Resolve document boundary tokens for Document-level RoPE
    let doc_boundary_tokens = if let Some(ref boundary_str) = doc_boundary_token {
        // Try to encode the boundary string as a token ID
        if let Ok(id) = boundary_str.parse::<u32>() {
            // Numeric: treat as raw token ID
            println!("Document-level RoPE enabled (boundary token ID: {})", id);
            vec![id]
        } else if let Some(ref tok) = tokenizer {
            // String: encode it to get the token ID(s)
            let ids = tok.encode(boundary_str, false).unwrap_or_default();
            if ids.is_empty() {
                eprintln!("Warning: --doc-boundary '{}' produced no tokens. Document RoPE disabled.", boundary_str);
                Vec::new()
            } else {
                println!("Document-level RoPE enabled (boundary tokens: {:?})", ids);
                ids
            }
        } else {
            eprintln!("Warning: --doc-boundary requires a tokenizer or numeric token ID.");
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let config = GenerateConfig {
        max_tokens,
        strategy,
        eos_token_id: eos_id,
        doc_boundary_tokens,
    };

    println!("\n--- Generating (max {} tokens) ---", max_tokens);
    println!("Prompt: {}\n", prompt);

    // Stream tokens to stdout as they're generated
    use std::io::Write;
    let result = generate_streaming(&mut model, &prompt_tokens, &config, |token_id| {
        if let Some(ref tok) = tokenizer {
            let piece = tok.decode_token(token_id);
            print!("{}", piece);
        } else if token_id < 256 {
            print!("{}", token_id as u8 as char);
        }
        let _ = std::io::stdout().flush();
    });

    println!();
    println!("\n--- Stats ---");
    println!("Prefill:   {:.1} ms ({} tokens)", result.prefill_time_ms, prompt_tokens.len());
    println!("Generated: {} tokens in {:.1} ms", result.token_ids.len(), result.total_time_ms);
    println!("Speed:     {:.1} tokens/sec", result.tokens_per_second);
}

fn print_usage() {
    println!("Usage: infer --model <path> [--prompt <text>] [--max-tokens <n>] [--temperature <f>] [--top-k <n>] [--kv-budget <MB>] [--mixing-mode <mode>] [--doc-boundary <token>]");
    println!();
    println!("Options:");
    println!("  --model          Path to quantized model dir (manifest.json + weights.bin + tokenizer.json)");
    println!("  --prompt         Input prompt text (default: \"Hello\")");
    println!("  --max-tokens     Maximum tokens to generate (default: 256)");
    println!("  --temperature    Sampling temperature (default: 0.7, use 0 for greedy)");
    println!("  --top-k          Top-K sampling (default: 40)");
    println!("  --kv-budget      KV cache memory budget in MB (default: unlimited)");
    println!("  --mixing-mode    Sequence mixing: attention (default), mamba, or hybrid");
    println!("  --doc-boundary   Token or ID marking document boundaries for RoPE reset");
    println!("                   (e.g. \"<doc>\" or a numeric token ID like 2)");
}
