use std::path::Path;
use candle_core::{Device, Tensor, IndexOp};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use std::time::Instant;

use cpu_ai_framework::core::infer::{Config, Model};

fn get_memory_usage() -> f64 {
    if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
        if let Some(pages) = statm.split_whitespace().nth(1) {
            if let Ok(pages) = pages.parse::<usize>() {
                return (pages * 4096) as f64 / 1_048_576.0;
            }
        }
    }
    0.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Next-Gen CPU/GPU AI Framework ---");
    println!("--- 1.58-bit Inference Engine ---\n");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <path_to_converted_model_dir>", args[0]);
        println!("Example: {} models/qwen1_5-0.5b-chat_1_58bit", args[0]);
        return Ok(());
    }

    let model_dir = Path::new(&args[1]);
    let safetensors_path = model_dir.join("model.safetensors");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !safetensors_path.exists() {
        return Err(format!("Model file not found: {}", safetensors_path.display()).into());
    }
    if !tokenizer_path.exists() {
        return Err(format!("Tokenizer file not found: {}", tokenizer_path.display()).into());
    }

    println!("Loading Tokenizer...");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Error loading tokenizer: {}", e))?;

    println!("Loading 1.58-bit Model Weights (Mmap)...");
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[safetensors_path], candle_core::DType::U8, &device)? };

    let config = Config::qwen_1_5b();

    println!("Initializing Network Architecture...");
    let model = Model::new(vb, &config)?;

    let initial_ram = get_memory_usage();
    println!("Model Loaded! Current RAM Usage: {:.2} MB\n", initial_ram);

    let prompt = "The capital of France is";
    println!("Prompt: {}", prompt);

    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("Encoding failed: {}", e))?;
    let mut tokens = encoding.get_ids().to_vec();

    print!("Generating: {}", prompt);
    use std::io::Write;
    std::io::stdout().flush()?;

    let start_time = Instant::now();
    let max_gen_len = 20;

    for _ in 0..max_gen_len {
        let input_tensor = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
        let positions: Vec<usize> = (0..tokens.len()).collect();

        let logits = model.forward(&input_tensor, &positions)?;

        let (_b_sz, seq_len, _vocab_size) = logits.dims3()?;
        let last_logit = logits.i((.., seq_len - 1, ..))?.squeeze(0)?;

        let next_token = last_logit.argmax(0)?.to_scalar::<u32>()?;

        tokens.push(next_token);

        let token_str = tokenizer.decode(&[next_token], true).map_err(|e| format!("{}", e))?;
        print!("{}", token_str);
        std::io::stdout().flush()?;

        if next_token == 151643 {
            break;
        }
    }

    let duration = start_time.elapsed();
    let tokens_per_sec = max_gen_len as f64 / duration.as_secs_f64();

    println!("\n\n--- Generation Complete ---");
    println!("Generated {} tokens in {:.2}s", max_gen_len, duration.as_secs_f64());
    println!("Speed: {:.2} Tokens / sec", tokens_per_sec);
    println!("Final RAM Usage: {:.2} MB", get_memory_usage());

    Ok(())
}
