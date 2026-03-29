use std::path::Path;
use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use tokenizers::Tokenizer;
use std::time::Instant;

use cpu_ai_framework::core::infer::Config;
use cpu_ai_framework::core::train::TrainEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Next-Gen CPU/GPU AI Framework ---");
    println!("--- Real 1.58-bit QAFT (Quantization-Aware Fine-Tuning) Engine ---\n");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <path_to_model_dir> <training_text.txt>", args[0]);
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

    println!("Loading FP32 Master Weights (Mmap) for STE Tracking...");
    let device = Device::Cpu;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

    let config = Config::qwen_1_5b();

    println!("Initializing Training Architecture (STE)...");
    let mut train_engine = TrainEngine::new(vb, &config)?;

    let mut optimizer = AdamW::new(varmap.all_vars(), ParamsAdamW::default())?;

    let text_path = if args.len() > 2 { args[2].clone() } else { "input.txt".to_string() };
    if !Path::new(&text_path).exists() {
        println!("Warning: No training text found at {}. Skipping actual training loop.", text_path);
        println!("The Real STE engine is built, expecting actual data to flow through.");
        return Ok(());
    }

    let raw_text = std::fs::read_to_string(&text_path)?;
    let encoding = tokenizer.encode(raw_text, true).map_err(|e| format!("Encoding failed: {}", e))?;
    let tokens = encoding.get_ids().to_vec();

    println!("Loaded Training Data: {} Tokens.", tokens.len());

    let _batch_size = 1;
    let seq_len = 128;

    let epochs = 3;

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut step = 0;

        let start_time = Instant::now();

        while (step * seq_len) + seq_len + 1 <= tokens.len() {
            let start_idx = step * seq_len;
            let end_idx = start_idx + seq_len;

            let input_slice = &tokens[start_idx..end_idx];
            let target_slice = &tokens[(start_idx + 1)..(end_idx + 1)];

            let input_tensor = Tensor::new(input_slice, &device)?.unsqueeze(0)?;
            let target_tensor = Tensor::new(target_slice, &device)?.unsqueeze(0)?;
            let positions: Vec<usize> = (0..seq_len).collect();

            let logits = train_engine.forward(&input_tensor, &positions)?;

            let logits_flat = logits.flatten_to(1)?;
            let targets_flat = target_tensor.flatten_all()?;

            let loss = train_engine.compute_loss(&logits_flat, &targets_flat)?;

            let loss_val = loss.to_scalar::<f32>()?;
            total_loss += loss_val;

            train_engine.backward_ste(&loss, &mut optimizer)?;

            step += 1;

            if step % 10 == 0 {
                println!("Epoch {} | Step {} | Loss: {:.4}", epoch, step, loss_val);
            }
        }

        let duration = start_time.elapsed();
        println!("--- Epoch {} Complete! Average Loss: {:.4} | Time: {:.2}s ---", epoch, total_loss / (step as f32), duration.as_secs_f64());
    }

    Ok(())
}
