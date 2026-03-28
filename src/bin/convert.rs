use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use dialoguer::{theme::ColorfulTheme, Select};
use indicatif::{ProgressBar, ProgressStyle};
use safetensors::tensor::SafeTensors;

fn get_safetensor_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |e| e == "safetensors") {
                files.push(path);
            }
        }
    }
    files
}

/// Executes the BitNet Absolute Mean Quantization and 2-bit Packing (4 weights per byte)
fn quantize_to_packed_2bit(weights: &[f32]) -> Vec<u8> {
    let mut sum_abs = 0.0;
    for &w in weights {
        sum_abs += w.abs();
    }
    let gamma = sum_abs / (weights.len() as f32 + 1e-8);

    // Ensure length is aligned to packing logic. Output is N / 4 bytes.
    let packed_len = (weights.len() + 3) / 4;
    let mut packed_weights = vec![0u8; packed_len];

    for i in 0..weights.len() {
        let scaled = weights[i] / gamma;
        let clamped = scaled.clamp(-1.0, 1.0).round() as i8;

        let encoded: u8 = match clamped {
            1 => 1,
            -1 => 2,
            _ => 0,
        };

        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;

        packed_weights[byte_idx] |= encoded << bit_offset;
    }

    packed_weights
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Next-Gen CPU/GPU AI Framework ---");
    println!("--- Automated Post-Training Quantizer (PTQ) ---\n");

    let models_dir = Path::new("models");
    if !models_dir.exists() {
        fs::create_dir(models_dir)?;
    }

    let files = get_safetensor_files(models_dir);
    if files.is_empty() {
        println!("No .safetensors models found in the 'models/' directory.");
        println!("Please download a Hugging Face model and place its .safetensors file in the 'models/' folder.");
        return Ok(());
    }

    let file_names: Vec<String> = files.iter().map(|f| f.file_name().unwrap().to_string_lossy().to_string()).collect();

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select a model to crush into 1.58-bit Ternary format")
        .default(0)
        .items(&file_names)
        .interact()?;

    let selected_file = &files[selection];
    println!("\nLoading {}...", selected_file.display());

    // 1. Read File
    let mut f = File::open(selected_file)?;
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer)?;

    // 2. Deserialize SafeTensors
    let tensors = SafeTensors::deserialize(&buffer)?;
    let tensor_names: Vec<_> = tensors.names().into_iter().collect();

    println!("Found {} tensors in the model.", tensor_names.len());

    let output_filename = format!("{}_1_58bit.bin", selected_file.file_stem().unwrap().to_string_lossy());
    let output_path = models_dir.join(&output_filename);
    let mut output_file = File::create(&output_path)?;

    // 3. Process Tensors with Real-time Progress Bar
    let pb = ProgressBar::new(tensor_names.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} tensors ({eta}) | {msg}")?
        .progress_chars("##-"));

    let mut total_original_bytes = 0;
    let mut total_crushed_bytes = 0;

    for name in tensor_names {
        pb.set_message(format!("Quantizing {}", name));

        let tensor = tensors.tensor(name)?;
        let shape = tensor.shape();

        // Extract raw f32 data
        let data_bytes = tensor.data();
        let float_data: &[f32] = unsafe {
            std::slice::from_raw_parts(data_bytes.as_ptr() as *const f32, data_bytes.len() / 4)
        };

        total_original_bytes += data_bytes.len();

        // If it's a weight matrix (2D), we crush it into 2-bit packed array
        // If it's a 1D bias or embedding, we might keep it FP32 in a real engine,
        // but for this full-quantization demo, we crush all parameters uniformly.
        let packed = quantize_to_packed_2bit(float_data);
        total_crushed_bytes += packed.len();

        // Save to disk
        // A robust format would serialize shapes and metadata headers.
        // We write the raw binary for this optimized edge engine.
        output_file.write_all(&packed)?;

        pb.inc(1);
    }

    pb.finish_with_message("Done!");

    println!("\n--- Conversion Complete! ---");
    println!("Original Size: {:.2} MB", total_original_bytes as f64 / 1_000_000.0);
    println!("Crushed Size:  {:.2} MB", total_crushed_bytes as f64 / 1_000_000.0);
    println!("Ratio:         {:.2}x Smaller!", (total_original_bytes as f64) / (total_crushed_bytes as f64));
    println!("Saved as:      models/{}", output_filename);

    Ok(())
}
