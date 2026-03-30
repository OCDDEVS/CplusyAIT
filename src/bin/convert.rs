use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use dialoguer::{theme::ColorfulTheme, Select};
use indicatif::{ProgressBar, ProgressStyle};
use safetensors::tensor::SafeTensors;
use serde::{Deserialize, Serialize};

/// Matches the inference::format::TensorMeta structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
    byte_offset: usize,
    byte_length: usize,
    gamma: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelManifest {
    model_type: String,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    vocab_size: usize,
    max_seq_len: usize,
    intermediate_dim: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    tensors: Vec<TensorMeta>,
    #[serde(default)]
    mamba_state_dim: Option<usize>,
}

/// Simple HF config.json fields we care about.
#[derive(Debug, Deserialize)]
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

fn get_safetensor_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |e| e == "safetensors") {
                files.push(path.clone());
            }
            // Also check subdirectories (HF models are often in subdirs)
            if path.is_dir() {
                if let Ok(sub_entries) = fs::read_dir(&path) {
                    for sub in sub_entries.flatten() {
                        let sp = sub.path();
                        if sp.is_file() && sp.extension().map_or(false, |e| e == "safetensors") {
                            files.push(sp);
                        }
                    }
                }
            }
        }
    }
    files.sort();
    files
}

/// Group safetensor files by their parent directory.
/// Returns a list of (display_name, directory_path, file_list).
fn group_model_dirs(files: &[PathBuf], models_dir: &Path) -> Vec<(String, PathBuf, Vec<PathBuf>)> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();

    for f in files {
        let parent = f.parent().unwrap_or(models_dir).to_path_buf();
        groups.entry(parent).or_default().push(f.clone());
    }

    groups.into_iter().map(|(dir, files)| {
        let name = if dir == models_dir {
            // Files directly in models/ — use the first file's stem
            files[0].file_stem().unwrap().to_string_lossy().to_string()
        } else {
            dir.file_name().unwrap().to_string_lossy().to_string()
        };
        let shard_info = if files.len() > 1 {
            format!("{} ({} shards)", name, files.len())
        } else {
            name.clone()
        };
        (shard_info, dir, files)
    }).collect()
}

fn quantize_to_packed_2bit(weights: &[f32]) -> (Vec<u8>, f32) {
    let sum_abs: f32 = weights.iter().map(|w| w.abs()).sum();
    let gamma = sum_abs / (weights.len() as f32 + 1e-8);

    let packed_len = (weights.len() + 3) / 4;
    let mut packed = vec![0u8; packed_len];

    for (i, &w) in weights.iter().enumerate() {
        let scaled = w / gamma;
        let q = scaled.clamp(-1.0, 1.0).round() as i8;
        let encoded: u8 = match q {
            1 => 1,
            -1 => 2,
            _ => 0,
        };
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        packed[byte_idx] |= encoded << bit_offset;
    }

    (packed, gamma)
}

/// Determines if a tensor should be quantized (2D weight matrices) or kept as f32
/// (1D biases, norms, embeddings).
fn should_quantize(name: &str, shape: &[usize]) -> bool {
    // Keep norm weights, biases, and embeddings in f32
    if name.contains("layernorm") || name.contains("norm") || name.contains("bias") {
        return false;
    }
    if name.contains("embed_tokens") {
        return false;
    }
    // Quantize 2D weight matrices
    shape.len() == 2
}

/// Convert raw tensor bytes to f32, handling bf16/f16 formats.
fn bytes_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    if data.len() == num_elements * 4 {
        let mut out = vec![0.0f32; num_elements];
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), out.as_mut_ptr() as *mut u8, data.len());
        }
        out
    } else if data.len() == num_elements * 2 {
        // bf16: shift left 16 bits to get f32
        let mut out = vec![0.0f32; num_elements];
        for i in 0..num_elements {
            let lo = data[i * 2] as u16;
            let hi = data[i * 2 + 1] as u16;
            let bits = lo | (hi << 8);
            let f32_bits = (bits as u32) << 16;
            out[i] = f32::from_bits(f32_bits);
        }
        out
    } else {
        panic!("Unexpected data size: {} bytes for {} elements", data.len(), num_elements);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Next-Gen CPU/GPU AI Framework ---");
    println!("--- Automated Post-Training Quantizer (PTQ) v2 ---");
    println!("--- Now with manifest.json for proper model loading ---\n");

    let models_dir = Path::new("models");
    if !models_dir.exists() {
        fs::create_dir(models_dir)?;
    }

    let files = get_safetensor_files(models_dir);
    if files.is_empty() {
        println!("No .safetensors models found in the 'models/' directory.");
        println!("Please download a Hugging Face model and place its .safetensors file(s) in 'models/'.");
        return Ok(());
    }

    let groups = group_model_dirs(&files, models_dir);
    let group_names: Vec<String> = groups.iter().map(|(name, _, _)| name.clone()).collect();

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select a model to crush into 1.58-bit Ternary format")
        .default(0)
        .items(&group_names)
        .interact()?;

    let (ref display_name, ref model_dir, ref shard_files) = groups[selection];
    println!("\nProcessing: {} ({} shard(s))", display_name, shard_files.len());

    // Try to load config.json from the model directory
    let hf_config: HFConfig = if let Ok(cfg_str) = fs::read_to_string(model_dir.join("config.json")) {
        serde_json::from_str(&cfg_str).unwrap_or_else(|_| default_hf_config())
    } else {
        println!("Warning: No config.json found. Using default Llama-3-8B dimensions.");
        default_hf_config()
    };

    // Output directory
    let stem = display_name.split(" (").next().unwrap_or(display_name);
    let output_dir = models_dir.join(format!("{}_1_58bit", stem));
    fs::create_dir_all(&output_dir)?;

    let mut blob: Vec<u8> = Vec::new();
    let mut tensor_metas: Vec<TensorMeta> = Vec::new();
    let mut total_original = 0usize;
    let mut total_packed = 0usize;
    let mut total_tensors = 0usize;

    // Count total tensors across all shards for the progress bar
    for shard_path in shard_files {
        let mut f = File::open(shard_path)?;
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;
        let tensors = SafeTensors::deserialize(&buffer)?;
        total_tensors += tensors.names().len();
    }

    let pb = ProgressBar::new(total_tensors as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) | {msg}")?
        .progress_chars("##-"));

    // Process each shard
    for (shard_idx, shard_path) in shard_files.iter().enumerate() {
        pb.set_message(format!("Shard {}/{}", shard_idx + 1, shard_files.len()));

        let mut f = File::open(shard_path)?;
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;

        let tensors = SafeTensors::deserialize(&buffer)?;
        let tensor_names: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();

        for name in &tensor_names {
            pb.set_message(format!("{}", name));

            let tensor = tensors.tensor(name)?;
            let shape: Vec<usize> = tensor.shape().to_vec();
            let data_bytes = tensor.data();
            total_original += data_bytes.len();

            let num_elements: usize = shape.iter().product();
            let float_data = bytes_to_f32(data_bytes, num_elements);

            let byte_offset = blob.len();

            if should_quantize(name, &shape) {
                let (packed, gamma) = quantize_to_packed_2bit(&float_data);
                let byte_length = packed.len();
                total_packed += byte_length;
                blob.extend_from_slice(&packed);

                tensor_metas.push(TensorMeta {
                    name: name.clone(),
                    shape,
                    byte_offset,
                    byte_length,
                    gamma,
                });
            } else {
                // Store as f32 bytes
                let f32_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(float_data.as_ptr() as *const u8, float_data.len() * 4)
                };
                let byte_length = f32_bytes.len();
                total_packed += byte_length;
                blob.extend_from_slice(f32_bytes);

                tensor_metas.push(TensorMeta {
                    name: name.clone(),
                    shape,
                    byte_offset,
                    byte_length,
                    gamma: 0.0,
                });
            }

            pb.inc(1);
        }
    }

    pb.finish_with_message("Done!");

    // Write blob
    let mut blob_file = File::create(output_dir.join("weights.bin"))?;
    blob_file.write_all(&blob)?;

    // Write manifest
    // Write manifest — infer missing config values from actual tensor shapes
    let hidden_dim = hf_config.hidden_size.unwrap_or_else(|| {
        tensor_metas.iter()
            .find(|m| m.name.contains("embed_tokens"))
            .map(|m| *m.shape.last().unwrap_or(&4096))
            .unwrap_or(4096)
    });
    let vocab_size = hf_config.vocab_size.unwrap_or_else(|| {
        tensor_metas.iter()
            .find(|m| m.name.contains("embed_tokens"))
            .map(|m| m.shape[0])
            .unwrap_or(32000)
    });
    let num_layers = hf_config.num_hidden_layers.unwrap_or_else(|| {
        tensor_metas.iter()
            .filter_map(|m| {
                m.name.strip_prefix("model.layers.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map(|m| m + 1)
            .unwrap_or(32)
    });
    let intermediate_dim = hf_config.intermediate_size.unwrap_or_else(|| {
        tensor_metas.iter()
            .find(|m| m.name.contains("gate_proj"))
            .map(|m| m.shape[0])
            .unwrap_or(14336)
    });
    let num_heads = hf_config.num_attention_heads.unwrap_or_else(|| {
        if hidden_dim > 0 { (hidden_dim / 64).max(1) } else { 32 }
    });

    let manifest = ModelManifest {
        model_type: hf_config.model_type.unwrap_or_else(|| "llama".to_string()),
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads: hf_config.num_key_value_heads.unwrap_or(num_heads),
        vocab_size,
        max_seq_len: hf_config.max_position_embeddings.unwrap_or(8192),
        intermediate_dim,
        rope_theta: hf_config.rope_theta.unwrap_or(500000.0),
        rms_norm_eps: hf_config.rms_norm_eps.unwrap_or(1e-5),
        tensors: tensor_metas,
        mamba_state_dim: None,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(output_dir.join("manifest.json"), &manifest_json)?;

    // Copy tokenizer.json if it exists in the source model directory
    let tokenizer_src = model_dir.join("tokenizer.json");
    if tokenizer_src.exists() {
        fs::copy(&tokenizer_src, output_dir.join("tokenizer.json"))?;
        println!("Copied tokenizer.json to output directory.");
    } else {
        println!("Note: No tokenizer.json found in source. You'll need to copy one manually.");
    }

    // Also copy tokenizer_config.json if present
    let tok_config_src = model_dir.join("tokenizer_config.json");
    if tok_config_src.exists() {
        fs::copy(&tok_config_src, output_dir.join("tokenizer_config.json"))?;
    }

    println!("\n--- Conversion Complete! ---");
    println!("Original Size:  {:.2} MB", total_original as f64 / 1_000_000.0);
    println!("Packed Size:    {:.2} MB", total_packed as f64 / 1_000_000.0);
    println!("Ratio:          {:.1}x smaller", total_original as f64 / total_packed as f64);
    println!("Output:         {}/", output_dir.display());
    println!("  manifest.json - model config + tensor metadata");
    println!("  weights.bin   - packed weight blob");

    Ok(())
}

fn default_hf_config() -> HFConfig {
    HFConfig {
        hidden_size: Some(4096),
        num_hidden_layers: Some(32),
        num_attention_heads: Some(32),
        num_key_value_heads: Some(8),
        vocab_size: Some(128256),
        max_position_embeddings: Some(8192),
        intermediate_size: Some(14336),
        rope_theta: Some(500000.0),
        rms_norm_eps: Some(1e-5),
        model_type: Some("llama".to_string()),
    }
}
