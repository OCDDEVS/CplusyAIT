//! Model format for serialized 1.58-bit ternary models.
//! Stores a JSON manifest alongside raw packed weight blobs so we can
//! reconstruct the model layer-by-layer at load time.

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

/// Metadata for a single packed tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub name: String,
    /// Original shape before packing (e.g. [4096, 4096]).
    pub shape: Vec<usize>,
    /// Byte offset into the .bin blob file.
    pub byte_offset: usize,
    /// Byte length in the .bin blob file.
    pub byte_length: usize,
    /// The absolute-mean scaling factor (gamma) used during quantization.
    pub gamma: f32,
}

/// Top-level model manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_type: String,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub intermediate_dim: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tensors: Vec<TensorMeta>,
}

/// A loaded model: manifest + raw packed bytes in memory.
pub struct PackedModel {
    pub manifest: ModelManifest,
    pub blob: Vec<u8>,
}

impl PackedModel {
    /// Load from a directory containing `manifest.json` and `weights.bin`.
    pub fn load<P: AsRef<Path>>(dir: P) -> std::io::Result<Self> {
        let dir = dir.as_ref();
        let manifest_path = dir.join("manifest.json");
        let blob_path = dir.join("weights.bin");

        let manifest_str = fs::read_to_string(&manifest_path)?;
        let manifest: ModelManifest = serde_json::from_str(&manifest_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut blob = Vec::new();
        File::open(&blob_path)?.read_to_end(&mut blob)?;

        Ok(Self { manifest, blob })
    }

    /// Get the raw packed bytes for a tensor by name.
    pub fn get_tensor_data(&self, name: &str) -> Option<(&TensorMeta, &[u8])> {
        self.manifest.tensors.iter().find(|t| t.name == name).map(|meta| {
            let data = &self.blob[meta.byte_offset..meta.byte_offset + meta.byte_length];
            (meta, data)
        })
    }

    /// Save manifest + blob to a directory.
    pub fn save<P: AsRef<Path>>(&self, dir: P) -> std::io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;

        let manifest_json = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(dir.join("manifest.json"), manifest_json)?;

        let mut f = File::create(dir.join("weights.bin"))?;
        f.write_all(&self.blob)?;
        Ok(())
    }
}

/// Quantize an f32 weight slice to 2-bit packed bytes using BitNet absolute-mean scaling.
/// Returns (packed_bytes, gamma).
pub fn quantize_and_pack(weights: &[f32]) -> (Vec<u8>, f32) {
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
