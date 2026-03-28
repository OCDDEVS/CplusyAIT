use candle_core::{Device, Result, Tensor};
use safetensors::tensor::SafeTensors;
use std::fs::File;
use std::io::Read;

/// Loads weights from a HuggingFace SafeTensors checkpoint and quantizes them directly
/// to our custom 1.58-bit packed format (uint8_t arrays for AVX2 execution).
pub struct CheckpointLoader;

impl CheckpointLoader {
    /// Loads a `.safetensors` file from disk.
    pub fn load_safetensors(path: &str) -> std::io::Result<Vec<u8>> {
        let mut f = File::open(path)?;
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    /// Reads a specific weight matrix (e.g., `model.layers.0.mlp.up_proj.weight`)
    /// from the loaded `.safetensors` file, and packs it into the 2-bit AVX2 array.
    pub fn load_and_pack_weight(buffer: &[u8], tensor_name: &str) -> Result<(Vec<u8>, usize, usize)> {
        let tensors = SafeTensors::deserialize(buffer)
            .map_err(|e| candle_core::Error::Msg(format!("SafeTensors Error: {:?}", e)))?;

        let tensor = tensors.tensor(tensor_name)
            .map_err(|e| candle_core::Error::Msg(format!("Tensor {} not found: {:?}", tensor_name, e)))?;

        // We assume 2D weights for MLPs. (m, k)
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(candle_core::Error::Msg(format!("Expected 2D tensor, got {:?}", shape)));
        }
        let (rows, cols) = (shape[0], shape[1]);

        // Extract raw f32 data (safetensors default typically returns bytes, we cast)
        let data_bytes = tensor.data();
        let float_data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                data_bytes.as_ptr() as *const f32,
                data_bytes.len() / 4
            )
        };

        // Quantize and Pack (1.58-bit BitNet scaling)
        let packed = Self::quantize_to_packed_2bit(float_data, rows, cols);

        Ok((packed, rows, cols))
    }

    /// Executes the BitNet Absolute Mean Quantization and 2-bit Packing (4 weights per byte)
    fn quantize_to_packed_2bit(weights: &[f32], m: usize, k: usize) -> Vec<u8> {
        let mut sum_abs = 0.0;
        for &w in weights {
            sum_abs += w.abs();
        }
        let gamma = sum_abs / (weights.len() as f32 + 1e-8);

        // Ensure K is aligned to packing logic. Output is M x (K / 4) bytes.
        let packed_len = (m * k) / 4;
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
}
