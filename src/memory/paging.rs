use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use memmap2::{Mmap, MmapMut};
use std::sync::Arc;

/// Simulates a 100M Token Memory Disk utilizing `mmap`.
/// This offloads Long-Term KV Cache from RAM directly to NVMe SSD.
pub struct DiskMemoryPager {
    pub file: File,
    pub mmap: Arc<Mmap>,
    pub max_blocks: usize,
    pub block_size: usize, // e.g., sizeof(f32) * hidden_dim
}

impl DiskMemoryPager {
    pub fn new<P: AsRef<Path>>(path: P, max_blocks: usize, block_size: usize) -> std::io::Result<Self> {
        let total_size = max_blocks * block_size;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        // Pre-allocate the file on disk
        file.set_len(total_size as u64)?;

        // Memory map the file (Read-only mapped into memory space)
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            file,
            mmap: Arc::new(mmap),
            max_blocks,
            block_size,
        })
    }

    /// Fast memory map fetch based on the MSA Router indices.
    /// This bypasses traditional slow I/O `read()` syscalls.
    pub fn fetch_blocks(&self, block_indices: &[i32], out_buffer: &mut [f32], vector_dim: usize) {
        for (i, &idx) in block_indices.iter().enumerate() {
            if idx >= 0 && (idx as usize) < self.max_blocks {
                let start = (idx as usize) * self.block_size;
                let end = start + self.block_size;

                // Get raw bytes from mmap
                let bytes = &self.mmap[start..end];

                // Unsafe cast to f32 (assuming correct alignment and little-endian)
                let floats: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const f32,
                        vector_dim
                    )
                };

                // Copy into Working Memory buffer
                let out_start = i * vector_dim;
                out_buffer[out_start..out_start + vector_dim].copy_from_slice(floats);
            } else {
                // Sentinels or invalid bounds return zeroed padding
                let out_start = i * vector_dim;
                for j in 0..vector_dim {
                    out_buffer[out_start + j] = 0.0;
                }
            }
        }
    }
}
