//! Disk-backed memory pager for KV cache overflow.
//!
//! When the KV cache exceeds the RAM budget, evicted positions are written
//! to an mmap'd file on NVMe SSD. The MSA router can later page them back
//! in when they're selected as relevant context.
//!
//! From the research doc: "Long-Term Memory (KV Cache): Full Key-Value tensors
//! stored in system RAM, intelligently paged to fast NVMe storage when exceeding
//! 8GB."

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use memmap2::MmapMut;

/// A single paged-out KV block on disk.
/// Tracks where in the mmap file this block lives.
#[derive(Debug, Clone)]
struct PagedBlock {
    /// Original sequence position this block came from.
    original_pos: usize,
    /// Layer index this block belongs to.
    layer_idx: usize,
    /// Offset (in bytes) into the mmap file where keys start.
    key_offset: usize,
    /// Offset (in bytes) into the mmap file where values start.
    value_offset: usize,
}

/// Disk-backed pager for KV cache overflow.
///
/// Architecture:
/// - Pre-allocates a file on disk sized for `max_blocks` KV entries
/// - Uses mmap for zero-copy reads (OS handles page faults from SSD)
/// - Writes evicted KV blocks sequentially, tracks them in a block table
/// - Supports selective page-in: given block indices, reads K/V back into RAM
pub struct DiskMemoryPager {
    /// Mutable memory-mapped file for read/write access.
    mmap: MmapMut,
    /// Path to the backing file (for diagnostics/cleanup).
    path: PathBuf,
    /// Size of a single K or V vector in bytes: num_kv_heads * head_dim * sizeof(f32).
    kv_vector_bytes: usize,
    /// Size of a full KV block (keys + values) in bytes: 2 * kv_vector_bytes.
    block_bytes: usize,
    /// Maximum number of blocks the file can hold.
    max_blocks: usize,
    /// Table of paged-out blocks (append-only, indexed by slot).
    block_table: Vec<PagedBlock>,
    /// Next free slot in the file.
    next_slot: usize,
    /// Number of KV heads (for reconstructing vectors).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl DiskMemoryPager {
    /// Create a new pager backed by a file at `path`.
    /// Pre-allocates space for `max_blocks` KV entries.
    pub fn new<P: AsRef<Path>>(
        path: P,
        max_blocks: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> io::Result<Self> {
        let kv_vector_bytes = num_kv_heads * head_dim * std::mem::size_of::<f32>();
        let block_bytes = kv_vector_bytes * 2; // K + V
        let total_bytes = max_blocks * block_bytes;

        if total_bytes == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "max_blocks or dimensions are zero"));
        }

        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        file.set_len(total_bytes as u64)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(Self {
            mmap,
            path,
            kv_vector_bytes,
            block_bytes,
            max_blocks,
            block_table: Vec::new(),
            next_slot: 0,
            num_kv_heads,
            head_dim,
        })
    }

    /// Page out a batch of evicted KV positions to disk.
    /// `keys` and `values` are flat f32 slices: n_positions * num_kv_heads * head_dim.
    /// `start_pos` is the original sequence position of the first evicted entry.
    /// `layer_idx` identifies which transformer layer these belong to.
    /// Returns the number of blocks successfully written.
    pub fn page_out(
        &mut self,
        keys: &[f32],
        values: &[f32],
        start_pos: usize,
        layer_idx: usize,
    ) -> usize {
        let floats_per_vec = self.num_kv_heads * self.head_dim;
        let n_positions = keys.len() / floats_per_vec;

        if n_positions == 0 || keys.len() != values.len() {
            return 0;
        }

        let mut written = 0;
        for i in 0..n_positions {
            if self.next_slot >= self.max_blocks {
                // Disk full — wrap around (circular buffer)
                self.next_slot = 0;
            }

            let slot = self.next_slot;
            let file_offset = slot * self.block_bytes;

            // Write keys
            let k_start = i * floats_per_vec;
            let k_bytes = as_bytes(&keys[k_start..k_start + floats_per_vec]);
            self.mmap[file_offset..file_offset + self.kv_vector_bytes]
                .copy_from_slice(k_bytes);

            // Write values
            let v_start = i * floats_per_vec;
            let v_bytes = as_bytes(&values[v_start..v_start + floats_per_vec]);
            let v_offset = file_offset + self.kv_vector_bytes;
            self.mmap[v_offset..v_offset + self.kv_vector_bytes]
                .copy_from_slice(v_bytes);

            // Record in block table
            let block = PagedBlock {
                original_pos: start_pos + i,
                layer_idx,
                key_offset: file_offset,
                value_offset: v_offset,
            };

            if slot < self.block_table.len() {
                self.block_table[slot] = block;
            } else {
                self.block_table.push(block);
            }

            self.next_slot += 1;
            written += 1;
        }

        // Flush to ensure data hits the SSD
        let _ = self.mmap.flush_async();

        written
    }

    /// Page in specific blocks by their slot indices.
    /// Returns (keys, values) as flat f32 vectors.
    pub fn page_in(&self, slot_indices: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let floats_per_vec = self.num_kv_heads * self.head_dim;
        let mut keys = Vec::with_capacity(slot_indices.len() * floats_per_vec);
        let mut values = Vec::with_capacity(slot_indices.len() * floats_per_vec);

        for &slot in slot_indices {
            if slot < self.block_table.len() {
                let block = &self.block_table[slot];

                // Read keys from mmap
                let k_bytes = &self.mmap[block.key_offset..block.key_offset + self.kv_vector_bytes];
                let k_floats = from_bytes(k_bytes);
                keys.extend_from_slice(k_floats);

                // Read values from mmap
                let v_bytes = &self.mmap[block.value_offset..block.value_offset + self.kv_vector_bytes];
                let v_floats = from_bytes(v_bytes);
                values.extend_from_slice(v_floats);
            } else {
                // Invalid slot — zero-fill
                keys.extend(std::iter::repeat(0.0f32).take(floats_per_vec));
                values.extend(std::iter::repeat(0.0f32).take(floats_per_vec));
            }
        }

        (keys, values)
    }

    /// Search paged-out blocks for a specific layer and return their slot indices.
    /// Used by the KV cache to find which disk blocks belong to a given layer.
    pub fn find_layer_slots(&self, layer_idx: usize) -> Vec<usize> {
        self.block_table.iter().enumerate()
            .filter(|(_, b)| b.layer_idx == layer_idx)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get the keys from all paged-out blocks for a given layer as a flat f32 slice.
    /// Used by the MSA router to include paged-out positions in routing decisions.
    pub fn get_layer_keys(&self, layer_idx: usize) -> Vec<f32> {
        let floats_per_vec = self.num_kv_heads * self.head_dim;
        let slots = self.find_layer_slots(layer_idx);
        let mut keys = Vec::with_capacity(slots.len() * floats_per_vec);

        for slot in &slots {
            let block = &self.block_table[*slot];
            let k_bytes = &self.mmap[block.key_offset..block.key_offset + self.kv_vector_bytes];
            keys.extend_from_slice(from_bytes(k_bytes));
        }

        keys
    }

    /// Total number of blocks currently paged out.
    pub fn paged_block_count(&self) -> usize {
        self.block_table.len()
    }

    /// Disk usage in bytes.
    pub fn disk_usage_bytes(&self) -> usize {
        self.block_table.len() * self.block_bytes
    }

    /// Clear all paged data (reset the pager).
    pub fn clear(&mut self) {
        self.block_table.clear();
        self.next_slot = 0;
    }

    /// Path to the backing file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for DiskMemoryPager {
    fn drop(&mut self) {
        // Best-effort cleanup: flush and optionally remove the file
        let _ = self.mmap.flush();
    }
}

/// Reinterpret a f32 slice as raw bytes.
fn as_bytes(floats: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            floats.as_ptr() as *const u8,
            floats.len() * std::mem::size_of::<f32>(),
        )
    }
}

/// Reinterpret raw bytes as a f32 slice.
fn from_bytes(bytes: &[u8]) -> &[f32] {
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const f32,
            bytes.len() / std::mem::size_of::<f32>(),
        )
    }
}
