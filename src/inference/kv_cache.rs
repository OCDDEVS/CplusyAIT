//! KV Cache for autoregressive transformer inference.
//! Supports memory budget enforcement via sliding window eviction
//! and MSA (Memory Sparse Attention) routing for selective retrieval.

use crate::ffi;

/// Per-layer KV cache storing keys and values as flat f32 vectors.
/// Keys shape: (seq_len, num_kv_heads, head_dim)
/// Values shape: (seq_len, num_kv_heads, head_dim)
pub struct LayerKVCache {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl LayerKVCache {
    pub fn new(num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            seq_len: 0,
            num_kv_heads,
            head_dim,
        }
    }

    /// Append a single position's K and V vectors.
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.num_kv_heads * self.head_dim);
        debug_assert_eq!(v.len(), self.num_kv_heads * self.head_dim);
        self.keys.extend_from_slice(k);
        self.values.extend_from_slice(v);
        self.seq_len += 1;
    }

    /// Evict the oldest `n` positions from the cache (sliding window).
    pub fn evict_oldest(&mut self, n: usize) {
        if n >= self.seq_len {
            self.clear();
            return;
        }
        let stride = self.num_kv_heads * self.head_dim;
        let drop_floats = n * stride;
        self.keys.drain(..drop_floats);
        self.values.drain(..drop_floats);
        self.seq_len -= n;
    }

    /// Memory usage in bytes for this layer's KV cache.
    pub fn memory_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * std::mem::size_of::<f32>()
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_len = 0;
    }

    /// MSA Routing: given a query vector, find the top-k most relevant cached
    /// positions using the C++ cosine similarity router.
    /// Returns (selected_keys, selected_values, indices) where keys/values
    /// contain only the top-k positions' data.
    pub fn msa_select_top_k(&self, query: &[f32], k: usize) -> (Vec<f32>, Vec<f32>, Vec<i32>) {
        let num_positions = self.seq_len;
        let kv_dim = self.num_kv_heads * self.head_dim;

        if num_positions == 0 || k == 0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let actual_k = k.min(num_positions);

        // Use the C++ MSA router to find top-k positions by cosine similarity
        // The router compares `query` against each position's key vector
        let mut top_k_indices = vec![0i32; actual_k];

        // We need to match query dim to key dim. If query is hidden_dim and keys
        // are kv_dim (num_kv_heads * head_dim), we use the first kv_dim elements.
        let query_dim = query.len().min(kv_dim);
        let routing_query = &query[..query_dim];

        unsafe {
            ffi::msa_route_top_k(
                routing_query.as_ptr(),
                self.keys.as_ptr(),
                top_k_indices.as_mut_ptr(),
                num_positions,
                kv_dim,
                actual_k,
            );
        }

        // Gather the selected K and V vectors
        let mut selected_keys = Vec::with_capacity(actual_k * kv_dim);
        let mut selected_values = Vec::with_capacity(actual_k * kv_dim);

        for &idx in &top_k_indices {
            if idx >= 0 && (idx as usize) < num_positions {
                let offset = idx as usize * kv_dim;
                selected_keys.extend_from_slice(&self.keys[offset..offset + kv_dim]);
                selected_values.extend_from_slice(&self.values[offset..offset + kv_dim]);
            }
        }

        (selected_keys, selected_values, top_k_indices)
    }
}

/// Full KV cache across all layers with memory budget enforcement.
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
    /// Maximum memory budget in bytes for the entire KV cache.
    /// 0 means unlimited.
    pub max_memory_bytes: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKVCache::new(num_kv_heads, head_dim))
            .collect();
        Self { layers, max_memory_bytes: 0 }
    }

    /// Create a KV cache with a memory budget (in bytes).
    pub fn with_budget(num_layers: usize, num_kv_heads: usize, head_dim: usize, budget_bytes: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKVCache::new(num_kv_heads, head_dim))
            .collect();
        Self { layers, max_memory_bytes: budget_bytes }
    }

    pub fn seq_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.seq_len)
    }

    /// Total memory usage across all layers.
    pub fn total_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Enforce the memory budget by evicting oldest positions if needed.
    /// Returns the number of positions evicted.
    pub fn enforce_budget(&mut self) -> usize {
        if self.max_memory_bytes == 0 {
            return 0;
        }

        let current = self.total_memory_bytes();
        if current <= self.max_memory_bytes {
            return 0;
        }

        // Calculate how many positions to evict
        // Each position across all layers uses:
        // num_layers * 2 (K+V) * num_kv_heads * head_dim * sizeof(f32)
        let first = &self.layers[0];
        let bytes_per_position = self.layers.len() * 2 * first.num_kv_heads * first.head_dim * 4;
        if bytes_per_position == 0 { return 0; }

        let excess = current - self.max_memory_bytes;
        let positions_to_evict = (excess + bytes_per_position - 1) / bytes_per_position;

        // Evict from all layers uniformly
        for layer in &mut self.layers {
            layer.evict_oldest(positions_to_evict);
        }

        positions_to_evict
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}
