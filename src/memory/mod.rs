use serde::{Deserialize, Serialize};

/// Implementation of the EverMemOS "Engram" Lifecycle
/// Phase I: Episodic Trace Formation - From raw chat logs to structured MemCells.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemCell {
    /// E (Episode): A concise third-person narrative of what happened
    pub episode: String,

    /// F (Atomic Facts): Discrete, verifiable statements
    pub atomic_facts: Vec<String>,

    /// P (Foresight): Forward-looking inferences with validity intervals
    pub foresight: Vec<Foresight>,

    /// M (Metadata): Timestamps and source pointers
    pub metadata: Metadata,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Foresight {
    pub inference: String,
    pub valid_from_timestamp: u64,
    pub valid_until_timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Metadata {
    pub timestamp_created: u64,
    pub source_id: String,
}

/// Phase II: Semantic Consolidation - Clustering MemCells into MemScenes
/// This allows the model to compress user profiles over time on the local machine
/// rather than keeping infinite chat history.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemScene {
    pub theme: String,
    pub clustered_cells: Vec<MemCell>,
    pub centroid_vector: Vec<f32>, // Used by the MSA Router (Routing Keys)
}

impl MemScene {
    /// Compares a new MemCell vector against the scene's centroid using cosine similarity.
    /// If similarity > tau, assimilate. Else, spawn a new scene.
    pub fn assimilate_or_spawn(&mut self, new_cell: MemCell, cell_vector: &[f32], tau: f32) -> bool {
        let similarity = cosine_similarity(&self.centroid_vector, cell_vector);

        if similarity > tau {
            self.clustered_cells.push(new_cell);
            // Recompute centroid as average of all cells
            let n = self.clustered_cells.len() as f32;
            for i in 0..self.centroid_vector.len() {
                self.centroid_vector[i] = ((self.centroid_vector[i] * (n - 1.0)) + cell_vector[i]) / n;
            }
            true
        } else {
            false
        }
    }
}

pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() || v1.is_empty() { return 0.0; }
    let mut dot = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    for i in 0..v1.len() {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    if norm1 == 0.0 || norm2 == 0.0 { return 0.0; }
    dot / (norm1.sqrt() * norm2.sqrt())
}

/// EverMemOS Memory Manager: Orchestrates Working Memory vs Long-Term Memory.
/// Maintains the contiguous array of Router Keys (Centroids) for fast C++ MSA Retrieval.
pub struct MemoryManager {
    pub scenes: Vec<MemScene>,
    pub vector_dim: usize,

    // Contiguous memory buffer to hold all centroid vectors for fast C++ access
    pub routing_keys_vram: Vec<f32>,
}

impl MemoryManager {
    pub fn new(vector_dim: usize) -> Self {
        Self {
            scenes: Vec::new(),
            vector_dim,
            routing_keys_vram: Vec::new(),
        }
    }

    /// Phase I & II: Process a new interaction, form a MemCell, and Semantically Consolidate it.
    pub fn ingest_episode(&mut self, episode_text: String, embedding: Vec<f32>, timestamp: u64) {
        let cell = MemCell {
            episode: episode_text.clone(),
            atomic_facts: vec![episode_text], // Simplified extraction
            foresight: vec![],
            metadata: Metadata {
                timestamp_created: timestamp,
                source_id: format!("doc_{}", timestamp),
            },
        };

        let tau = 0.85; // Similarity threshold for assimilation
        let mut assimilated = false;

        for scene in self.scenes.iter_mut() {
            if scene.assimilate_or_spawn(cell.clone(), &embedding, tau) {
                assimilated = true;
                break;
            }
        }

        if !assimilated {
            // Spawn new scene
            let new_scene = MemScene {
                theme: format!("Theme_{}", timestamp),
                clustered_cells: vec![cell],
                centroid_vector: embedding.clone(),
            };
            self.scenes.push(new_scene);
        }

        // Rebuild contiguous routing keys for the fast C++ MSA Router
        self.rebuild_routing_keys();
    }

    fn rebuild_routing_keys(&mut self) {
        self.routing_keys_vram.clear();
        self.routing_keys_vram.reserve(self.scenes.len() * self.vector_dim);
        for scene in &self.scenes {
            self.routing_keys_vram.extend_from_slice(&scene.centroid_vector);
        }
    }
}
