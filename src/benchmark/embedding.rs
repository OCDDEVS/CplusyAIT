use rand::Rng;

/// A Real Token Embedding Layer managing high-precision parameters.
pub struct EmbeddingLayer {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub weights: Vec<f32>,
}

impl EmbeddingLayer {
    pub fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        // Initialize with small uniform distribution
        let weights: Vec<f32> = (0..(vocab_size * hidden_dim))
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        Self { vocab_size, hidden_dim, weights }
    }

    /// Forward pass: Retrieve the embedding vector for a given token ID.
    pub fn forward(&self, token_id: usize, output: &mut [f32]) {
        let start = token_id * self.hidden_dim;
        let end = start + self.hidden_dim;
        output.copy_from_slice(&self.weights[start..end]);
    }

    /// Backward pass: Apply gradients directly to the token's embedding vector.
    pub fn backward_update(&mut self, token_id: usize, gradients: &[f32], learning_rate: f32) {
        let start = token_id * self.hidden_dim;
        for i in 0..self.hidden_dim {
            self.weights[start + i] -= learning_rate * gradients[i];
        }
    }
}
