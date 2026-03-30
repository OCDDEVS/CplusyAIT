//! Token sampling strategies: greedy, top-k, top-p (nucleus).

use rand::Rng;

pub enum SamplingStrategy {
    Greedy,
    TopK { k: usize, temperature: f32 },
    TopP { p: f32, temperature: f32 },
}

pub fn sample(logits: &[f32], strategy: &SamplingStrategy) -> u32 {
    match strategy {
        SamplingStrategy::Greedy => {
            logits.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        }
        SamplingStrategy::TopK { k, temperature } => {
            let scaled = apply_temperature(logits, *temperature);
            let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(*k);
            sample_from_probs(&indexed)
        }
        SamplingStrategy::TopP { p, temperature } => {
            let scaled = apply_temperature(logits, *temperature);
            let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Softmax over sorted
            let max_val = indexed[0].1;
            let mut sum = 0.0f32;
            for item in &mut indexed {
                item.1 = (item.1 - max_val).exp();
                sum += item.1;
            }
            for item in &mut indexed {
                item.1 /= sum;
            }

            // Nucleus: keep tokens until cumulative prob >= p
            let mut cumulative = 0.0f32;
            let mut cutoff = indexed.len();
            for (i, item) in indexed.iter().enumerate() {
                cumulative += item.1;
                if cumulative >= *p {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed.truncate(cutoff);
            sample_from_probs(&indexed)
        }
    }
}

fn apply_temperature(logits: &[f32], temp: f32) -> Vec<f32> {
    let t = if temp < 1e-6 { 1e-6 } else { temp };
    logits.iter().map(|&v| v / t).collect()
}

fn sample_from_probs(items: &[(usize, f32)]) -> u32 {
    // Softmax
    let max_val = items.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = items.iter().map(|x| (x.1 - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return items[i].0 as u32;
        }
    }
    items.last().map(|x| x.0 as u32).unwrap_or(0)
}
