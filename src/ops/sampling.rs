// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Token sampling strategies

/// Greedy sampling: select the token with highest probability
pub fn sample_greedy(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Top-K sampling: sample from top K tokens
/// Returns (token_id, probability)
pub fn sample_top_k(
    logits: &[f32],
    k: usize,
    temperature: f32,
) -> (usize, f32) {
    let vocab_size = logits.len();

    // Apply temperature
    let scaled: Vec<f32> = if temperature > 0.0 {
        logits.iter().map(|&x| x / temperature).collect()
    } else {
        logits.to_vec()
    };

    // Get top-k indices and values
    let mut indices = vec![0usize; vocab_size];
    for i in 0..vocab_size {
        indices[i] = i;
    }

    // Partial sort to get top-k
    indices.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
    let top_k_indices: Vec<usize> = indices.iter().take(k).copied().collect();

    // Softmax over top-k
    let top_k_values: Vec<f32> =
        top_k_indices.iter().map(|&i| scaled[i]).collect();
    let max_val =
        top_k_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut exp_sum = 0.0f32;
    let probs: Vec<f32> = top_k_values
        .iter()
        .map(|&v| {
            let exp = (v - max_val).exp();
            exp_sum += exp;
            exp
        })
        .collect();

    // Normalize
    let probs: Vec<f32> = probs.iter().map(|p| p / exp_sum).collect();

    // Simple deterministic: choose highest from top-k
    // TODO: add random sampling
    let best_idx = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    (top_k_indices[best_idx], probs[best_idx])
}

/// Sampling strategy
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    Greedy,
    TopK { k: usize, temperature: f32 },
}

/// Sample a token from logits
pub fn sample(logits: &[f32], strategy: &SamplingStrategy) -> usize {
    match strategy {
        SamplingStrategy::Greedy => sample_greedy(logits),
        SamplingStrategy::TopK { k, temperature } => {
            sample_top_k(logits, *k, *temperature).0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = sample_greedy(&logits);
        assert_eq!(token, 3); // Highest at index 3
    }

    #[test]
    fn test_top_k() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let (token, prob) = sample_top_k(&logits, 3, 1.0);
        assert!(token < 5);
        assert!(prob > 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_sampling_strategies() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];

        let greedy = sample(&logits, &SamplingStrategy::Greedy);
        assert_eq!(greedy, 3);

        let topk =
            sample(&logits, &SamplingStrategy::TopK { k: 2, temperature: 1.0 });
        assert!(topk < 5);
    }
}
