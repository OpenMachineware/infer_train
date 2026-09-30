//! Sampling operators: ARGMAX, ARGSORT, TOP_K

/// Find index of maximum value in array
pub fn argmax_f32(data: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = data[0];
    for (i, &v) in data.iter().enumerate().skip(1) {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

/// Find indices of top-k largest values
pub fn top_k_f32(data: &[f32], k: usize) -> Vec<(usize, f32)> {
    // Use partial sort for efficiency
    let mut indexed: Vec<(usize, f32)> =
        data.iter().enumerate().map(|(i, &v)| (i, v)).collect();

    // Partial sort - only need top k
    indexed.select_nth_unstable_by(k, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    indexed[..k].to_vec()
}

/// Sort indices by values in descending order
pub fn argsort_f32(data: &[f32]) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> =
        data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let data = [1.0, 5.0, 3.0, 2.0, 4.0];
        assert_eq!(argmax_f32(&data), 1);
    }

    #[test]
    fn test_top_k() {
        let data = [1.0, 5.0, 3.0, 2.0, 4.0];
        let top = top_k_f32(&data, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 1); // 5.0 is largest
    }

    #[test]
    fn test_argsort() {
        let data = [1.0, 5.0, 3.0, 2.0, 4.0];
        let sorted = argsort_f32(&data);
        assert_eq!(sorted[0], 1); // 5.0
        assert_eq!(sorted[1], 4); // 4.0
    }
}
