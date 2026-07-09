// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
// use crate::ops::registry::{OpAttrs};

// ============================================================
// Float Generic Forward
// ============================================================

pub fn embedding<T: DType + Send + Sync>(
    indices: &Tensor<i64>,
    weight: &Tensor<T>,
) -> Tensor<T> {
    let weight_shape = weight.shape();
    assert!(weight_shape.len() >= 2, "embedding weight must be at least 2D");

    let num_embeddings = weight_shape[0];
    let embedding_dim = weight_shape[1];
    let indices_data = indices.data();

    let batch_dims = indices.shape();
    let out_size: usize = batch_dims.iter().product::<usize>() * embedding_dim;
    let mut out_data = vec![T::from_f32(0.0); out_size];

    let weight_data = weight.data();

    for (i, &idx) in indices_data.iter().enumerate() {
        let idx_usize = idx as usize;
        assert!(idx_usize < num_embeddings, "Index out of range");
        let base = i * embedding_dim;
        let weight_base = idx_usize * embedding_dim;
        for d in 0..embedding_dim {
            out_data[base + d] = weight_data[weight_base + d];
        }
    }

    let mut out_shape = indices.shape().to_vec();
    out_shape.push(embedding_dim);

    Tensor::new(out_data, &out_shape)
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn embedding_backward<T: DType>(
    grad_output: &Tensor<T>,
    indices: &Tensor<i64>,
    num_embeddings: usize,
    embedding_dim: usize,
) -> Vec<Tensor<T>> {
    let indices_data = indices.data();
    let mut grad_weight =
        vec![T::from_f32(0.0); num_embeddings * embedding_dim];
    let grad_data = grad_output.data();

    for (i, &idx) in indices_data.iter().enumerate() {
        let idx_usize = idx as usize;
        let base = i * embedding_dim;
        let weight_base = idx_usize * embedding_dim;
        for d in 0..embedding_dim {
            grad_weight[weight_base + d] =
                grad_weight[weight_base + d] + grad_data[base + d];
        }
    }

    let weight_shape = vec![num_embeddings, embedding_dim];
    vec![Tensor::new(grad_weight, &weight_shape)]
}

// ============================================================
// Embedding Op (Standalone Implementation)
// ============================================================

pub struct EmbeddingOp;

impl EmbeddingOp {
    pub fn name(&self) -> &'static str {
        "embedding"
    }
    pub fn forward<T: DType + Send + Sync>(
        &self,
        indices: &Tensor<i64>,
        weight: &Tensor<T>,
    ) -> Tensor<T> {
        embedding(indices, weight)
    }
    pub fn backward<T: DType + Send + Sync>(
        &self,
        grad: &Tensor<T>,
        indices: &Tensor<i64>,
        num_embeddings: usize,
        embedding_dim: usize,
    ) -> Vec<Tensor<T>> {
        embedding_backward(grad, indices, num_embeddings, embedding_dim)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding() {
        let indices = Tensor::new(vec![0, 2, 1], &[3]);
        let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let c = embedding(&indices, &weight);
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0]);
    }

    #[test]
    fn test_embedding_op() {
        let indices = Tensor::new(vec![0, 2, 1], &[3]);
        let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let op = EmbeddingOp;
        let c = op.forward(&indices, &weight);
        assert_eq!(c.data(), &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0]);
    }
}
