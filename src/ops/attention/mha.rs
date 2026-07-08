// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
// use crate::ops::linalg::matmul;
use super::sdpa::scaled_dot_product_attention;

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn multi_head_attention<T: DType + Send + Sync>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    num_heads: usize,
    scale: f32,
    is_causal: bool,
) -> Tensor<T> {
    let q_shape = query.shape();
    let k_shape = key.shape();
    let v_shape = value.shape();

    assert_eq!(q_shape.len(), 3, "MHA: query must be 3D [B, L, D]");
    assert_eq!(k_shape.len(), 3, "MHA: key must be 3D [B, L, D]");
    assert_eq!(v_shape.len(), 3, "MHA: value must be 3D [B, L, D]");

    let (_batch, _seq_len, embed_dim) = (q_shape[0], q_shape[1], q_shape[2]);
    assert_eq!(
        embed_dim % num_heads,
        0,
        "Embedding dim must be divisible by num_heads"
    );

    let head_dim = embed_dim / num_heads;

    // Reshape: [B, L, D] -> [B, H, L, D/H]
    let q_reshaped = reshape_to_4d(query, num_heads, head_dim);
    let k_reshaped = reshape_to_4d(key, num_heads, head_dim);
    let v_reshaped = reshape_to_4d(value, num_heads, head_dim);

    // SDPA
    let attn_output = scaled_dot_product_attention(
        &q_reshaped,
        &k_reshaped,
        &v_reshaped,
        scale,
        is_causal,
    );

    // Reshape back: [B, H, L, D/H] -> [B, L, D]
    reshape_to_3d(&attn_output, num_heads, head_dim)
}

// ============================================================
// 辅助函数：重排维度
// ============================================================

fn reshape_to_4d<T: DType + Clone>(
    x: &Tensor<T>,
    num_heads: usize,
    head_dim: usize,
) -> Tensor<T> {
    let shape = x.shape();
    let batch = shape[0];
    let seq_len = shape[1];
    let embed_dim = shape[2];

    let mut data =
        vec![T::from_f32(0.0); batch * num_heads * seq_len * head_dim];
    let x_data = x.data();

    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src_idx =
                        (b * seq_len + s) * embed_dim + h * head_dim + d;
                    let dst_idx =
                        ((b * num_heads + h) * seq_len + s) * head_dim + d;
                    data[dst_idx] = x_data[src_idx];
                }
            }
        }
    }

    Tensor::new(data, &[batch, num_heads, seq_len, head_dim])
}

fn reshape_to_3d<T: DType + Clone>(
    x: &Tensor<T>,
    num_heads: usize,
    head_dim: usize,
) -> Tensor<T> {
    let shape = x.shape();
    let batch = shape[0];
    let seq_len = shape[2];
    let embed_dim = num_heads * head_dim;

    let mut data = vec![T::from_f32(0.0); batch * seq_len * embed_dim];
    let x_data = x.data();

    for b in 0..batch {
        for h in 0..num_heads {
            for s in 0..seq_len {
                for d in 0..head_dim {
                    let src_idx =
                        ((b * num_heads + h) * seq_len + s) * head_dim + d;
                    let dst_idx =
                        (b * seq_len + s) * embed_dim + h * head_dim + d;
                    data[dst_idx] = x_data[src_idx];
                }
            }
        }
    }

    Tensor::new(data, &[batch, seq_len, embed_dim])
}

// ============================================================
// 2. 浮点泛型 Backward (简化版)
// ============================================================

pub fn multi_head_attention_backward<T: DType>(
    grad_output: &Tensor<T>,
) -> Vec<Tensor<T>> {
    vec![grad_output.clone(), grad_output.clone(),
         grad_output.clone()]
}

// ============================================================
// 3. Operator Trait 实现
// ============================================================

pub struct MhaOp;

impl<T: DType + Send + Sync> Operator<T> for MhaOp {
    fn name(&self) -> &'static str {
        "multi_head_attention"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 3);
        let num_heads = attrs.get_int("num_heads").unwrap_or(8) as usize;
        let scale = attrs.get_float("scale").unwrap_or(0.0);
        let is_causal = attrs.get_bool("is_causal").unwrap_or(false);
        multi_head_attention(
            inputs[0], inputs[1], inputs[2], num_heads, scale, is_causal,
        )
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        multi_head_attention_backward(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mha() {
        let q = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
        let k = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
        let v = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
        let c = multi_head_attention(&q, &k, &v, 3, 0.0, false);
        assert_eq!(c.shape(), &[1, 2, 3]);
    }
}
