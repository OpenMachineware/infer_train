// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::activation::softmax;
use crate::ops::linalg::matmul;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Float Generic Forward
// ============================================================

pub fn scaled_dot_product_attention<T: DType + Send + Sync>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    scale: f32,
    is_causal: bool,
) -> Tensor<T> {
    let q_shape = query.shape();
    let k_shape = key.shape();
    let v_shape = value.shape();

    assert_eq!(q_shape.len(), 4, "SDPA: query must be 4D [B, H, L, D]");
    assert_eq!(k_shape.len(), 4, "SDPA: key must be 4D [B, H, L, D]");
    assert_eq!(v_shape.len(), 4, "SDPA: value must be 4D [B, H, L, D]");

    let (batch, heads, seq_len, head_dim) =
        (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    assert_eq!(key.shape()[3], head_dim, "Key head dim mismatch");
    assert_eq!(value.shape()[2], seq_len, "Value seq len mismatch");
    assert_eq!(value.shape()[3], head_dim, "Value head dim mismatch");

    // Q * K^T / sqrt(d)
    let k_t = crate::ops::linalg::transpose::transpose(key);
    let mut scores = matmul::matmul(query, &k_t);

    // Apply scale
    let scale_val =
        if scale == 0.0 { 1.0 / (head_dim as f32).sqrt() } else { scale };

    for v in scores.data_mut() {
        *v = T::from_f32(v.to_f32() * scale_val);
    }

    // Apply causal mask (if enabled)
    if is_causal {
        // Simplified: apply mask to last two dimensions only
        let score_shape = scores.shape();
        let l = score_shape[2];
        for b in 0..batch {
            for h in 0..heads {
                for i in 0..l {
                    for j in 0..l {
                        if j > i {
                            let idx = ((b * heads + h) * l + i) * l + j;
                            scores.data_mut()[idx] =
                                T::from_f32(f32::NEG_INFINITY);
                        }
                    }
                }
            }
        }
    }

    // Softmax over last dimension
    let attn = softmax::softmax(&scores, scores.shape().len() - 1);

    // Attn * V
    let output = matmul::matmul(&attn, value);
    output
}

// ============================================================
// Float Generic Backward - Simplified TODO: Improve
// ============================================================

pub fn scaled_dot_product_attention_backward<T: DType>(
    grad_output: &Tensor<T>,
    _query: &Tensor<T>,
    _key: &Tensor<T>,
    _value: &Tensor<T>,
) -> Vec<Tensor<T>> {
    vec![grad_output.clone(), grad_output.clone(), grad_output.clone()]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct SdpaOp;

impl<T: DType + Send + Sync> Operator<T> for SdpaOp {
    fn name(&self) -> &'static str {
        "scaled_dot_product_attention"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 3);
        let scale = attrs.get_float("scale").unwrap_or(0.0);
        let is_causal = attrs.get_bool("is_causal").unwrap_or(false);
        scaled_dot_product_attention(
            inputs[0], inputs[1], inputs[2], scale, is_causal,
        )
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 3);
        scaled_dot_product_attention_backward(
            grad, inputs[0], inputs[1], inputs[2],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdpa() {
        let q = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let k = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let v = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let c = scaled_dot_product_attention(&q, &k, &v, 0.5, false);
        assert_eq!(c.shape(), &[1, 1, 2, 2]);
    }
}
