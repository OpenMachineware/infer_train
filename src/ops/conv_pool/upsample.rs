// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Float Generic Forward (nearest neighbor)
// ============================================================

pub fn upsample<T: DType + Send + Sync>(
    x: &Tensor<T>,
    scale_factor: &[usize],
) -> Tensor<T> {
    let shape = x.shape();
    let spatial_dims = shape.len() - 2;
    assert_eq!(
        spatial_dims,
        scale_factor.len(),
        "Scale factor must match spatial dims"
    );

    let mut out_shape = vec![shape[0], shape[1]];
    for i in 0..spatial_dims {
        out_shape.push(shape[i + 2] * scale_factor[i]);
    }

    let batch = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = if spatial_dims >= 2 { shape[3] } else { 1 };
    let scale_h = scale_factor[0];
    let scale_w = if spatial_dims >= 2 { scale_factor[1] } else { 1 };
    let out_h = h * scale_h;
    let out_w = w * scale_w;

    let x_data = x.data();
    let mut out_data = vec![T::from_f32(0.0); batch * c * out_h * out_w];

    for b in 0..batch {
        for ch in 0..c {
            for oh in 0..out_h {
                let ih = oh / scale_h;
                for ow in 0..out_w {
                    let iw = ow / scale_w;
                    let idx = ((b * c + ch) * h + ih) * w + iw;
                    let out_idx = ((b * c + ch) * out_h + oh) * out_w + ow;
                    out_data[out_idx] = x_data[idx];
                }
            }
        }
    }

    Tensor::new(out_data, &out_shape)
}

// ============================================================
// Float Generic Backward - Simplified   TODO: Improve
// ============================================================

pub fn upsample_backward<T: DType>(
    grad_output: &Tensor<T>,
    _scale_factor: &[usize],
) -> Vec<Tensor<T>> {
    // Simplified: return gradient directly
    vec![grad_output.clone()]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct UpsampleOp;

impl<T: DType + Send + Sync> Operator<T> for UpsampleOp {
    fn name(&self) -> &'static str {
        "upsample"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let scale_factor = attrs
            .get_int_list("scale_factor")
            .expect("upsample requires scale_factor")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        upsample(inputs[0], &scale_factor)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        let scale_factor = attrs
            .get_int_list("scale_factor")
            .expect("upsample requires scale_factor")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        upsample_backward(grad, &scale_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upsample() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let c = upsample(&x, &[2, 2]);
        assert_eq!(c.shape(), &[1, 1, 4, 4]);
        assert_eq!(c.data()[0], 1.0);
        assert_eq!(c.data()[4], 1.0);
        assert_eq!(c.data()[8], 3.0);
    }
}
