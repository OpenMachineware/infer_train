// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Helper Functions
// ============================================================

pub fn pool_output_size(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    ceil_mode: bool,
) -> usize {
    if ceil_mode {
        ((input + 2 * padding - kernel) as f64 / stride as f64).ceil() as usize
            + 1
    } else {
        ((input + 2 * padding - kernel) as f64 / stride as f64).floor() as usize
            + 1
    }
}

// ============================================================
// Float Generic Forward
// ============================================================

pub fn max_pool<T: DType + Send + Sync>(
    x: &Tensor<T>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Tensor<T> {
    let shape = x.shape();
    assert!(shape.len() >= 3, "max_pool requires at least 3D tensor");

    let spatial_dims = shape.len() - 2;
    let (h, w) =
        if spatial_dims == 1 { (shape[2], 1) } else { (shape[2], shape[3]) };

    let out_h = pool_output_size(h, kernel_size, stride, padding, false);
    let out_w = if spatial_dims == 1 {
        1
    } else {
        pool_output_size(w, kernel_size, stride, padding, false)
    };

    let c = shape[1];
    let batch = shape[0];

    let x_data = x.data();
    let mut out_data = vec![T::from_f32(0.0); batch * c * out_h * out_w];

    for b in 0..batch {
        for ch in 0..c {
            for oh in 0..out_h {
                let h_start = oh * stride;
                for ow in 0..out_w {
                    let w_start = ow * stride;
                    let mut max_val = f32::NEG_INFINITY;

                    for kh in 0..kernel_size {
                        let ih = h_start + kh;
                        if ih >= h {
                            continue;
                        }
                        for kw in 0..kernel_size {
                            let iw = w_start + kw;
                            if iw >= w {
                                continue;
                            }
                            let idx = ((b * c + ch) * h + ih) * w + iw;
                            let v = x_data[idx].to_f32();
                            if v > max_val {
                                max_val = v;
                            }
                        }
                    }

                    let out_idx = ((b * c + ch) * out_h + oh) * out_w + ow;
                    out_data[out_idx] = T::from_f32(max_val);
                }
            }
        }
    }

    Tensor::new(out_data, &[batch, c, out_h, out_w])
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn max_pool_backward<T: DType + Send + Sync>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    kernel_size: usize,
    stride: usize,
    _padding: usize,
) -> Tensor<T> {
    let i_shape = input.shape();
    let g_shape = grad_output.shape();
    let n = i_shape[0];
    let c = i_shape[1];
    let h = i_shape[2];
    let w = i_shape[3];
    let oh = g_shape[2];
    let ow = g_shape[3];

    let mut grad_input = vec![T::zero(); n * c * h * w];
    let input_data = input.data();
    let grad_data = grad_output.data();

    for b in 0..n {
        for ch in 0..c {
            for i in 0..oh {
                for j in 0..ow {
                    let grad_val = grad_data[((b * c + ch) * oh + i) * ow + j];
                    if grad_val == T::zero() {
                        continue;
                    }

                    let h_start = i * stride;
                    let w_start = j * stride;
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_h = h_start;
                    let mut max_w = w_start;

                    for kh in 0..kernel_size {
                        let ih = h_start + kh;
                        if ih >= h {
                            continue;
                        }
                        for kw in 0..kernel_size {
                            let iw = w_start + kw;
                            if iw >= w {
                                continue;
                            }
                            let idx = ((b * c + ch) * h + ih) * w + iw;
                            let v = input_data[idx].to_f32();
                            if v > max_val {
                                max_val = v;
                                max_h = ih;
                                max_w = iw;
                            }
                        }
                    }

                    let grad_idx = ((b * c + ch) * h + max_h) * w + max_w;
                    grad_input[grad_idx] = grad_input[grad_idx] + grad_val;
                }
            }
        }
    }

    Tensor::new(grad_input, &[n, c, h, w])
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct MaxPoolOp;

impl<T: DType + Send + Sync> Operator<T> for MaxPoolOp {
    fn name(&self) -> &'static str {
        "max_pool"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let kernel_size =
            attrs.get_int("kernel_size").map(|v| v as usize).unwrap_or(2);
        let stride =
            attrs.get_int("stride").map(|v| v as usize).unwrap_or(kernel_size);
        let padding = attrs.get_int("padding").map(|v| v as usize).unwrap_or(0);
        max_pool(inputs[0], kernel_size, stride, padding)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        let kernel_size =
            attrs.get_int("kernel_size").map(|v| v as usize).unwrap_or(2);
        let stride =
            attrs.get_int("stride").map(|v| v as usize).unwrap_or(kernel_size);
        let padding = attrs.get_int("padding").map(|v| v as usize).unwrap_or(0);
        vec![max_pool_backward(grad, inputs[0], kernel_size, stride, padding)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        );
        let c = max_pool(&x, 2, 2, 0);
        assert_eq!(c.shape(), &[1, 1, 1, 1]);
        assert_eq!(c.data()[0], 5.0);
    }

    #[test]
    fn test_max_pool_stride() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        );
        let c = max_pool(&x, 2, 1, 0);
        assert_eq!(c.shape(), &[1, 1, 2, 2]);
        assert_eq!(c.data()[0], 5.0);
        assert_eq!(c.data()[1], 6.0);
        assert_eq!(c.data()[2], 8.0);
        assert_eq!(c.data()[3], 9.0);
    }
}
