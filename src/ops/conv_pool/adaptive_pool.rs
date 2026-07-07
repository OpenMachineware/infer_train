// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 1. Adaptive Avg Pool Forward
// ============================================================

pub fn adaptive_avg_pool<T: DType + Send + Sync>(
    x: &Tensor<T>,
    output_size: &[usize],
) -> Tensor<T> {
    let shape = x.shape();
    let spatial_dims = shape.len() - 2;
    assert_eq!(spatial_dims, output_size.len(), "Output size must match spatial dims");

    let mut out_shape = vec![shape[0], shape[1]];
    out_shape.extend_from_slice(output_size);

    let batch = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = if spatial_dims >= 2 { shape[3] } else { 1 };
    let out_h = output_size[0];
    let out_w = if spatial_dims >= 2 { output_size[1] } else { 1 };

    let x_data = x.data();
    let mut out_data = vec![T::from_f32(0.0); batch * c * out_h * out_w];

    let _h_stride = (h as f32 / out_h as f32).ceil() as usize;
    let _w_stride = (w as f32 / out_w as f32).ceil() as usize;

    for b in 0..batch {
        for ch in 0..c {
            for oh in 0..out_h {
                let h_start = oh * h / out_h;
                let h_end = (oh + 1) * h / out_h;
                for ow in 0..out_w {
                    let w_start = ow * w / out_w;
                    let w_end = (ow + 1) * w / out_w;

                    let mut sum = 0.0;
                    let mut count = 0;
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            let idx = ((b * c + ch) * h + ih) * w + iw;
                            sum += x_data[idx].to_f32();
                            count += 1;
                        }
                    }
                    let out_idx = ((b * c + ch) * out_h + oh) * out_w + ow;
                    out_data[out_idx] = T::from_f32(sum / count as f32);
                }
            }
        }
    }

    Tensor::new(out_data, &out_shape)
}

// ============================================================
// 2. Adaptive Max Pool Forward
// ============================================================

pub fn adaptive_max_pool<T: DType + Send + Sync>(
    x: &Tensor<T>,
    output_size: &[usize],
) -> Tensor<T> {
    let shape = x.shape();
    let spatial_dims = shape.len() - 2;
    assert_eq!(spatial_dims, output_size.len());

    let mut out_shape = vec![shape[0], shape[1]];
    out_shape.extend_from_slice(output_size);

    let batch = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = if spatial_dims >= 2 { shape[3] } else { 1 };
    let out_h = output_size[0];
    let out_w = if spatial_dims >= 2 { output_size[1] } else { 1 };

    let x_data = x.data();
    let mut out_data = vec![T::from_f32(0.0); batch * c * out_h * out_w];

    for b in 0..batch {
        for ch in 0..c {
            for oh in 0..out_h {
                let h_start = oh * h / out_h;
                let h_end = (oh + 1) * h / out_h;
                for ow in 0..out_w {
                    let w_start = ow * w / out_w;
                    let w_end = (ow + 1) * w / out_w;

                    let mut max_val = f32::NEG_INFINITY;
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            let idx = ((b * c + ch) * h + ih) * w + iw;
                            let v = x_data[idx].to_f32();
                            if v > max_val { max_val = v; }
                        }
                    }
                    let out_idx = ((b * c + ch) * out_h + oh) * out_w + ow;
                    out_data[out_idx] = T::from_f32(max_val);
                }
            }
        }
    }

    Tensor::new(out_data, &out_shape)
}

// ============================================================
// 3. Operator Trait 实现
// ============================================================

pub struct AdaptivePoolOp;

impl<T: DType + Send + Sync> Operator<T> for AdaptivePoolOp {
    fn name(&self) -> &'static str { "adaptive_pool" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let output_size = attrs.get_int_list("output_size")
            .expect("adaptive_pool requires output_size")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        let pool_type = attrs.get_string("pool_type").unwrap_or("avg".to_string());
        if pool_type == "max" {
            adaptive_max_pool(inputs[0], &output_size)
        } else {
            adaptive_avg_pool(inputs[0], &output_size)
        }
    }
    fn backward(&self, grad: &Tensor<T>, _inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Vec<Tensor<T>> {
        vec![grad.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_avg_pool() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[1, 1, 3, 3]);
        let c = adaptive_avg_pool(&x, &[2, 2]);
        assert_eq!(c.shape(), &[1, 1, 2, 2]);
    }
}
