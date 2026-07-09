use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

pub fn fused_conv_bn<T: DType + Send + Sync>(
    x: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    _eps: f32,
) -> Tensor<T> {
    // Fused Conv+BN: directly perform convolution,
    // BN parameters already merged into weight and bias
    // Actual implementation can directly call
    // conv2d since BN is already merged at compile time
    crate::ops::conv_pool::conv2d::conv2d(
        x, weight, bias, stride, padding, dilation, groups,
    )
}

pub struct FusedConvBnOp;

impl<T: DType + Send + Sync> Operator<T> for FusedConvBnOp {
    fn name(&self) -> &'static str {
        "fused_conv_bn"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert!(inputs.len() >= 2 && inputs.len() <= 3);
        let stride = attrs.get_int("stride").map(|v| v as usize).unwrap_or(1);
        let padding = attrs.get_int("padding").map(|v| v as usize).unwrap_or(0);
        let dilation =
            attrs.get_int("dilation").map(|v| v as usize).unwrap_or(1);
        let groups = attrs.get_int("groups").map(|v| v as usize).unwrap_or(1);
        let eps = attrs.get_float("eps").unwrap_or(1e-5);
        let bias = if inputs.len() == 3 { Some(inputs[2]) } else { None };
        fused_conv_bn(
            inputs[0], inputs[1], bias, stride, padding, dilation, groups, eps,
        )
    }
    fn backward(
        &self,
        _grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        vec![]
    }
}
