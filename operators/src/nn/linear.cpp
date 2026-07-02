#include "infer_train/nn/linear.hpp"

namespace infer_train {

// FP32
template Tensor<F32> linear(
    const Tensor<F32>&,
    const Tensor<F32>&,
    const Tensor<F32>*
);

// FP64
template Tensor<F64> linear(
    const Tensor<F64>&,
    const Tensor<F64>&,
    const Tensor<F64>*
);

// FP16
template Tensor<F16> linear(
    const Tensor<F16>&,
    const Tensor<F16>&,
    const Tensor<F16>*
);

// BF16
template Tensor<BF16> linear(
    const Tensor<BF16>&,
    const Tensor<BF16>&,
    const Tensor<BF16>*
);

} // namespace infer_train
