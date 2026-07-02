#include "infer_train/nn/layernorm.hpp"

namespace infer_train {

// FP32
template Tensor<F32> layernorm(
    const Tensor<F32>&,
    const Tensor<F32>&,
    const Tensor<F32>&,
    float
);

// FP64
template Tensor<F64> layernorm(
    const Tensor<F64>&,
    const Tensor<F64>&,
    const Tensor<F64>&,
    float
);

// FP16
template Tensor<F16> layernorm(
    const Tensor<F16>&,
    const Tensor<F16>&,
    const Tensor<F16>&,
    float
);

// BF16
template Tensor<BF16> layernorm(
    const Tensor<BF16>&,
    const Tensor<BF16>&,
    const Tensor<BF16>&,
    float
);

} // namespace infer_train
