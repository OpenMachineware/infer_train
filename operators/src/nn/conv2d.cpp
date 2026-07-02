#include "infer_train/nn/conv2d.hpp"

namespace infer_train {

// FP32
template Tensor<F32> conv2d(
    const Tensor<F32>&,
    const Tensor<F32>&,
    const Tensor<F32>*,
    int, int, int, int
);

// FP64
template Tensor<F64> conv2d(
    const Tensor<F64>&,
    const Tensor<F64>&,
    const Tensor<F64>*,
    int, int, int, int
);

// FP16
template Tensor<F16> conv2d(
    const Tensor<F16>&,
    const Tensor<F16>&,
    const Tensor<F16>*,
    int, int, int, int
);

// BF16
template Tensor<BF16> conv2d(
    const Tensor<BF16>&,
    const Tensor<BF16>&,
    const Tensor<BF16>*,
    int, int, int, int
);

} // namespace infer_train
