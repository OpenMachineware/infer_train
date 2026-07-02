#include "infer_train/math/abs.hpp"

namespace infer_train {

// FP32
template Tensor<F32> abs(const Tensor<F32>&);
template Tensor<F32> neg(const Tensor<F32>&);

// FP64
template Tensor<F64> abs(const Tensor<F64>&);
template Tensor<F64> neg(const Tensor<F64>&);

// FP16
template Tensor<F16> abs(const Tensor<F16>&);
template Tensor<F16> neg(const Tensor<F16>&);

// BF16
template Tensor<BF16> abs(const Tensor<BF16>&);
template Tensor<BF16> neg(const Tensor<BF16>&);

} // namespace infer_train
