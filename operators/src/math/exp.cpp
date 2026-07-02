#include "infer_train/math/exp.hpp"

namespace infer_train {

// FP32
template Tensor<F32> exp(const Tensor<F32>&);
template Tensor<F32> expm1(const Tensor<F32>&);

// FP64
template Tensor<F64> exp(const Tensor<F64>&);
template Tensor<F64> expm1(const Tensor<F64>&);

// FP16
template Tensor<F16> exp(const Tensor<F16>&);
template Tensor<F16> expm1(const Tensor<F16>&);

// BF16
template Tensor<BF16> exp(const Tensor<BF16>&);
template Tensor<BF16> expm1(const Tensor<BF16>&);

} // namespace infer_train
