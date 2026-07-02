#include "infer_train/math/sqrt.hpp"

namespace infer_train {

// FP32
template Tensor<F32> sqrt(const Tensor<F32>&);
template Tensor<F32> rsqrt(const Tensor<F32>&);

// FP64
template Tensor<F64> sqrt(const Tensor<F64>&);
template Tensor<F64> rsqrt(const Tensor<F64>&);

// FP16
template Tensor<F16> sqrt(const Tensor<F16>&);
template Tensor<F16> rsqrt(const Tensor<F16>&);

// BF16
template Tensor<BF16> sqrt(const Tensor<BF16>&);
template Tensor<BF16> rsqrt(const Tensor<BF16>&);

} // namespace infer_train
