#include "infer_train/math/mul.hpp"

namespace infer_train {

// FP32
template Tensor<F32> mul(const Tensor<F32>&, const Tensor<F32>&);
template Tensor<F32> mul_scalar(const Tensor<F32>&, float);

// FP64
template Tensor<F64> mul(const Tensor<F64>&, const Tensor<F64>&);
template Tensor<F64> mul_scalar(const Tensor<F64>&, double);

// FP16
template Tensor<F16> mul(const Tensor<F16>&, const Tensor<F16>&);
template Tensor<F16> mul_scalar(const Tensor<F16>&, uint16_t);

// BF16
template Tensor<BF16> mul(const Tensor<BF16>&, const Tensor<BF16>&);
template Tensor<BF16> mul_scalar(const Tensor<BF16>&, uint16_t);

} // namespace infer_train
