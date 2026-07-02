#include "infer_train/math/div.hpp"

namespace infer_train {

// FP32
template Tensor<F32> div(const Tensor<F32>&, const Tensor<F32>&);
template Tensor<F32> div_scalar(const Tensor<F32>&, float);
template Tensor<F32> scalar_div(float, const Tensor<F32>&);

// FP64
template Tensor<F64> div(const Tensor<F64>&, const Tensor<F64>&);
template Tensor<F64> div_scalar(const Tensor<F64>&, double);
template Tensor<F64> scalar_div(double, const Tensor<F64>&);

// FP16
template Tensor<F16> div(const Tensor<F16>&, const Tensor<F16>&);
template Tensor<F16> div_scalar(const Tensor<F16>&, uint16_t);
template Tensor<F16> scalar_div(uint16_t, const Tensor<F16>&);

// BF16
template Tensor<BF16> div(const Tensor<BF16>&, const Tensor<BF16>&);
template Tensor<BF16> div_scalar(const Tensor<BF16>&, uint16_t);
template Tensor<BF16> scalar_div(uint16_t, const Tensor<BF16>&);

} // namespace infer_train
