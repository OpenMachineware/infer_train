#include "infer_train/math/clamp.hpp"

namespace infer_train {

// FP32
template Tensor<F32> clamp(const Tensor<F32>&, float, float);
template Tensor<F32> clamp_min(const Tensor<F32>&, float);
template Tensor<F32> clamp_max(const Tensor<F32>&, float);

// FP64
template Tensor<F64> clamp(const Tensor<F64>&, double, double);
template Tensor<F64> clamp_min(const Tensor<F64>&, double);
template Tensor<F64> clamp_max(const Tensor<F64>&, double);

// FP16
template Tensor<F16> clamp(const Tensor<F16>&, uint16_t, uint16_t);
template Tensor<F16> clamp_min(const Tensor<F16>&, uint16_t);
template Tensor<F16> clamp_max(const Tensor<F16>&, uint16_t);

// BF16
template Tensor<BF16> clamp(const Tensor<BF16>&, uint16_t, uint16_t);
template Tensor<BF16> clamp_min(const Tensor<BF16>&, uint16_t);
template Tensor<BF16> clamp_max(const Tensor<BF16>&, uint16_t);

} // namespace infer_train
