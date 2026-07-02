#include "infer_train/nn/norm.hpp"

namespace infer_train {

// ============================================================
// instancenorm2d
// ============================================================
template Tensor<F32> instancenorm2d(const Tensor<F32>&, const Tensor<F32>&, const Tensor<F32>&, float);
template Tensor<F64> instancenorm2d(const Tensor<F64>&, const Tensor<F64>&, const Tensor<F64>&, float);
template Tensor<F16> instancenorm2d(const Tensor<F16>&, const Tensor<F16>&, const Tensor<F16>&, float);
template Tensor<BF16> instancenorm2d(const Tensor<BF16>&, const Tensor<BF16>&, const Tensor<BF16>&, float);

// ============================================================
// groupnorm
// ============================================================
template Tensor<F32> groupnorm(const Tensor<F32>&, const Tensor<F32>&, const Tensor<F32>&, int, float);
template Tensor<F64> groupnorm(const Tensor<F64>&, const Tensor<F64>&, const Tensor<F64>&, int, float);
template Tensor<F16> groupnorm(const Tensor<F16>&, const Tensor<F16>&, const Tensor<F16>&, int, float);
template Tensor<BF16> groupnorm(const Tensor<BF16>&, const Tensor<BF16>&, const Tensor<BF16>&, int, float);

} // namespace infer_train
