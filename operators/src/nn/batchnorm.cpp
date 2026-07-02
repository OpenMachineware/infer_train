#include "infer_train/nn/batchnorm.hpp"

namespace infer_train {

// ============================================================
// batchnorm2d
// ============================================================
template Tensor<F32> batchnorm2d_inference(const Tensor<F32>&, const Tensor<F32>&, const Tensor<F32>&, const Tensor<F32>&, const Tensor<F32>&, float);
template Tensor<F64> batchnorm2d_inference(const Tensor<F64>&, const Tensor<F64>&, const Tensor<F64>&, const Tensor<F64>&, const Tensor<F64>&, float);
template Tensor<F16> batchnorm2d_inference(const Tensor<F16>&, const Tensor<F16>&, const Tensor<F16>&, const Tensor<F16>&, const Tensor<F16>&, float);
template Tensor<BF16> batchnorm2d_inference(const Tensor<BF16>&, const Tensor<BF16>&, const Tensor<BF16>&, const Tensor<BF16>&, const Tensor<BF16>&, float);

// ============================================================
// batchnorm1d
// ============================================================
template Tensor<F32> batchnorm1d_inference(const Tensor<F32>&, const Tensor<F32>&, const Tensor<F32>&, const Tensor<F32>&, const Tensor<F32>&, float);
template Tensor<F64> batchnorm1d_inference(const Tensor<F64>&, const Tensor<F64>&, const Tensor<F64>&, const Tensor<F64>&, const Tensor<F64>&, float);
template Tensor<F16> batchnorm1d_inference(const Tensor<F16>&, const Tensor<F16>&, const Tensor<F16>&, const Tensor<F16>&, const Tensor<F16>&, float);
template Tensor<BF16> batchnorm1d_inference(const Tensor<BF16>&, const Tensor<BF16>&, const Tensor<BF16>&, const Tensor<BF16>&, const Tensor<BF16>&, float);

} // namespace infer_train