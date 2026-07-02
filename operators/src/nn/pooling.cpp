#include "infer_train/nn/pooling.hpp"

namespace infer_train {

// ============================================================
// FP32
// ============================================================
template Tensor<F32> maxpool2d(const Tensor<F32>&, int, int, int);
template Tensor<F32> avgpool2d(const Tensor<F32>&, int, int, int);

// ============================================================
// FP64
// ============================================================
template Tensor<F64> maxpool2d(const Tensor<F64>&, int, int, int);
template Tensor<F64> avgpool2d(const Tensor<F64>&, int, int, int);

// ============================================================
// FP16
// ============================================================
template Tensor<F16> maxpool2d(const Tensor<F16>&, int, int, int);
template Tensor<F16> avgpool2d(const Tensor<F16>&, int, int, int);

// ============================================================
// BF16
// ============================================================
template Tensor<BF16> maxpool2d(const Tensor<BF16>&, int, int, int);
template Tensor<BF16> avgpool2d(const Tensor<BF16>&, int, int, int);

} // namespace infer_train
