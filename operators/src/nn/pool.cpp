#include "infer_train/nn/pool.hpp"

namespace infer_train {

// ============================================================
// maxpool1d
// ============================================================
template Tensor<F32> maxpool1d(const Tensor<F32>&, int, int, int);
template Tensor<F64> maxpool1d(const Tensor<F64>&, int, int, int);
template Tensor<F16> maxpool1d(const Tensor<F16>&, int, int, int);
template Tensor<BF16> maxpool1d(const Tensor<BF16>&, int, int, int);

// ============================================================
// maxpool3d
// ============================================================
template Tensor<F32> maxpool3d(const Tensor<F32>&, int, int, int);
template Tensor<F64> maxpool3d(const Tensor<F64>&, int, int, int);
template Tensor<F16> maxpool3d(const Tensor<F16>&, int, int, int);
template Tensor<BF16> maxpool3d(const Tensor<BF16>&, int, int, int);

// ============================================================
// avgpool1d
// ============================================================
template Tensor<F32> avgpool1d(const Tensor<F32>&, int, int, int);
template Tensor<F64> avgpool1d(const Tensor<F64>&, int, int, int);
template Tensor<F16> avgpool1d(const Tensor<F16>&, int, int, int);
template Tensor<BF16> avgpool1d(const Tensor<BF16>&, int, int, int);

// ============================================================
// avgpool3d
// ============================================================
template Tensor<F32> avgpool3d(const Tensor<F32>&, int, int, int);
template Tensor<F64> avgpool3d(const Tensor<F64>&, int, int, int);
template Tensor<F16> avgpool3d(const Tensor<F16>&, int, int, int);
template Tensor<BF16> avgpool3d(const Tensor<BF16>&, int, int, int);

} // namespace infer_train
