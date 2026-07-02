#include "infer_train/nn/conv.hpp"

namespace infer_train {

// ============================================================
// conv1d
// ============================================================
template Tensor<F32> conv1d(const Tensor<F32>&, const Tensor<F32>&, const Tensor<F32>*, int, int, int, int);
template Tensor<F64> conv1d(const Tensor<F64>&, const Tensor<F64>&, const Tensor<F64>*, int, int, int, int);
template Tensor<F16> conv1d(const Tensor<F16>&, const Tensor<F16>&, const Tensor<F16>*, int, int, int, int);
template Tensor<BF16> conv1d(const Tensor<BF16>&, const Tensor<BF16>&, const Tensor<BF16>*, int, int, int, int);

// ============================================================
// conv3d
// ============================================================
template Tensor<F32> conv3d(const Tensor<F32>&, const Tensor<F32>&, const Tensor<F32>*, int, int, int, int);
template Tensor<F64> conv3d(const Tensor<F64>&, const Tensor<F64>&, const Tensor<F64>*, int, int, int, int);
template Tensor<F16> conv3d(const Tensor<F16>&, const Tensor<F16>&, const Tensor<F16>*, int, int, int, int);
template Tensor<BF16> conv3d(const Tensor<BF16>&, const Tensor<BF16>&, const Tensor<BF16>*, int, int, int, int);

} // namespace infer_train
