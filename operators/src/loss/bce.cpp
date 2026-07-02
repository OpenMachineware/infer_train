#include "infer_train/loss/bce.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> bce_loss(const Tensor<F32>&, const Tensor<F32>&, bool, float);
template Tensor<F64> bce_loss(const Tensor<F64>&, const Tensor<F64>&, bool, float);
template Tensor<F16> bce_loss(const Tensor<F16>&, const Tensor<F16>&, bool, float);
template Tensor<BF16> bce_loss(const Tensor<BF16>&, const Tensor<BF16>&, bool, float);

} // namespace infer_train
