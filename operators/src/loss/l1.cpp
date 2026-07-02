#include "infer_train/loss/l1.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> l1_loss(const Tensor<F32>&, const Tensor<F32>&, bool);
template Tensor<F64> l1_loss(const Tensor<F64>&, const Tensor<F64>&, bool);
template Tensor<F16> l1_loss(const Tensor<F16>&, const Tensor<F16>&, bool);
template Tensor<BF16> l1_loss(const Tensor<BF16>&, const Tensor<BF16>&, bool);

} // namespace infer_train
