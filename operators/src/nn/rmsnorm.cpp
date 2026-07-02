#include "infer_train/nn/rmsnorm.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> rmsnorm(const Tensor<F32>&, const Tensor<F32>&, float);
template Tensor<F64> rmsnorm(const Tensor<F64>&, const Tensor<F64>&, float);
template Tensor<F16> rmsnorm(const Tensor<F16>&, const Tensor<F16>&, float);
template Tensor<BF16> rmsnorm(const Tensor<BF16>&, const Tensor<BF16>&, float);

} // namespace infer_train
