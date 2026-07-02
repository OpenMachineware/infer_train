#include "infer_train/nn/norm_q.hpp"

namespace infer_train {

template Tensor<I8> quantized_instancenorm2d(const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, float);
template Tensor<I8> quantized_groupnorm(const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, int, float);

} // namespace infer_train
