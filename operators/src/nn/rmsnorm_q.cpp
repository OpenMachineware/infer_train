#include "infer_train/nn/rmsnorm_q.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<I8> quantized_rmsnorm(const Tensor<I8>&, const Tensor<I8>&, float);

} // namespace infer_train
