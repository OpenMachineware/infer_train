#include "infer_train/math/arg_q.hpp"

namespace infer_train {

template Tensor<int64_t> quantized_argmax(const Tensor<I8>&, int);
template Tensor<int64_t> quantized_argmin(const Tensor<I8>&, int);

} // namespace infer_train
