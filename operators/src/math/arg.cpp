#include "infer_train/math/arg.hpp"

namespace infer_train {

template Tensor<int64_t> argmax(const Tensor<F32>&, int);
template Tensor<int64_t> argmax(const Tensor<F64>&, int);
template Tensor<int64_t> argmax(const Tensor<F16>&, int);
template Tensor<int64_t> argmax(const Tensor<BF16>&, int);

template Tensor<int64_t> argmin(const Tensor<F32>&, int);
template Tensor<int64_t> argmin(const Tensor<F64>&, int);
template Tensor<int64_t> argmin(const Tensor<F16>&, int);
template Tensor<int64_t> argmin(const Tensor<BF16>&, int);

} // namespace infer_train
