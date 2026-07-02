#include "infer_train/math/slice.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> slice(const Tensor<F32>&, int, int, int, int);
template Tensor<F64> slice(const Tensor<F64>&, int, int, int, int);
template Tensor<F16> slice(const Tensor<F16>&, int, int, int, int);
template Tensor<BF16> slice(const Tensor<BF16>&, int, int, int, int);
template Tensor<I8> slice(const Tensor<I8>&, int, int, int, int);

} // namespace infer_train
