#include "infer_train/math/cumprod.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> cumprod(const Tensor<F32>&, int);
template Tensor<F64> cumprod(const Tensor<F64>&, int);
template Tensor<F16> cumprod(const Tensor<F16>&, int);
template Tensor<BF16> cumprod(const Tensor<BF16>&, int);

} // namespace infer_train
