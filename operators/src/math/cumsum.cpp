#include "infer_train/math/cumsum.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> cumsum(const Tensor<F32>&, int);
template Tensor<F64> cumsum(const Tensor<F64>&, int);
template Tensor<F16> cumsum(const Tensor<F16>&, int);
template Tensor<BF16> cumsum(const Tensor<BF16>&, int);

} // namespace infer_train
