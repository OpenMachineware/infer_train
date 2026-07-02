#include "infer_train/nn/swiglu.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> swiglu(const Tensor<F32>&);
template Tensor<F64> swiglu(const Tensor<F64>&);
template Tensor<F16> swiglu(const Tensor<F16>&);
template Tensor<BF16> swiglu(const Tensor<BF16>&);

} // namespace infer_train
