#include "infer_train/math/std_var.hpp"

namespace infer_train {

template Tensor<F32> var(const Tensor<F32>&, bool);
template Tensor<F32> std(const Tensor<F32>&, bool);

template Tensor<F64> var(const Tensor<F64>&, bool);
template Tensor<F64> std(const Tensor<F64>&, bool);

template Tensor<F16> var(const Tensor<F16>&, bool);
template Tensor<F16> std(const Tensor<F16>&, bool);

template Tensor<BF16> var(const Tensor<BF16>&, bool);
template Tensor<BF16> std(const Tensor<BF16>&, bool);

} // namespace infer_train
