#include "infer_train/math/log.hpp"

namespace infer_train {

template Tensor<F32> log(const Tensor<F32>&);
template Tensor<F32> log2(const Tensor<F32>&);
template Tensor<F32> log10(const Tensor<F32>&);

template Tensor<F64> log(const Tensor<F64>&);
template Tensor<F64> log2(const Tensor<F64>&);
template Tensor<F64> log10(const Tensor<F64>&);

template Tensor<F16> log(const Tensor<F16>&);
template Tensor<F16> log2(const Tensor<F16>&);
template Tensor<F16> log10(const Tensor<F16>&);

template Tensor<BF16> log(const Tensor<BF16>&);
template Tensor<BF16> log2(const Tensor<BF16>&);
template Tensor<BF16> log10(const Tensor<BF16>&);

} // namespace infer_train
