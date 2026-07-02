#include "infer_train/math/round.hpp"

namespace infer_train {

template Tensor<F32> floor(const Tensor<F32>&);
template Tensor<F32> ceil(const Tensor<F32>&);
template Tensor<F32> round(const Tensor<F32>&);

template Tensor<F64> floor(const Tensor<F64>&);
template Tensor<F64> ceil(const Tensor<F64>&);
template Tensor<F64> round(const Tensor<F64>&);

template Tensor<F16> floor(const Tensor<F16>&);
template Tensor<F16> ceil(const Tensor<F16>&);
template Tensor<F16> round(const Tensor<F16>&);

template Tensor<BF16> floor(const Tensor<BF16>&);
template Tensor<BF16> ceil(const Tensor<BF16>&);
template Tensor<BF16> round(const Tensor<BF16>&);

} // namespace infer_train
