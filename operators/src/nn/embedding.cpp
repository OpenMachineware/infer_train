#include "infer_train/nn/embedding.hpp"

namespace infer_train {

template Tensor<F32> embedding(const std::vector<int64_t>&, const Tensor<F32>&, int);
template Tensor<F64> embedding(const std::vector<int64_t>&, const Tensor<F64>&, int);
template Tensor<F16> embedding(const std::vector<int64_t>&, const Tensor<F16>&, int);
template Tensor<BF16> embedding(const std::vector<int64_t>&, const Tensor<BF16>&, int);

} // namespace infer_train