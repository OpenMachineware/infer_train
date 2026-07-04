#include "infer_train/math/reshape.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> reshape(const Tensor<F32>&, const std::vector<size_t>&);
template Tensor<F64> reshape(const Tensor<F64>&, const std::vector<size_t>&);
template Tensor<F16> reshape(const Tensor<F16>&, const std::vector<size_t>&);
template Tensor<BF16> reshape(const Tensor<BF16>&, const std::vector<size_t>&);
template Tensor<I8> reshape(const Tensor<I8>&, const std::vector<size_t>&);

} // namespace infer_train
