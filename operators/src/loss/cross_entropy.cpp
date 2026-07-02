#include "infer_train/loss/cross_entropy.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> cross_entropy_loss(const Tensor<F32>&, const std::vector<int64_t>&, bool);
template Tensor<F64> cross_entropy_loss(const Tensor<F64>&, const std::vector<int64_t>&, bool);
template Tensor<F16> cross_entropy_loss(const Tensor<F16>&, const std::vector<int64_t>&, bool);
template Tensor<BF16> cross_entropy_loss(const Tensor<BF16>&, const std::vector<int64_t>&, bool);

} // namespace infer_train
