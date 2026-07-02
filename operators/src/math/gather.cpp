#include "infer_train/math/gather.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> gather(const Tensor<F32>&, const std::vector<int64_t>&, const std::vector<size_t>&, int);
template Tensor<F64> gather(const Tensor<F64>&, const std::vector<int64_t>&, const std::vector<size_t>&, int);
template Tensor<F16> gather(const Tensor<F16>&, const std::vector<int64_t>&, const std::vector<size_t>&, int);
template Tensor<BF16> gather(const Tensor<BF16>&, const std::vector<int64_t>&, const std::vector<size_t>&, int);

} // namespace infer_train
