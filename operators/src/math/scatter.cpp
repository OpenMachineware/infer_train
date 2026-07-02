#include "infer_train/math/scatter.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

// 标量版本
template Tensor<F32> scatter(const Tensor<F32>&, const std::vector<int64_t>&, const std::vector<size_t>&, float, int);
template Tensor<F64> scatter(const Tensor<F64>&, const std::vector<int64_t>&, const std::vector<size_t>&, double, int);
template Tensor<F16> scatter(const Tensor<F16>&, const std::vector<int64_t>&, const std::vector<size_t>&, uint16_t, int);
template Tensor<BF16> scatter(const Tensor<BF16>&, const std::vector<int64_t>&, const std::vector<size_t>&, uint16_t, int);

// 张量版本
template Tensor<F32> scatter(const Tensor<F32>&, const std::vector<int64_t>&, const std::vector<size_t>&, const Tensor<F32>&, int);
template Tensor<F64> scatter(const Tensor<F64>&, const std::vector<int64_t>&, const std::vector<size_t>&, const Tensor<F64>&, int);
template Tensor<F16> scatter(const Tensor<F16>&, const std::vector<int64_t>&, const std::vector<size_t>&, const Tensor<F16>&, int);
template Tensor<BF16> scatter(const Tensor<BF16>&, const std::vector<int64_t>&, const std::vector<size_t>&, const Tensor<BF16>&, int);

} // namespace infer_train
