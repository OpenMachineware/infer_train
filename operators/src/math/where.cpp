#include "infer_train/math/where.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> where(const std::vector<uint8_t>&, const std::vector<size_t>&, const Tensor<F32>&, const Tensor<F32>&);
template Tensor<F64> where(const std::vector<uint8_t>&, const std::vector<size_t>&, const Tensor<F64>&, const Tensor<F64>&);
template Tensor<F16> where(const std::vector<uint8_t>&, const std::vector<size_t>&, const Tensor<F16>&, const Tensor<F16>&);
template Tensor<BF16> where(const std::vector<uint8_t>&, const std::vector<size_t>&, const Tensor<BF16>&, const Tensor<BF16>&);

template Tensor<F32> where_scalar(const std::vector<uint8_t>&, const std::vector<size_t>&, float, float);
template Tensor<F64> where_scalar(const std::vector<uint8_t>&, const std::vector<size_t>&, double, double);
template Tensor<F16> where_scalar(const std::vector<uint8_t>&, const std::vector<size_t>&, uint16_t, uint16_t);
template Tensor<BF16> where_scalar(const std::vector<uint8_t>&, const std::vector<size_t>&, uint16_t, uint16_t);

} // namespace infer_train
