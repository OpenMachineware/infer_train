#include "infer_train/math/arg.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

// ============================================================
// argmax
// ============================================================
template std::vector<int64_t> argmax(const Tensor<F32>&);
template std::vector<int64_t> argmax(const Tensor<F64>&);
template std::vector<int64_t> argmax(const Tensor<F16>&);
template std::vector<int64_t> argmax(const Tensor<BF16>&);
// I8 版本不是模板，不需要实例化

// ============================================================
// argmin
// ============================================================
template std::vector<int64_t> argmin(const Tensor<F32>&);
template std::vector<int64_t> argmin(const Tensor<F64>&);
template std::vector<int64_t> argmin(const Tensor<F16>&);
template std::vector<int64_t> argmin(const Tensor<BF16>&);
// I8 版本不是模板，不需要实例化

} // namespace infer_train
