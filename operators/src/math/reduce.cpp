#include "infer_train/math/reduce.hpp"

namespace infer_train {

// ============================================================
// FP32
// ============================================================
template Tensor<F32> sum(const Tensor<F32>&, const std::vector<int>&, bool);
template Tensor<F32> mean(const Tensor<F32>&, const std::vector<int>&, bool);
template Tensor<F32> max_all(const Tensor<F32>&);
template Tensor<F32> min_all(const Tensor<F32>&);

// ============================================================
// FP64
// ============================================================
template Tensor<F64> sum(const Tensor<F64>&, const std::vector<int>&, bool);
template Tensor<F64> mean(const Tensor<F64>&, const std::vector<int>&, bool);
template Tensor<F64> max_all(const Tensor<F64>&);
template Tensor<F64> min_all(const Tensor<F64>&);

// ============================================================
// FP16
// ============================================================
template Tensor<F16> sum(const Tensor<F16>&, const std::vector<int>&, bool);
template Tensor<F16> mean(const Tensor<F16>&, const std::vector<int>&, bool);
template Tensor<F16> max_all(const Tensor<F16>&);
template Tensor<F16> min_all(const Tensor<F16>&);

// ============================================================
// BF16
// ============================================================
template Tensor<BF16> sum(const Tensor<BF16>&, const std::vector<int>&, bool);
template Tensor<BF16> mean(const Tensor<BF16>&, const std::vector<int>&, bool);
template Tensor<BF16> max_all(const Tensor<BF16>&);
template Tensor<BF16> min_all(const Tensor<BF16>&);

template Tensor<F32> prod_all(const Tensor<F32>&);
template Tensor<F64> prod_all(const Tensor<F64>&);
template Tensor<F16> prod_all(const Tensor<F16>&);
template Tensor<BF16> prod_all(const Tensor<BF16>&);

} // namespace infer_train
