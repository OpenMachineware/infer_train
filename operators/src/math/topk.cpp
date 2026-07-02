#include "infer_train/math/topk.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template std::pair<Tensor<F32>, std::vector<int64_t>> topk(const Tensor<F32>&, size_t, int, bool, bool);
template std::pair<Tensor<F64>, std::vector<int64_t>> topk(const Tensor<F64>&, size_t, int, bool, bool);
template std::pair<Tensor<F16>, std::vector<int64_t>> topk(const Tensor<F16>&, size_t, int, bool, bool);
template std::pair<Tensor<BF16>, std::vector<int64_t>> topk(const Tensor<BF16>&, size_t, int, bool, bool);

} // namespace infer_train
