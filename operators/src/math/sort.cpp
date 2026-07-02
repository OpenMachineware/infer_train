#include "infer_train/math/sort.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template std::pair<Tensor<F32>, std::vector<int64_t>> sort(const Tensor<F32>&, int, bool);
template std::pair<Tensor<F64>, std::vector<int64_t>> sort(const Tensor<F64>&, int, bool);
template std::pair<Tensor<F16>, std::vector<int64_t>> sort(const Tensor<F16>&, int, bool);
template std::pair<Tensor<BF16>, std::vector<int64_t>> sort(const Tensor<BF16>&, int, bool);

} // namespace infer_train
