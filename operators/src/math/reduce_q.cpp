#include "infer_train/math/reduce_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_sum(const Tensor<I8>&, const std::vector<int>&, bool);
template Tensor<I8> quantized_mean(const Tensor<I8>&, const std::vector<int>&, bool);
template Tensor<I8> quantized_max_all(const Tensor<I8>&);
template Tensor<I8> quantized_min_all(const Tensor<I8>&);
}
