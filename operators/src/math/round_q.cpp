#include "infer_train/math/round_q.hpp"

namespace infer_train {

template Tensor<I8> quantized_floor(const Tensor<I8>&);
template Tensor<I8> quantized_ceil(const Tensor<I8>&);
template Tensor<I8> quantized_round(const Tensor<I8>&);

} // namespace infer_train
