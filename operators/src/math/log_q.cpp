#include "infer_train/math/log_q.hpp"

namespace infer_train {

template Tensor<I8> quantized_log(const Tensor<I8>&);
template Tensor<I8> quantized_log2(const Tensor<I8>&);
template Tensor<I8> quantized_log10(const Tensor<I8>&);

} // namespace infer_train
