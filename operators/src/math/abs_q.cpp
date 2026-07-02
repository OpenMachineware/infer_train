#include "infer_train/math/abs_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_abs(const Tensor<I8>&);
template Tensor<I8> quantized_neg(const Tensor<I8>&);
}
