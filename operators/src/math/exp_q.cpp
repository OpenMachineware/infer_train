#include "infer_train/math/exp_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_exp(const Tensor<I8>&);
template Tensor<I8> quantized_expm1(const Tensor<I8>&);
}
