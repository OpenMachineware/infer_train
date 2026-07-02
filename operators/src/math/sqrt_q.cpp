#include "infer_train/math/sqrt_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_sqrt(const Tensor<I8>&);
template Tensor<I8> quantized_rsqrt(const Tensor<I8>&);
}
