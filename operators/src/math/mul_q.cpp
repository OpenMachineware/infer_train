#include "infer_train/math/mul_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_mul(const Tensor<I8>&, const Tensor<I8>&);
}
