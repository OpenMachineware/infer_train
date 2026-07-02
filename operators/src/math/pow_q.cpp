#include "infer_train/math/pow_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_pow(const Tensor<I8>&, const Tensor<I8>&);
}
