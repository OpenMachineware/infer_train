#include "infer_train/math/add_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_add(const Tensor<I8>&, const Tensor<I8>&);
}
