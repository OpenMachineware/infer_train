#include "infer_train/math/div_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_div(const Tensor<I8>&, const Tensor<I8>&);
}
