#include "infer_train/math/sub_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_sub(const Tensor<I8>&, const Tensor<I8>&);
}
