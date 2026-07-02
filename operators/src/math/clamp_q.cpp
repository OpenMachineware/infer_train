#include "infer_train/math/clamp_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_clamp(const Tensor<I8>&, int8_t, int8_t);
}
