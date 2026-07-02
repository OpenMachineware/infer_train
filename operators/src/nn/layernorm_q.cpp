#include "infer_train/nn/layernorm_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_layernorm(
    const Tensor<I8>&,
    const Tensor<I8>&,
    const Tensor<I8>&,
    float
);
}
