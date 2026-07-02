#include "infer_train/nn/linear_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_linear(
    const Tensor<I8>&,
    const Tensor<I8>&,
    const Tensor<I8>*
);
}
