#include "infer_train/nn/conv2d_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_conv2d(
    const Tensor<I8>&,
    const Tensor<I8>&,
    const Tensor<I8>*,
    int, int, int, int
);
}
