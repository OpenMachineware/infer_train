#include "infer_train/nn/pooling_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_maxpool2d(const Tensor<I8>&, int, int, int);
template Tensor<I8> quantized_avgpool2d(const Tensor<I8>&, int, int, int);
}
