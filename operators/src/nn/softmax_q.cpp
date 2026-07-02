#include "infer_train/nn/softmax_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_softmax(const Tensor<I8>&, int);
template Tensor<I8> quantized_log_softmax(const Tensor<I8>&, int);
}
