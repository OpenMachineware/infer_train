#include "infer_train/nn/pool_q.hpp"

namespace infer_train {

template Tensor<I8> quantized_maxpool1d(const Tensor<I8>&, int, int, int);
template Tensor<I8> quantized_maxpool3d(const Tensor<I8>&, int, int, int);
template Tensor<I8> quantized_avgpool1d(const Tensor<I8>&, int, int, int);
template Tensor<I8> quantized_avgpool3d(const Tensor<I8>&, int, int, int);

} // namespace infer_train
