#include "infer_train/nn/batchnorm_q.hpp"

namespace infer_train {

template Tensor<I8> quantized_batchnorm2d_inference(const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, float);
// 注释掉，等声明补上再说
// template Tensor<I8> quantized_batchnorm1d_inference(const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, float);

}
