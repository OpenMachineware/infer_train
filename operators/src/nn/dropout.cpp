#include "infer_train/nn/dropout.hpp"

namespace infer_train {

// 推理模式
template Tensor<F32> dropout_inference(const Tensor<F32>&, float);
template Tensor<F64> dropout_inference(const Tensor<F64>&, float);
template Tensor<F16> dropout_inference(const Tensor<F16>&, float);
template Tensor<BF16> dropout_inference(const Tensor<BF16>&, float);

// 训练模式
template Tensor<F32> dropout_training(const Tensor<F32>&, float, uint32_t);
template Tensor<F64> dropout_training(const Tensor<F64>&, float, uint32_t);
template Tensor<F16> dropout_training(const Tensor<F16>&, float, uint32_t);
template Tensor<BF16> dropout_training(const Tensor<BF16>&, float, uint32_t);

} // namespace infer_train
