#include "infer_train/nn/quantized.hpp"

namespace infer_train {

    // ============================================================
    // I8 量化激活函数实例化
    // ============================================================
    template Tensor<I8> quantized_relu(const Tensor<I8>&);
    template Tensor<I8> quantized_leaky_relu(const Tensor<I8>&, float);
    template Tensor<I8> quantized_elu(const Tensor<I8>&, float);
    template Tensor<I8> quantized_gelu(const Tensor<I8>&);
    template Tensor<I8> quantized_softmax(const Tensor<I8>&, int);
    template Tensor<I8> quantized_log_softmax(const Tensor<I8>&, int);

} // namespace infer_train
