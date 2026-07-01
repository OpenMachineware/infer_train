#include "infer_train/math/quantized.hpp"

namespace infer_train {

    // I8
    template Tensor<I8> quantized_add(const Tensor<I8>&, const Tensor<I8>&);
    template Tensor<I8> quantized_matmul(const Tensor<I8>&, const Tensor<I8>&);
    template Tensor<I8> quantized_vec_matmul(const Tensor<I8>&, const Tensor<I8>&);

} // namespace infer_train
