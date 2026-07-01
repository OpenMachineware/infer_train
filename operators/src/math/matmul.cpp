#include "infer_train/math/matmul.hpp"

namespace infer_train {

    // FP32
    template Tensor<F32> matmul(const Tensor<F32>&, const Tensor<F32>&);
    template Tensor<F32> batch_matmul(const Tensor<F32>&, const Tensor<F32>&);
    template Tensor<F32> transpose(const Tensor<F32>&);
    template Tensor<F32> vec_matmul(const Tensor<F32>&, const Tensor<F32>&);

    // FP64
    template Tensor<F64> matmul(const Tensor<F64>&, const Tensor<F64>&);
    template Tensor<F64> batch_matmul(const Tensor<F64>&, const Tensor<F64>&);
    template Tensor<F64> transpose(const Tensor<F64>&);
    template Tensor<F64> vec_matmul(const Tensor<F64>&, const Tensor<F64>&);

    // FP16
    template Tensor<F16> matmul(const Tensor<F16>&, const Tensor<F16>&);
    template Tensor<F16> batch_matmul(const Tensor<F16>&, const Tensor<F16>&);
    template Tensor<F16> transpose(const Tensor<F16>&);
    template Tensor<F16> vec_matmul(const Tensor<F16>&, const Tensor<F16>&);

    // BF16
    template Tensor<BF16> matmul(const Tensor<BF16>&, const Tensor<BF16>&);
    template Tensor<BF16> batch_matmul(const Tensor<BF16>&, const Tensor<BF16>&);
    template Tensor<BF16> transpose(const Tensor<BF16>&);
    template Tensor<BF16> vec_matmul(const Tensor<BF16>&, const Tensor<BF16>&);

} // namespace infer_train
