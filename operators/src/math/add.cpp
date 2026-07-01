#include "infer_train/math/add.hpp"

namespace infer_train {

    // FP32
    template Tensor<F32> add(const Tensor<F32>&, const Tensor<F32>&);
    template Tensor<F32> add_scalar(const Tensor<F32>&, float);
    template Tensor<F32> add_n(const std::vector<const Tensor<F32>*>&);

    // FP64
    template Tensor<F64> add(const Tensor<F64>&, const Tensor<F64>&);
    template Tensor<F64> add_scalar(const Tensor<F64>&, double);
    template Tensor<F64> add_n(const std::vector<const Tensor<F64>*>&);

    // FP16
    template Tensor<F16> add(const Tensor<F16>&, const Tensor<F16>&);
    template Tensor<F16> add_scalar(const Tensor<F16>&, uint16_t);
    template Tensor<F16> add_n(const std::vector<const Tensor<F16>*>&);

    // BF16
    template Tensor<BF16> add(const Tensor<BF16>&, const Tensor<BF16>&);
    template Tensor<BF16> add_scalar(const Tensor<BF16>&, uint16_t);
    template Tensor<BF16> add_n(const std::vector<const Tensor<BF16>*>&);

} // namespace infer_train
