#include "infer_train/optimizer/adam.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template void adam_update<F32>(
    std::vector<Tensor<F32>>&,
    std::vector<Tensor<F32>>&,
    AdamState<F32>&,
    float, float, float, float, float
);

template void adam_update<F64>(
    std::vector<Tensor<F64>>&,
    std::vector<Tensor<F64>>&,
    AdamState<F64>&,
    float, float, float, float, float
);

template void adam_update<F16>(
    std::vector<Tensor<F16>>&,
    std::vector<Tensor<F16>>&,
    AdamState<F16>&,
    float, float, float, float, float
);

template void adam_update<BF16>(
    std::vector<Tensor<BF16>>&,
    std::vector<Tensor<BF16>>&,
    AdamState<BF16>&,
    float, float, float, float, float
);

} // namespace infer_train
