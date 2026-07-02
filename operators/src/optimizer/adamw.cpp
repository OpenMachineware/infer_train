#include "infer_train/optimizer/adamw.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template void adamw_update<F32>(
    std::vector<Tensor<F32>>&,
    std::vector<Tensor<F32>>&,
    AdamWState<F32>&,
    float, float, float, float, float
);

template void adamw_update<F64>(
    std::vector<Tensor<F64>>&,
    std::vector<Tensor<F64>>&,
    AdamWState<F64>&,
    float, float, float, float, float
);

template void adamw_update<F16>(
    std::vector<Tensor<F16>>&,
    std::vector<Tensor<F16>>&,
    AdamWState<F16>&,
    float, float, float, float, float
);

template void adamw_update<BF16>(
    std::vector<Tensor<BF16>>&,
    std::vector<Tensor<BF16>>&,
    AdamWState<BF16>&,
    float, float, float, float, float
);

} // namespace infer_train
