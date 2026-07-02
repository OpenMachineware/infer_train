#include "infer_train/optimizer/sgd.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template void sgd_update<F32>(
    std::vector<Tensor<F32>>&,
    std::vector<Tensor<F32>>&,
    float, float, float, bool
);

template void sgd_update<F64>(
    std::vector<Tensor<F64>>&,
    std::vector<Tensor<F64>>&,
    float, float, float, bool
);

template void sgd_update<F16>(
    std::vector<Tensor<F16>>&,
    std::vector<Tensor<F16>>&,
    float, float, float, bool
);

template void sgd_update<BF16>(
    std::vector<Tensor<BF16>>&,
    std::vector<Tensor<BF16>>&,
    float, float, float, bool
);

} // namespace infer_train
