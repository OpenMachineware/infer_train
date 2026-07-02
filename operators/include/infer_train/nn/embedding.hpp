#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>

namespace infer_train {

template<typename T>
Tensor<T> embedding(
    const std::vector<int64_t>& indices,
    const Tensor<T>& weight,
    int padding_idx = -1
) {
    if (weight.shape.size() != 2) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;

    size_t vocab_size = weight.shape[0];
    size_t embed_dim = weight.shape[1];

    std::vector<size_t> output_shape;
    output_shape.push_back(indices.size());
    output_shape.push_back(embed_dim);

    Tensor<T> output(output_shape);

    for (size_t i = 0; i < indices.size(); ++i) {
        int64_t idx = indices[i];

        if (padding_idx >= 0 && idx == padding_idx) {
            for (size_t e = 0; e < embed_dim; ++e) {
                output.data[i * embed_dim + e] = Conv::from_float(0.0f);
            }
            continue;
        }

        if (idx < 0 || idx >= (int64_t)vocab_size) {
            return Tensor<T>();
        }

        for (size_t e = 0; e < embed_dim; ++e) {
            output.data[i * embed_dim + e] = weight.data[idx * embed_dim + e];
        }
    }

    return output;
}

} // namespace infer_train
