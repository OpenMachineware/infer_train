#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>

namespace infer_train {

// ============================================================
// reshape: 改变张量形状（不改变数据）
// ============================================================
template<typename T>
Tensor<T> reshape(const Tensor<T>& input, const std::vector<size_t>& new_shape) {
    // 计算新形状的总大小
    size_t new_size = 1;
    for (auto d : new_shape) new_size *= d;

    // 检查大小是否匹配
    if (new_size != input.size()) {
        return Tensor<T>();  // 大小不匹配，返回空
    }

    Tensor<T> output(new_shape);

    // 复制数据（顺序不变）
    for (size_t i = 0; i < input.size(); ++i) {
        output.data[i] = input.data[i];
    }

    return output;
}

} // namespace infer_train
