#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include "infer_train/dtype.hpp"

namespace infer_train {

    // ============================================================
    // Tensor 模板（浮点 + 量化统一）
    // ============================================================
    template<typename T>
    struct Tensor {
        using DType = T;
        using Storage = typename T::storage;

        std::vector<Storage> data;
        std::vector<size_t> shape;

        // 量化参数（仅量化类型使用）
        float scale = 1.0f;
        float zero_point = 0.0f;

        Tensor() = default;

        explicit Tensor(const std::vector<size_t>& s) : shape(s) {
            size_t n = 1;
            for (auto d : s) n *= d;
            data.resize(n);
        }

        Tensor(const Storage* d, const std::vector<size_t>& s) : shape(s) {
            size_t n = 1;
            for (auto d : s) n *= d;
            data.assign(d, d + n);
        }

        size_t size() const {
            size_t n = 1;
            for (auto d : shape) n *= d;
            return n;
        }

        bool empty() const {
            return data.empty() || shape.empty();
        }

        void reshape(const std::vector<size_t>& s) {
            size_t n = 1;
            for (auto d : s) n *= d;
            if (n != data.size()) {
                throw std::runtime_error("reshape size mismatch");
            }
            shape = s;
        }

        Storage* ptr() { return data.data(); }
        const Storage* ptr() const { return data.data(); }

        bool is_quantized() const {
            return ::infer_train::is_quantized<T>::value;
        }
    };

} // namespace infer_train
