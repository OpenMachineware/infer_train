#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace infer_train {

// ============================================================
// 类型标签（用于模板分派）
// ============================================================
struct F32 { using storage = float; static constexpr const char* name = "f32"; };
struct F64 { using storage = double; static constexpr const char* name = "f64"; };
struct F16 { using storage = uint16_t; static constexpr const char* name = "f16"; };
struct BF16 { using storage = uint16_t; static constexpr const char* name = "bf16"; };

// ============================================================
// 量化类型（带 scale + zero_point）
// ============================================================
template<typename T>
struct Quantized {
    using storage = T;
    float scale = 1.0f;
    float zero_point = 0.0f;

    Quantized() = default;
    Quantized(float s, float zp) : scale(s), zero_point(zp) {}
};

using I8 = Quantized<int8_t>;
using Q4 = Quantized<int8_t>;  // 4-bit 打包
using Q8 = Quantized<int8_t>;  // 8-bit

// ============================================================
// 类型转换器（浮点 → float 统一计算）
// ============================================================
template<typename T>
struct DTypeConverter;

// FP32
template<>
struct DTypeConverter<F32> {
    using Storage = float;
    static float to_float(float x) { return x; }
    static float from_float(float x) { return x; }
};

// FP64
template<>
struct DTypeConverter<F64> {
    using Storage = double;
    static float to_float(double x) { return static_cast<float>(x); }
    static double from_float(float x) { return static_cast<double>(x); }
};

// FP16
template<>
struct DTypeConverter<F16> {
    using Storage = uint16_t;
    static float to_float(uint16_t x) {
        uint32_t bits = (uint32_t)x << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        return f;
    }
    static uint16_t from_float(float x) {
        uint32_t bits;
        std::memcpy(&bits, &x, sizeof(float));
        uint16_t fp16 = (bits >> 16) & 0x7FFF;
        fp16 |= (bits >> 16) & 0x8000;
        int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
        exp = std::max(0, std::min(31, exp));
        fp16 = (fp16 & 0x8000) | (exp << 10) | ((bits >> 13) & 0x3FF);
        return fp16;
    }
};

// BF16
template<>
struct DTypeConverter<BF16> {
    using Storage = uint16_t;
    static float to_float(uint16_t x) {
        uint32_t bits = (uint32_t)x << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        return f;
    }
    static uint16_t from_float(float x) {
        uint32_t bits;
        std::memcpy(&bits, &x, sizeof(float));
        return (uint16_t)(bits >> 16);
    }
};

// ============================================================
// 量化转换器
// ============================================================
template<typename T>
struct QuantizedConverter {
    using Storage = typename T::storage;

    static float dequantize(Storage q, float scale, float zero_point) {
        return (static_cast<float>(q) - zero_point) * scale;
    }

    static Storage quantize(float x, float scale, float zero_point) {
        float normalized = x / scale + zero_point;
        int32_t int_val = std::nearbyint(normalized);
        int_val = std::max(-128, std::min(127, int_val));
        return static_cast<Storage>(int_val);
    }
};

// I8 特化
template<>
struct QuantizedConverter<I8> {
    using Storage = int8_t;
    static float dequantize(int8_t q, float scale, float zero_point) {
        return (static_cast<float>(q) - zero_point) * scale;
    }
    static int8_t quantize(float x, float scale, float zero_point) {
        float normalized = x / scale + zero_point;
        int32_t int_val = std::nearbyint(normalized);
        int_val = std::max(-128, std::min(127, int_val));
        return static_cast<int8_t>(int_val);
    }
};

// ============================================================
// 判断类型
// ============================================================
template<typename T>
struct is_quantized {
    static constexpr bool value = false;
};

template<typename T>
struct is_quantized<Quantized<T>> {
    static constexpr bool value = true;
};

} // namespace infer_train
