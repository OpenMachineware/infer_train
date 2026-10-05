#!/bin/bash
# 为所有 IQ 格式生成 dump 程序

formats=(
    "iq2_xs:17"
    "iq2_s:18"
    "iq3_xxs:19"
    "iq3_s:20"
    "iq1_s:21"
    "iq1_m:22"
)

for fmt in "${formats[@]}"; do
    IFS=':' read -r name type <<< "$fmt"
    cat > "${name}_llama_dump.c" << EOFILE
// 直接调用 llama.cpp 的 quantize_${name} 函数
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K/8];
} block_${name};

// llama.cpp 函数声明
void ggml_quantize_init(int type);
size_t quantize_${name}(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

#define GGML_TYPE_${name^^} ${type}

int main() {
    printf("=== 直接调用 llama.cpp ${name^^} 量化 ===\\n\\n");
    
    // 初始化量化表
    ggml_quantize_init(GGML_TYPE_${name^^});
    
    // 测试数据（一行，256个元素）
    float test_data[QK_K] = {
        0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812, 0.934245, 0.814190,
        -0.321567, 0.456789, -0.234567, 0.876543, -0.543210, 0.123456, -0.789012, 0.345678,
        0.567890, -0.234567, 0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
        0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678, 0.567890, -0.234567,
    };
    
    // 量化权重（全1）
    float quant_weights[QK_K];
    for (int i = 0; i < QK_K; i++) quant_weights[i] = 1.0f;
    
    // 量化
    block_${name} block;
    size_t result = quantize_${name}(test_data, &block, 1, QK_K, quant_weights);
    
    // 输出结果
    printf("Result size: %zu\\n", result);
    printf("Block:\\n");
    printf("  d = 0x%04x\\n", block.d);
    printf("  qs[0:8] = ");
    for (int i = 0; i < 8; i++) printf("0x%04x ", block.qs[i]);
    printf("\\n");
    
    return 0;
}
EOFILE
done

echo "生成完成"
