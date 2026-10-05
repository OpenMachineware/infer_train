// Direct call to llama.cpp quantization with dump comparison
// This writes test data and calls llama.cpp's quantize function

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;

// External function from llama.cpp
// We'll link against ggml library
void ggml_quantize_init(int type);
void quantize_row_iq2_xxs_impl(const float * x, void * vy, int64_t n, const float * quant_weights);

int main() {
    printf("=== llama.cpp Direct Quantization Test ===\n\n");
    
    // Initialize quantization tables
    ggml_quantize_init(13);  // GGML_TYPE_IQ2_XXS = 13
    
    // Test data
    float test_data[QK_K] = {
        0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812, 0.934245, 0.814190,
        -0.321567, 0.456789, -0.234567, 0.876543, -0.543210, 0.123456, -0.789012, 0.345678,
        0.567890, -0.234567, 0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
        0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678, 0.567890, -0.234567,
    };
    // Fill rest with zeros
    for (int i = 32; i < QK_K; i++) test_data[i] = 0.0f;
    
    // Create uniform quant weights (all 1.0)
    float quant_weights[QK_K];
    for (int i = 0; i < QK_K; i++) quant_weights[i] = 1.0f;
    
    // Quantize
    block_iq2_xxs block;
    quantize_row_iq2_xxs_impl(test_data, &block, QK_K, quant_weights);
    
    // Output results
    printf("Block structure:\n");
    printf("  d = 0x%04x\n", block.d);
    printf("  qs[0:8] = ");
    for (int i = 0; i < 8; i++) printf("0x%04x ", block.qs[i]);
    printf("\n");
    
    // Decode qs to u32
    uint32_t q2[16];
    for (int i = 0; i < 16; i++) {
        q2[i] = block.qs[2*i] | (block.qs[2*i+1] << 16);
    }
    
    printf("\n  q2 as u32:\n");
    for (int i = 0; i < 8; i++) {
        printf("    q2[%d] = 0x%08x\n", i, q2[i]);
    }
    
    // Write results to file for Rust to compare
    FILE *fp = fopen("/tmp/iq2_xxs_llama_output.txt", "w");
    if (fp) {
        fprintf(fp, "d=0x%04x\n", block.d);
        for (int i = 0; i < 8; i++) {
            fprintf(fp, "q2[%d]=0x%08x\n", i, q2[i]);
        }
        fclose(fp);
        printf("\nOutput written to /tmp/iq2_xxs_llama_output.txt\n");
    }
    
    return 0;
}
