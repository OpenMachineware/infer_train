// Test llama.cpp flash attention for all formats
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <cstdio>
#include <vector>
#include <cstring>

extern "C" {
struct FlashAttnResult {
    double time_us;
    int status;
    char error_msg[256];
};

FlashAttnResult run_llama_flash_attn(
    const char * format,
    const float * q_data,
    const void * k_data,
    const void * v_data,
    float * out_data,
    int num_heads,
    int head_dim,
    int seq_k,
    int rounds
);
}

int main() {
    const int num_heads = 32;
    const int head_dim = 128;
    const int rounds = 200;

    printf("=== llama.cpp Flash Attention Decode - All Formats ===\n");
    printf("Config: %d heads, %d head_dim, %d rounds\n\n", num_heads, head_dim, rounds);

    const char* formats[] = {"f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"};
    int seq_ks[] = {256, 512, 1024, 2048, 4096};

    // Print header
    printf("%-8s", "seq_k");
    for (const char* fmt : formats) {
        printf(" %10s", fmt);
    }
    printf("\n");
    printf("%s\n", std::string(8 + 11*7, '-').c_str());

    for (int seq_k : seq_ks) {
        printf("%-8d", seq_k);
        fflush(stdout);

        for (const char* fmt : formats) {
            // Create test data
            std::vector<float> q_data(num_heads * head_dim, 1.0f);
            // For quantized formats, we need larger K/V buffers
            // F16/BF16: 2 bytes per element
            // Q4_0: 18 bytes per 32 elements = 0.5625 bytes per element
            // Q4_1: 20 bytes per 32 elements = 0.625 bytes per element
            // Q5_0: 22 bytes per 32 elements = 0.6875 bytes per element
            // Q5_1: 24 bytes per 32 elements = 0.75 bytes per element
            // Q8_0: 34 bytes per 32 elements = 1.0625 bytes per element

            size_t kv_bytes;
            if (strcmp(fmt, "f16") == 0 || strcmp(fmt, "bf16") == 0) {
                kv_bytes = seq_k * head_dim * 2;
            } else if (strcmp(fmt, "q4_0") == 0) {
                kv_bytes = seq_k * (head_dim / 32) * 18;
            } else if (strcmp(fmt, "q4_1") == 0) {
                kv_bytes = seq_k * (head_dim / 32) * 20;
            } else if (strcmp(fmt, "q5_0") == 0) {
                kv_bytes = seq_k * (head_dim / 32) * 22;
            } else if (strcmp(fmt, "q5_1") == 0) {
                kv_bytes = seq_k * (head_dim / 32) * 24;
            } else if (strcmp(fmt, "q8_0") == 0) {
                kv_bytes = seq_k * (head_dim / 32) * 34;
            } else {
                kv_bytes = seq_k * head_dim * 2;
            }

            std::vector<uint8_t> k_data(kv_bytes, 0);
            std::vector<uint8_t> v_data(kv_bytes, 0);
            std::vector<float> out_data(num_heads * head_dim, 0.0f);

            auto result = run_llama_flash_attn(
                fmt,
                q_data.data(),
                k_data.data(),
                v_data.data(),
                out_data.data(),
                num_heads,
                head_dim,
                seq_k,
                rounds
            );

            if (result.status != 0) {
                printf(" %10s", "ERROR");
            } else {
                double gflops = 4.0 * seq_k * head_dim * num_heads / result.time_us / 1000.0;
                printf(" %10.2f", gflops);
            }
            fflush(stdout);
        }
        printf("\n");
    }

    return 0;
}
