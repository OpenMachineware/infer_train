// Test llama.cpp flash attention
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <cstdio>
#include <vector>

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

    printf("=== llama.cpp Flash Attention Decode ===\n");
    printf("Config: %d heads, %d head_dim, %d rounds\n\n", num_heads, head_dim, rounds);

    int seq_ks[] = {256, 512, 1024, 2048, 4096};

    printf("%-8s %12s %10s\n", "seq_k", "Time (µs)", "GFLOPS");
    printf("%s\n", std::string(34, '-').c_str());

    for (int seq_k : seq_ks) {
        // Create F16 test data
        std::vector<float> q_data(num_heads * head_dim, 1.0f);
        std::vector<uint16_t> k_data(seq_k * 1 * head_dim, 0x3C00);
        std::vector<uint16_t> v_data(seq_k * 1 * head_dim, 0x3C00);
        std::vector<float> out_data(num_heads * head_dim, 0.0f);

        auto result = run_llama_flash_attn(
            "f16",
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
            printf("%-8d ERROR: %s\n", seq_k, result.error_msg);
        } else {
            double gflops = 4.0 * seq_k * head_dim * num_heads / result.time_us / 1000.0;
            printf("%-8d %12.2f %10.2f\n", seq_k, result.time_us, gflops);
        }
        fflush(stdout);
    }

    return 0;
}
