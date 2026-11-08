// 测量llama.cpp的FFN绝对时间
// FFN: gate + up + down 三个GEMM

#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

#include <arm_neon.h>

#define GGML_F32_EPR 4
#define GGML_F32_VEC float32x4_t
#define GGML_F32_VEC_LOAD(x) vld1q_f32(x)
#define GGML_F32_VEC_STORE(x, v) vst1q_f32(x, v)
#define GGML_F32_VEC_SET1(x) vdupq_n_f32(x)
#define GGML_F32_VEC_FMA(a, b, c) vfmaq_f32(a, b, c)

static constexpr int GEMM_RM = 4;
static constexpr int GEMM_RN = 4;

template <int RM, int RN>
static inline void simd_gemm_ukernel(
    float * __restrict__ C,
    const float * __restrict__ A,
    const float * __restrict__ B,
    int K, int N)
{
    const int KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    for (int i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * KN);
        }
    }

    for (int kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        for (int r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * KN);
        }
        for (int i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            for (int r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    for (int i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

template <int RM>
static inline void simd_gemm_ukernel_tail(
    float * __restrict__ C,
    const float * __restrict__ A,
    const float * __restrict__ B,
    int K, int N, int cols)
{
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < RM; i++) {
            float a = C[i * N + j];
            for (int kk = 0; kk < K; kk++) {
                a += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = a;
        }
    }
}

static void simd_gemm(
    float * __restrict__ C,
    const float * __restrict__ A,
    const float * __restrict__ B,
    int M, int K, int N)
{
    const int KN = GGML_F32_EPR;

    int ii = 0;
    for (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        if (jj < N) {
            simd_gemm_ukernel_tail<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    for (; ii < M; ii++) {
        int jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        if (jj < N) {
            simd_gemm_ukernel_tail<1>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += K;
        C += N;
    }
}

using namespace std::chrono;

int main() {
    const int n_tokens = 64;
    const int hidden_size = 896;
    const int intermediate_size = 4864;

    std::cout << "=== llama.cpp FFN 绝对时间测量 ===\n";
    std::cout << "配置: " << n_tokens << " tokens, hidden=" << hidden_size
              << ", intermediate=" << intermediate_size << "\n\n";

    // 创建测试数据
    std::vector<float> input(hidden_size * n_tokens);
    std::vector<float> gate_weight(hidden_size * intermediate_size);
    std::vector<float> up_weight(hidden_size * intermediate_size);
    std::vector<float> down_weight(intermediate_size * hidden_size);

    for (size_t i = 0; i < input.size(); i++) input[i] = sinf(i * 0.001f);
    for (size_t i = 0; i < gate_weight.size(); i++) gate_weight[i] = cosf(i * 0.001f);
    for (size_t i = 0; i < up_weight.size(); i++) up_weight[i] = sinf(i * 0.002f);
    for (size_t i = 0; i < down_weight.size(); i++) down_weight[i] = cosf(i * 0.001f);

    std::vector<float> gate_out(intermediate_size * n_tokens);
    std::vector<float> up_out(intermediate_size * n_tokens);
    std::vector<float> ffn_out(hidden_size * n_tokens);

    const int iterations = 10;

    // 测量每个GEMM
    std::cout << "=== GEMM分解 ===\n";

    auto time_gate = nanoseconds(0);
    for (int iter = 0; iter < iterations; iter++) {
        auto start = high_resolution_clock::now();
        simd_gemm(gate_out.data(), input.data(), gate_weight.data(),
                  n_tokens, hidden_size, intermediate_size);
        time_gate += duration_cast<nanoseconds>(high_resolution_clock::now() - start);
    }
    std::cout << "Gate GEMM (" << n_tokens << "×" << hidden_size << "×" << intermediate_size << "):\n";
    std::cout << "  绝对时间: " << time_gate.count() / iterations << " ns\n";

    auto time_up = nanoseconds(0);
    for (int iter = 0; iter < iterations; iter++) {
        auto start = high_resolution_clock::now();
        simd_gemm(up_out.data(), input.data(), up_weight.data(),
                  n_tokens, hidden_size, intermediate_size);
        time_up += duration_cast<nanoseconds>(high_resolution_clock::now() - start);
    }
    std::cout << "Up GEMM (" << n_tokens << "×" << hidden_size << "×" << intermediate_size << "):\n";
    std::cout << "  绝对时间: " << time_up.count() / iterations << " ns\n";

    // Element-wise multiply: gate * up
    for (size_t i = 0; i < gate_out.size(); i++) {
        gate_out[i] *= up_out[i];
    }

    auto time_down = nanoseconds(0);
    for (int iter = 0; iter < iterations; iter++) {
        auto start = high_resolution_clock::now();
        simd_gemm(ffn_out.data(), gate_out.data(), down_weight.data(),
                  n_tokens, intermediate_size, hidden_size);
        time_down += duration_cast<nanoseconds>(high_resolution_clock::now() - start);
    }
    std::cout << "Down GEMM (" << n_tokens << "×" << intermediate_size << "×" << hidden_size << "):\n";
    std::cout << "  绝对时间: " << time_down.count() / iterations << " ns\n";

    auto total_ns = (time_gate + time_up + time_down).count() / iterations;
    std::cout << "\n=== 总时间 ===\n";
    std::cout << "llama.cpp FFN: " << total_ns << " ns (" << total_ns / 1000.0 << " µs)\n";

    std::cout << "\n=== 对比 ===\n";
    std::cout << "我们FFN: 51070700 ns (51071 µs)\n";
    std::cout << "llama.cpp FFN: " << total_ns << " ns\n";
    std::cout << "差距: " << 51070700.0 / total_ns << "x\n";

    return 0;
}
