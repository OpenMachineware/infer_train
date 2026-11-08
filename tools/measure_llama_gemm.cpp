// 测量llama.cpp的GEMM绝对时间
// 编译: clang++ -O3 -mcpu=apple-m1 -I../llama.cpp-0.9.0/ggml/src/ggml-cpu -I../llama.cpp-0.9.0/ggml/src -I../llama.cpp-0.9.0/ggml/include measure_llama_gemm.cpp -o measure_llama_gemm

#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#include <arm_neon.h>

#define GGML_F32_EPR 4
#define GGML_F32_VEC float32x4_t
#define GGML_F32_VEC_LOAD(x) vld1q_f32(x)
#define GGML_F32_VEC_STORE(x, v) vst1q_f32(x, v)
#define GGML_F32_VEC_SET1(x) vdupq_n_f32(x)
#define GGML_F32_VEC_FMA(a, b, c) vfmaq_f32(a, b, c)

static constexpr int GEMM_RM = 4;
static constexpr int GEMM_RN = 4;

// llama.cpp的simd_gemm实现（完整复制）
template <int RM, int RN>
static inline void simd_gemm_ukernel(
    float * __restrict__ C,
    const float * __restrict__ A,
    const float * __restrict__ B,
    int K, int N)
{
    static constexpr int KN = GGML_F32_EPR;

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
    // 标量处理tail columns
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

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float * __restrict__ C,
    const float * __restrict__ A,
    const float * __restrict__ B,
    int M, int K, int N)
{
    static constexpr int KN = GGML_F32_EPR;

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

    // Tail rows: one at a time
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
    const int m = 64;
    const int k = 896;
    const int n = 896;

    std::cout << "=== llama.cpp GEMM绝对时间测量 ===\n";
    std::cout << "矩阵尺寸: " << m << "×" << k << "×" << n << "\n\n";

    // 创建测试数据
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C(m * n, 0.0f);

    for (size_t i = 0; i < A.size(); i++) A[i] = sinf(i * 0.001f);
    for (size_t i = 0; i < B.size(); i++) B[i] = cosf(i * 0.001f);

    const int iterations = 100;

    // 测试GEMM
    auto total_time = nanoseconds(0);

    for (int iter = 0; iter < iterations; iter++) {
        C.assign(C.size(), 0.0f);  // Reset

        auto start = high_resolution_clock::now();
        simd_gemm(C.data(), A.data(), B.data(), m, k, n);
        total_time += duration_cast<nanoseconds>(high_resolution_clock::now() - start);
    }

    auto avg_time_ns = total_time.count() / iterations;
    auto avg_time_us = avg_time_ns / 1000.0;

    const long flops = 2L * m * k * n;
    const double gflops = flops * iterations / total_time.count() / 1e9;

    std::cout << "llama.cpp simd_gemm:\n";
    std::cout << "  绝对时间: " << avg_time_ns << " ns (" << avg_time_us << " µs)\n";
    std::cout << "  GFLOPS: " << gflops << "\n";

    std::cout << "\n=== 对比 ===\n";
    std::cout << "我们的绝对时间: 1392008 ns (1392 µs)\n";
    std::cout << "llama.cpp绝对时间: " << avg_time_ns << " ns\n";
    std::cout << "差距: " << 1392008.0 / avg_time_ns << "x\n";

    return 0;
}
