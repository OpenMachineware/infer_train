// 测量llama.cpp flash attention每个步骤的绝对时间
// 编译: clang++ -O3 -mcpu=apple-m1 -I../llama.cpp-0.9.0/ggml/src/ggml-cpu -I../llama.cpp-0.9.0/ggml/src -I../llama.cpp-0.9.0/ggml/include measure_llama_attention.cpp -o measure_llama_attention

#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

// Minimal definitions from llama.cpp
#define GGML_F32_EPR 4
#define GGML_F32_VEC float32x4_t
#define GGML_F32_VEC_LOAD(x) vld1q_f32(x)
#define GGML_F32_VEC_STORE(x, v) vst1q_f32(x, v)
#define GGML_F32_VEC_SET1(x) vdupq_n_f32(x)
#define GGML_F32_VEC_FMA(a, b, c) vfmaq_f32(a, b, c)

#include <arm_neon.h>

const int Q_TILE_SZ = 64;
const int KV_TILE_SZ = 64;

// SIMD GEMM from llama.cpp
template <int RM, int RN>
static inline void simd_gemm_ukernel(
    float * __restrict__ C,
    const float * __restrict__ A,
    const float * __restrict__ B,
    int K, int N)
{
    GGML_F32_VEC acc[RM][RN];
    for (int i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * GGML_F32_EPR);
        }
    }

    for (int kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        for (int r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * GGML_F32_EPR);
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
            GGML_F32_VEC_STORE(C + i * N + r * GGML_F32_EPR, acc[i][r]);
        }
    }
}

void simd_gemm(float * C, const float * A, const float * B, int M, int K, int N) {
    // Main tile loop
    int ii = 0;
    for (; ii < M; ii += 4) {
        if (ii + 4 <= M) {
            int jj = 0;
            for (; jj < N; jj += 4) {
                if (jj + 4 <= N) {
                    simd_gemm_ukernel<4, 4>(C + ii * N + jj, A + ii * K, B + jj, K, N);
                }
            }
        }
    }

    // Handle tail rows
    for (; ii < M; ii++) {
        for (int jj = 0; jj < N; jj++) {
            float sum = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                sum += A[ii * K + kk] * B[kk * N + jj];
            }
            C[ii * N + jj] = sum;
        }
    }
}

using namespace std::chrono;

int main() {
    // Qwen3-0.6B config
    const int n_tokens = 64;
    const int n_heads = 16;
    const int n_kv_heads = 8;
    const int head_dim = 128;
    const int n_layers = 28;

    const int kv_size = n_kv_heads * head_dim;
    const int q_size = n_heads * head_dim;
    const float scale = 1.0f / sqrtf(head_dim);

    std::cout << "=== llama.cpp Flash Attention 绝对时间测量 ===\n";
    std::cout << "配置: " << n_tokens << " tokens, " << n_heads << " heads, "
              << n_kv_heads << " kv_heads, head_dim " << head_dim << "\n";
    std::cout << "Tile sizes: Q=" << Q_TILE_SZ << ", KV=" << KV_TILE_SZ << "\n\n";

    // Create test data
    std::vector<float> q(q_size * n_tokens);
    std::vector<float> k(kv_size * n_tokens);
    std::vector<float> v(kv_size * n_tokens);

    for (size_t i = 0; i < q.size(); i++) q[i] = sinf(i * 0.001f);
    for (size_t i = 0; i < k.size(); i++) k[i] = cosf(i * 0.001f);
    for (size_t i = 0; i < v.size(); i++) v[i] = sinf(i * 0.002f);

    // Scratch space
    std::vector<float> Q_tile(Q_TILE_SZ * head_dim, 0.0f);
    std::vector<float> K_tile(head_dim * KV_TILE_SZ, 0.0f);  // Transposed
    std::vector<float> V_tile(KV_TILE_SZ * head_dim, 0.0f);
    std::vector<float> KQ(Q_TILE_SZ * KV_TILE_SZ, 0.0f);
    std::vector<float> VKQ(Q_TILE_SZ * head_dim, 0.0f);

    const int h = 0;
    const int kv_head = 0;
    const int tile_rows = std::min(n_tokens, Q_TILE_SZ);
    const int iterations = 1000;

    std::cout << "测量单个head的注意力计算（" << iterations << "次迭代）:\n\n";

    auto time_load_q = nanoseconds(0);
    auto time_load_k = nanoseconds(0);
    auto time_load_v = nanoseconds(0);
    auto time_gemm_qk = nanoseconds(0);
    auto time_softmax = nanoseconds(0);
    auto time_gemm_vkq = nanoseconds(0);

    for (int iter = 0; iter < iterations; iter++) {
        const int max_kv_pos = n_tokens;

        for (int kv_tile_start = 0; kv_tile_start < max_kv_pos; kv_tile_start += KV_TILE_SZ) {
            const int kv_tile_end = std::min(kv_tile_start + KV_TILE_SZ, max_kv_pos);
            const int kv_tile_size = kv_tile_end - kv_tile_start;

            // Load Q tile
            auto start = high_resolution_clock::now();
            for (int tq = 0; tq < tile_rows; tq++) {
                const int q_offset = tq * q_size + h * head_dim;
                memcpy(&Q_tile[tq * head_dim], &q[q_offset], head_dim * sizeof(float));
            }
            time_load_q += duration_cast<nanoseconds>(high_resolution_clock::now() - start);

            // Load K tile transposed (exactly as in llama.cpp ops.cpp lines 9258-9271)
            auto start_k = high_resolution_clock::now();
            for (int tk = 0; tk < kv_tile_size; tk++) {
                const int kv_pos = kv_tile_start + tk;
                const int k_offset = kv_pos * kv_size + kv_head * head_dim;
                for (int dk = 0; dk < head_dim; dk++) {
                    K_tile[dk * KV_TILE_SZ + tk] = k[k_offset + dk];
                }
            }
            time_load_k += duration_cast<nanoseconds>(high_resolution_clock::now() - start_k);

            // Load V tile
            auto start_v = high_resolution_clock::now();
            for (int tk = 0; tk < kv_tile_size; tk++) {
                const int kv_pos = kv_tile_start + tk;
                const int v_offset = kv_pos * kv_size + kv_head * head_dim;
                memcpy(&V_tile[tk * head_dim], &v[v_offset], head_dim * sizeof(float));
            }
            time_load_v += duration_cast<nanoseconds>(high_resolution_clock::now() - start_v);

            // GEMM: KQ = Q @ K^T
            auto start_gemm = high_resolution_clock::now();
            memset(&KQ[0], 0, tile_rows * kv_tile_size * sizeof(float));
            simd_gemm(&KQ[0], &Q_tile[0], &K_tile[0], tile_rows, head_dim, kv_tile_size);
            time_gemm_qk += duration_cast<nanoseconds>(high_resolution_clock::now() - start_gemm);

            // Softmax
            auto start_softmax = high_resolution_clock::now();
            for (int i = 0; i < tile_rows; i++) {
                const int q_pos = i;
                for (int j = 0; j < kv_tile_size; j++) {
                    const int kv_pos = kv_tile_start + j;
                    if (kv_pos <= q_pos) {
                        KQ[i * KV_TILE_SZ + j] *= scale;
                    } else {
                        KQ[i * KV_TILE_SZ + j] = -INFINITY;
                    }
                }

                // Find max
                float max_val = -INFINITY;
                for (int j = 0; j < kv_tile_size; j++) {
                    if (KQ[i * KV_TILE_SZ + j] > max_val && KQ[i * KV_TILE_SZ + j] != -INFINITY) {
                        max_val = KQ[i * KV_TILE_SZ + j];
                    }
                }

                // Exp and sum
                float sum = 0.0f;
                for (int j = 0; j < kv_tile_size; j++) {
                    if (KQ[i * KV_TILE_SZ + j] != -INFINITY) {
                        KQ[i * KV_TILE_SZ + j] = expf(KQ[i * KV_TILE_SZ + j] - max_val);
                        sum += KQ[i * KV_TILE_SZ + j];
                    } else {
                        KQ[i * KV_TILE_SZ + j] = 0.0f;
                    }
                }

                // Normalize
                if (sum > 0.0f) {
                    for (int j = 0; j < kv_tile_size; j++) {
                        KQ[i * KV_TILE_SZ + j] /= sum;
                    }
                }
            }
            time_softmax += duration_cast<nanoseconds>(high_resolution_clock::now() - start_softmax);

            // GEMM: VKQ = KQ @ V
            auto start_vkq = high_resolution_clock::now();
            simd_gemm(&VKQ[0], &KQ[0], &V_tile[0], tile_rows, kv_tile_size, head_dim);
            time_gemm_vkq += duration_cast<nanoseconds>(high_resolution_clock::now() - start_vkq);
        }
    }

    const int kv_tiles = (n_tokens + KV_TILE_SZ - 1) / KV_TILE_SZ;

    std::cout << "=== 单个head单次迭代绝对时间 ===\n";
    std::cout << "Load Q tile:   " << time_load_q.count() / iterations << " ns (平均 "
              << time_load_q.count() / iterations / kv_tiles << " ns)\n";
    std::cout << "Load K tile:   " << time_load_k.count() / iterations << " ns (平均 "
              << time_load_k.count() / iterations / kv_tiles << " ns)\n";
    std::cout << "Load V tile:   " << time_load_v.count() / iterations << " ns (平均 "
              << time_load_v.count() / iterations / kv_tiles << " ns)\n";
    std::cout << "GEMM Q·K:      " << time_gemm_qk.count() / iterations << " ns (平均 "
              << time_gemm_qk.count() / iterations / kv_tiles << " ns)\n";
    std::cout << "Softmax:       " << time_softmax.count() / iterations << " ns (平均 "
              << time_softmax.count() / iterations / kv_tiles << " ns)\n";
    std::cout << "GEMM VKQ:      " << time_gemm_vkq.count() / iterations << " ns (平均 "
              << time_gemm_vkq.count() / iterations / kv_tiles << " ns)\n";

    const auto total_per_iter = time_load_q + time_load_k + time_load_v +
                                 time_gemm_qk + time_softmax + time_gemm_vkq;
    std::cout << "\n单个head单次迭代总时间: " << total_per_iter.count() / iterations << " ns\n";

    std::cout << "\n=== 对比分析 ===\n";
    std::cout << "步骤            | 我们 (ns)  | llama.cpp (ns) | 差距\n";
    std::cout << "Load Q tile     |     953.7  |       " << time_load_q.count() / iterations
              << " | " << 953.7 * iterations / time_load_q.count() << "x\n";
    std::cout << "Load K tile     |    3126.8  |       " << time_load_k.count() / iterations
              << " | " << 3126.8 * iterations / time_load_k.count() << "x\n";
    std::cout << "Load V tile     |     800.3  |       " << time_load_v.count() / iterations
              << " | " << 800.3 * iterations / time_load_v.count() << "x\n";
    std::cout << "GEMM Q·K        |   11661.8  |       " << time_gemm_qk.count() / iterations
              << " | " << 11661.8 * iterations / time_gemm_qk.count() << "x\n";
    std::cout << "Softmax         |   13905.2  |       " << time_softmax.count() / iterations
              << " | " << 13905.2 * iterations / time_softmax.count() << "x\n";
    std::cout << "GEMM VKQ        |   11331.2  |       " << time_gemm_vkq.count() / iterations
              << " | " << 11331.2 * iterations / time_gemm_vkq.count() << "x\n";

    return 0;
}
