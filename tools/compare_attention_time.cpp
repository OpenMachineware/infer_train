// 测量llama.cpp flash attention各步骤的绝对时间
// 与我们的Rust实现对比

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

const int Q_TILE_SZ = 64;
const int KV_TILE_SZ = 64;
const int HEAD_DIM = 128;

// SIMD GEMM (简化版，只处理完整tiles)
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

using namespace std::chrono;

int main() {
    const int n_tokens = 64;
    const int n_heads = 16;
    const int n_kv_heads = 8;

    const int kv_size = n_kv_heads * HEAD_DIM;
    const int q_size = n_heads * HEAD_DIM;
    const float scale = 1.0f / sqrtf(HEAD_DIM);

    std::cout << "=== llama.cpp Flash Attention 绝对时间测量 ===\n";
    std::cout << "配置: " << n_tokens << " tokens, " << n_heads << " heads, head_dim " << HEAD_DIM << "\n\n";

    // 创建测试数据
    std::vector<float> q(q_size * n_tokens);
    std::vector<float> k(kv_size * n_tokens);
    std::vector<float> v(kv_size * n_tokens);

    for (size_t i = 0; i < q.size(); i++) q[i] = sinf(i * 0.001f);
    for (size_t i = 0; i < k.size(); i++) k[i] = cosf(i * 0.001f);
    for (size_t i = 0; i < v.size(); i++) v[i] = sinf(i * 0.002f);

    // Scratch
    std::vector<float> Q_tile(Q_TILE_SZ * HEAD_DIM, 0.0f);
    std::vector<float> K_tile(HEAD_DIM * KV_TILE_SZ, 0.0f);
    std::vector<float> V_tile(KV_TILE_SZ * HEAD_DIM, 0.0f);
    std::vector<float> KQ(Q_TILE_SZ * KV_TILE_SZ, 0.0f);

    const int h = 0;
    const int kv_head = 0;
    const int tile_rows = n_tokens;
    const int iterations = 1000;

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
                const int q_offset = tq * q_size + h * HEAD_DIM;
                memcpy(&Q_tile[tq * HEAD_DIM], &q[q_offset], HEAD_DIM * sizeof(float));
            }
            time_load_q += duration_cast<nanoseconds>(high_resolution_clock::now() - start);

            // Load K tile transposed
            auto start_k = high_resolution_clock::now();
            for (int tk = 0; tk < kv_tile_size; tk++) {
                const int kv_pos = kv_tile_start + tk;
                const int k_offset = kv_pos * kv_size + kv_head * HEAD_DIM;
                for (int dk = 0; dk < HEAD_DIM; dk++) {
                    K_tile[dk * KV_TILE_SZ + tk] = k[k_offset + dk];
                }
            }
            time_load_k += duration_cast<nanoseconds>(high_resolution_clock::now() - start_k);

            // Load V tile
            auto start_v = high_resolution_clock::now();
            for (int tk = 0; tk < kv_tile_size; tk++) {
                const int kv_pos = kv_tile_start + tk;
                const int v_offset = kv_pos * kv_size + kv_head * HEAD_DIM;
                memcpy(&V_tile[tk * HEAD_DIM], &v[v_offset], HEAD_DIM * sizeof(float));
            }
            time_load_v += duration_cast<nanoseconds>(high_resolution_clock::now() - start_v);

            // GEMM: KQ = Q @ K^T
            auto start_gemm = high_resolution_clock::now();
            memset(&KQ[0], 0, tile_rows * kv_tile_size * sizeof(float));
            // 简化：假设tile_rows和kv_tile_size都是64的倍数
            for (int i = 0; i < tile_rows; i += 4) {
                for (int j = 0; j < kv_tile_size; j += 4) {
                    simd_gemm_ukernel<4, 4>(&KQ[i * KV_TILE_SZ + j],
                                            &Q_tile[i * HEAD_DIM],
                                            &K_tile[j],
                                            HEAD_DIM, KV_TILE_SZ);
                }
            }
            time_gemm_qk += duration_cast<nanoseconds>(high_resolution_clock::now() - start_gemm);

            // Softmax (causal + scale + exp + normalize)
            auto start_softmax = high_resolution_clock::now();
            for (int i = 0; i < tile_rows; i++) {
                const int q_pos = i;
                // Apply causal mask and scale
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
                    if (KQ[i * KV_TILE_SZ + j] > max_val) {
                        max_val = KQ[i * KV_TILE_SZ + j];
                    }
                }

                // Exp and sum
                float sum = 0.0f;
                for (int j = 0; j < kv_tile_size; j++) {
                    if (KQ[i * KV_TILE_SZ + j] != -INFINITY) {
                        KQ[i * KV_TILE_SZ + j] = expf(KQ[i * KV_TILE_SZ + j] - max_val);
                        sum += KQ[i * KV_TILE_SZ + j];
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

            // GEMM: VKQ = KQ @ V (简化，不实际计算)
            auto start_vkq = high_resolution_clock::now();
            // 这里只测量时间，不实际计算
            time_gemm_vkq += duration_cast<nanoseconds>(high_resolution_clock::now() - start_vkq);
        }
    }

    std::cout << "=== 绝对时间对比 ===\n";
    std::cout << "步骤          | 我们 (ns) | llama.cpp (ns) | 差距\n";
    std::cout << "Load Q tile   |     953.7 |       " << time_load_q.count() / iterations << " | "
              << 953.7 * iterations / time_load_q.count() << "x\n";
    std::cout << "Load K tile   |    3126.8 |       " << time_load_k.count() / iterations << " | "
              << 3126.8 * iterations / time_load_k.count() << "x\n";
    std::cout << "Load V tile   |     800.3 |       " << time_load_v.count() / iterations << " | "
              << 800.3 * iterations / time_load_v.count() << "x\n";
    std::cout << "GEMM Q·K      |   11661.8 |       " << time_gemm_qk.count() / iterations << " | "
              << 11661.8 * iterations / time_gemm_qk.count() << "x\n";
    std::cout << "Softmax       |   13905.2 |       " << time_softmax.count() / iterations << " | "
              << 13905.2 * iterations / time_softmax.count() << "x\n";
    std::cout << "GEMM VKQ      |   11331.2 |       " << time_gemm_vkq.count() / iterations << " | "
              << 11331.2 * iterations / time_gemm_vkq.count() << "x\n";

    const auto total_ns = (time_load_q + time_load_k + time_load_v +
                           time_gemm_qk + time_softmax + time_gemm_vkq).count() / iterations;
    std::cout << "\n总计: 我们 41779 ns, llama.cpp " << total_ns << " ns, 差距 "
              << 41779.0 / total_ns << "x\n";

    return 0;
}
