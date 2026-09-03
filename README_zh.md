# InferTrain

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CLA assistant](https://cla-assistant.io/readme/badge/OpenMachineware/infer_train)](https://cla-assistant.io/allowlist/OpenMachineware/infer_train)

[![英文文档](https://img.shields.io/badge/English_Docs-Click_here-brightgreen?style=for-the-badge)](./README.md)

**推训一体引擎** —— 纯 Mojo 编写的从零开始的 LLM 推理与训练引擎，仅依赖最小 C 运行时辅助层（pthread 任务池 + `mmap`）。

```
┌─────────────────────────────── infer_train ───────────────────────────────┐
│                                                                           │
│  it-cli (llama-cli 兼容)          it-server (OpenAI 兼容 HTTP)            │
│  -m M -p P -n N -sm layer --rpc   /v1/models /v1/chat/completions         │
│  it-rpc-server (RPC worker)       /v1/completions (SSE) /v1/finetune      │
│  --infer-train-mode off|lora|full it-server quantize                      │
│                          │                              │                │
│  ┌───────────────────────▼──────────────────────────────▼──────────────┐  │
│  │  Python 绑定 (C ABI)  ·  torch.compile 后端 "infer_train"           │  │
│  └───────────────────────┬──────────────────────────────┬──────────────┘  │
│                          │                              │                 │
│  ┌───────────────────────▼───────────────┐  ┌───────────▼──────────────┐  │
│  │  运行时: Model / generate /           │  │  训练: TrainModel /       │  │
│  │  finetune (边推边训)                   │  │  run_with_grad / AdamW    │  │
│  └───────────────────────┬───────────────┘  └───────────┬──────────────┘  │
│                          │                              │                 │
│  ┌───────────────────────▼──────────────────────────────▼──────────────┐  │
│  │  核心: tensor / graph / interpreter / optimizer(IR) / JIT           │  │
│  │  算子: CPU 内核 · 融合内核 · 量化内核 (GGUF Q4_K/Q5_K/Q6_K/         │  │
│  │        Q8_0/IQ4_NL/IQ4_XS/NF4) · 注意力 (KV cache: dense/paged/    │  │
│  │        滑动窗口) · 自动微分                                           │  │
│  │  Transformer: qwen2 · hunyuan-dense · qwen35 (Gated DeltaNet 混合)  │  │
│  │  分词器: Qwen / Llama / Hunyuan (GPT-2 byte-level BPE,              │  │
│  │           根据 GGUF 元数据自动选择) + 自定义注册表                    │  │
│  │  存储: GGUF 加载器 (mmap) · .mmdl 检查点 (GGUF 兼容)                │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 构建所有组件并运行完整测试套件
make test-m7

# 使用 1.5B 模型生成
make cli
./it-cli -m DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf \
    -p "<｜User｜>What is 1+1?<｜Assistant｜><think>" -n 120 --seed 7

# 7B 翻译模型 (Hy-MT2, hunyuan-dense)
./it-cli -m Hy-MT2-7B-Q4_K_M.gguf \
    -p "Translate to English: 今天天气很好。" -n 64

# 27B 混合模型 (Qwen3.8, Gated DeltaNet + 全注意力)
./it-cli -m Qwen3.8-27B-UD-Q5_K_M.gguf -c 32768 -p "Hello world" -n 64

# OpenAI 兼容 HTTP 服务器（原生 Mojo 二进制，llama-server 兼容）
make server
./it-server -m DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf --port 8080
curl http://127.0.0.1:8080/v1/completions \
    -d '{"prompt":"1+1=","max_tokens":32,"temperature":0.6}'

# OpenAI 兼容 HTTP 服务器（Python 封装，支持并发）
INFERTRAIN_MODEL=DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf \
    python -m infer_train.server --port 8080

# Python API
from infer_train import load_model
m = load_model("DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf")
print(m.generate("1+1=", max_tokens=32, seed=7))
losses = m.finetune("What is 1+1?", "1+1 equals 2.", lr=1e-5)  # 边推边训

# 微调后重量化 (.mmdl -> Q4_K_M / Q8_0 / NF4)
./it-server quantize -i model.mmdl -o model_quantized.gguf -f Q4_K_M
```

## GGUF 分卷（多文件）模型

大模型常以*分卷* GGUF 分发——多个命名为 `<base>.gguf-NNNNN-of-NNNNN.gguf`
（5 位补零）的分卷文件，由 llama.cpp 的 `llama-gguf-split` 生成。InferTrain
可透明加载：

```bash
# 分割模型 (brew install llama.cpp)
llama-gguf-split --split-max-size 600M model.gguf model.gguf
# -> model.gguf-00001-of-00003.gguf  model.gguf-00002-of-00003.gguf  model.gguf-00003-of-00003.gguf

# 加载任意一卷 —— 自动 mmap 并合并全部分卷
./it-cli -m model.gguf-00001-of-00003.gguf -p "Hello" -n 64

# Python API
m = load_model("model.gguf-00001-of-00003.gguf")
```

工作原理：

* 每个分卷都是合法 GGUF 文件，只含自己的 tensor 子集；**第一卷**携带完整
  metadata（其余卷只有 `split.no` / `split.tensors.count` / `split.count`），
  且每个 tensor 的 offset 相对自己所在卷的 data 段。
* `load_gguf` 自动识别输入：分卷 part 路径（`…gguf-NNNNN-of-NNNNN.gguf`）
  加载整个分卷；普通 `.gguf` 按单文件加载；磁盘上存在分卷文件的 base 名
  会被展开为分卷加载。
* 每个 tensor 都标记了字节所属的分卷（`GGUFTensor.file_idx`），反量化从
  正确的映射读取——分卷模型与单文件数值完全一致。

`make test-gguf-split` 验证这一点：将 1.5B 模型分别按单文件和 3 卷分卷加载，
检查各卷反量化权重逐字节一致（`max_diff = 0.0`）。

## 多进程 / 多机 RPC（llama.cpp 风格）

引擎可以像 llama.cpp 一样跨进程（单机）或跨机器分布式运行，命令行用法相同：
独立的 worker 二进制（`it-rpc-server`，对应 `llama-rpc-server`）+
`it-cli` 上的 `-sm` / `--rpc` 参数。

```bash
# 每个进程 / 机器一个 worker（各自 mmap 同一个 GGUF 文件）：
./it-rpc-server -m model.gguf --port 50052 &
./it-rpc-server -m model.gguf --port 50053 &

# 把层切分到各 worker 上并生成：
./it-cli -m model.gguf -sm layer \
    --rpc 127.0.0.1:50052 --rpc 127.0.0.1:50053 \
    -p "Hello" -n 64
```

工作原理：

* `-sm layer`（层切分）：master（`it-cli`）保留 embedding + 输出头与生成循环；
  每个 `--rpc` 端点负责一段连续的层（28 层分到 3 个 worker → 10/9/9）以及
  这些层的 KV / SSM 状态。每个 token 由 master 串联各 worker：
  `embedding → worker 0 → worker 1 → … → output head → 采样`。
* 线上协议为纯 TCP，4 字节长度前缀消息（`INIT` / `FORWARD` / `RESET` /
  `PING`）；fp16 hidden state 以原始位模式过网，因此分布式运行与本地运行
  **数值完全一致**。
* `--rpc` 可重复、接受 `host:port`，因此同一条命令跨机器也能用
  （把端点指向各 worker 的地址即可）。
* `-sm row`（行并行）已接受但尚未实现。

`make test-rpc` 在本地验证这一点：启动两个 localhost worker，同一段贪心
生成分别跑一次本地、一次跨 worker，要求输出完全一致。

## 核心功能

| 功能 | 说明 |
|---|---|
| **推理** | 三种架构（qwen2 / hunyuan-dense / qwen35 混合 SSM），GGUF 权重直接加载（Q4_K/Q5_K/Q6_K/Q8_0/IQ4_NL/IQ4_XS），分卷/多文件 GGUF（`llama-gguf-split`，自动识别），KV Cache（稠密 / Paged / 滑动窗口 / 内存自适应），llama.cpp 兼容采样 |
| **训练** | 反向算子、`run_with_grad` 自动微分、AdamW/SGD、`train_step`/`eval_step`、梯度累积、混合精度（AMP）、动态量化 |
| **微调** | `finetune_step`（Mojo）与 `Model.finetune`（Python）——服务不停机，LoRA 风格只更新可训练子集 |
| **量化** | 微调后重量化：FP16 → Q4_K_M / Q8_0 / NF4（`it-server quantize`），量化文件兼容 GGUF 工具链 |
| **存储** | 私有 `.mmdl` 检查点：权重 + 梯度 + AdamW m/v + 训练元数据；增量更新、delta 追加、断点续训；`strip_to_gguf` 导出纯权重 |
| **分词** | 每种算法一个通用引擎（GPT-2 byte-level BPE；SentencePiece 待补充）——模型家族只在数据上有差异（flavor 标签、附加 token、bos/eos），按 `tokenizer.ggml.model`/`tokenizer.ggml.pre` 自动选择，`register_tokenizer` 注册自定义分词器 |
| **分布式** | 多进程 / 多机 RPC（llama.cpp 风格）：`it-rpc-server` worker + `it-cli` 上的 `-sm layer` / `--rpc host:port`，层切分、每个 worker 持有自己的 KV/SSM 状态，与本地运行数值完全一致 |
| **API** | C ABI + Python 绑定 + `torch.compile` 后端；OpenAI 兼容 HTTP 端点（`/v1/models`、`/v1/chat/completions`、`/v1/completions` SSE、`/v1/finetune`、`/v1/finetune/status`），`INFERTRAIN_API_KEY` 鉴权 |
| **CLI** | 三个入口共用同一套核心：`it-cli`（llama-cli 兼容的快速验证：`-m -p -c -n --temp --top-p --top-k --repeat-penalty -t --seed -sm --rpc`）、`it-server`（llama-server 兼容的 OpenAI HTTP 服务 + `quantize`）、`it-rpc-server`（RPC worker）；均支持 `--infer-train-*` 参数组 |

## 性能基准（Apple M1 Max，64 GB；InferTrain 为 CPU 多线程）

| 模型 | 大小 | 预填充 | 解码 | 峰值内存 | llama.cpp（CPU）解码 | 对比 |
|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B（Q5_K_M） | 1.28 GB | 4.4 t/s | 4.2 t/s | ~5 GB | 60.3 t/s | 7% |
| Hy-MT2-7B（Q4_K_M, hunyuan-dense） | 4.6 GB | 0.82 t/s | 0.82 t/s | ~15 GB | 19.6 t/s | 4% |
| Qwen3.8-27B（UD-Q5_K_M, qwen35 hybrid） | 19.7 GB | 0.23 t/s | 0.22 t/s | ~55 GB | 4.4 t/s | 5% |

> ⚠️ **性能目标未达成**：M7 的「推理速度达 llama.cpp 85%」目标当前为 4–7%（CPU 基线）。数值正确性不受影响（与 llama.cpp 逐 token 一致），差距在 内核效率（标量 DeltaNet 递推、无 AMX 权重重排、逐元素注意力），完整分析、 32K 上下文内存数据与后续版本优化路线见 `docs/M7_PERFORMANCE_REPORT.md`； M5/M6 数据见 `docs/M5_PERFORMANCE_REPORT.md` 与 `docs/M6_TRAINING_REPORT.md`。

## 已验证的数值正确性

* **Hy-MT2-7B（hunyuan-dense）**：与 llama.cpp 逐 token 对比——连续 4 步贪心解码 token 完全一致，全 logits 向量相关系数 ≥ 0.9995。
* **Qwen3.8-27B（qwen35 混合）**：单 token / 双 token / 6 步贪心解码与 llama.cpp 逐 token 一致（单 token logits 相关系数 0.9991）。
* **1.5B（qwen2）**：M3 回归基准 `reference_logits_5.npy` 逐位一致（含新架构重构后）。
* 所有 GGUF 反量化格式对 gguf-py（llama.cpp 官方 Python 实现）逐一校验。FP32 反量化路径（`dequantize_into_f32` / `dequantize_q4_k_m_f32`）与 llama.cpp 的 `ggml-quants.c` **逐位一致**（Q4_K/Q5_K/Q6_K/Q8_0/IQ4_NL/IQ4_XS/F32，每 1024 值 0 失配）；fp16 推理路径存储的是精确值的一次 fp16 舍入（最大相对误差 4.85e-4，受 fp16 半 ULP 2⁻¹¹ ≈ 4.88e-4 约束）——分析与对训练的影响见 `docs/M7_PERFORMANCE_REPORT.md` §4。
* M6 训练：与 PyTorch eager 逐位一致（loss 3.58 → 1.77）。

## 生态

* [CONTRIBUTING.md](CONTRIBUTING.md) — 贡献指南（中文版：[CONTRIBUTING_zh.md](CONTRIBUTING_zh.md)）
* [CLA.md](CLA.md) — 贡献者许可协议（中文版：[CLA_zh.md](CLA_zh.md)）
* [docs/NEW_OPERATOR_GUIDE.md](docs/NEW_OPERATOR_GUIDE.md) — 如何发现并添加新算子
* [docs/FUSION_GUIDE.md](docs/FUSION_GUIDE.md) — 如何扩展算子融合 Pass
* [docs/PORTING_GUIDE.md](docs/PORTING_GUIDE.md) — 如何移植到新硬件平台（CUDA/ROCm/…）
* [docs/M5_PERFORMANCE_REPORT.md](docs/M5_PERFORMANCE_REPORT.md) / [docs/M6_TRAINING_REPORT.md](docs/M6_TRAINING_REPORT.md) — 里程碑报告
* [LICENSE](LICENSE)

## 开发

```bash
make tp          # 构建 C 运行时辅助库 (线程池 + mmap)
make test        # M1/M2 核心测试
make test-m3     # 分词器 / 算子 / 前向验证
make test-gguf-split  # GGUF 分卷（多文件）加载
make test-m5     # IR 优化器 + JIT + torch.compile 后端
make test-m6     # 训练套件
make test-m7     # 所有测试 + M7 功能套件
make test-rpc    # 多进程 RPC（两个 localhost worker，-sm layer）
make cli         # it-cli 二进制（llama-cli 兼容）
make server      # it-server 二进制（llama-server 兼容 HTTP）
make rpc-server  # it-rpc-server worker 二进制（llama-rpc-server 兼容）
```

测试组成：Mojo 可执行文件（`tests/*.mojo`，`pixi run mojo build -I .` 编译）+ Python 验收套件（`tests/python/`）。Mojo 1.0 约束：统一 `def`、无运行时全局变量、 `fn` 已弃用；C 侧辅助库经 `-Xlinker` 链接（`tools/thread_pool.c`）。

## 已知限制 / TODO

* **Qwen3.8 的 MTP（nextn_predict）模块未启用**——与 llama.cpp 默认单模型解码一致， MTP 投机解码留待后续版本。
* **NF4 为私有 GGML 扩展类型（30）**，llama.cpp 无法读取 NF4 权重；导出给 llama.cpp 请用 `-f Q4_K_M` / `Q8_0`。
* **fp16 反量化精度缺口**——推理权重是精确（FP32）反量化值的一次 fp16 舍入：每元素相对误差 ≤ 4.88e-4，确定性、对推理可忽略（逐 token 一致）、对微调影响远小于导出时的重量化误差，但会破坏与 FP32 初始化运行的跨引擎逐位复现。逐位一致的 FP32 路径（`dequantize_into_f32` / `dequantize_q4_k_m_f32`）已就绪；后续版本方向：量化点积内核（彻底消除该缺口）、FP32 微调初始化、按张量 fp32 选项——见 `docs/M7_PERFORMANCE_REPORT.md` §4。
* 混合精度训练（AMP）为 fp32 主权重 + fp16 影子；行并行（`-sm row`，带 allreduce 的张量并行）尚未实现——RPC 传输层与层切分（`-sm layer`）已为其就绪。
* GPU 后端（CUDA/ROCm）接口已就绪（见 PORTING_GUIDE），内核留待移植。
* 更多算子 / 更多模型（MoE、VL）、更多平台（Windows/Linux 打包）。
