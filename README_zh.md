# InferTrain

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CLA assistant](https://cla-assistant.io/readme/badge/OpenMachineware/infer_train)](https://cla-assistant.io/allowlist/OpenMachineware/infer_train)

[![英文文档](https://img.shields.io/badge/English_Docs-Click_here-brightgreen?style=for-the-badge)](./README.md)

**推训一体引擎** —— 纯 Mojo 编写的从零开始的 LLM 推理与训练引擎，仅依赖最小 C 运行时辅助层（pthread 任务池 + `mmap`）。

```
┌─────────────────────────────── infer_train ───────────────────────────────┐
│                                                                           │
│  CLI (llama.cpp 兼容)                HTTP 服务器 (OpenAI 兼容)             │
│  infer_train -m M -p P -n N       /v1/models /v1/chat/completions         │
│  quantize / serve                 /v1/completions (SSE) /v1/finetune      │
│  --infer-train-mode off|lora|full                                         │
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
./infer_train -m DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf \
    -p "<｜User｜>What is 1+1?<｜Assistant｜><think>" -n 120 --seed 7

# 7B 翻译模型 (Hy-MT2, hunyuan-dense)
./infer_train -m Hy-MT2-7B-Q4_K_M.gguf \
    -p "Translate to English: 今天天气很好。" -n 64

# 27B 混合模型 (Qwen3.8, Gated DeltaNet + 全注意力)
./infer_train -m Qwen3.8-27B-UD-Q5_K_M.gguf -c 32768 -p "Hello world" -n 64

# OpenAI 兼容 HTTP 服务器
INFERTRAIN_MODEL=DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf \
    python -m infer_train.server --port 8080
curl http://127.0.0.1:8080/v1/completions \
    -d '{"prompt":"1+1=","max_tokens":32,"temperature":0.6}'

# Python API
from infer_train import load_model
m = load_model("DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf")
print(m.generate("1+1=", max_tokens=32, seed=7))
losses = m.finetune("What is 1+1?", "1+1 equals 2.", lr=1e-5)  # 边推边训

# 微调后重量化 (.mmdl -> Q4_K_M / Q8_0 / NF4)
./infer_train quantize -i model.mmdl -o model_quantized.gguf -f Q4_K_M
```

## 核心功能

| 功能 | 说明 |
|---|---|
| **推理** | 三种架构（qwen2 / hunyuan-dense / qwen35 混合 SSM），GGUF 权重直接加载（Q4_K/Q5_K/Q6_K/Q8_0/IQ4_NL/IQ4_XS），KV Cache（稠密 / Paged / 滑动窗口 / 内存自适应），llama.cpp 兼容采样 |
| **训练** | 反向算子、`run_with_grad` 自动微分、AdamW/SGD、`train_step`/`eval_step`、梯度累积、混合精度（AMP）、动态量化 |
| **微调** | `finetune_step`（Mojo）与 `Model.finetune`（Python）——服务不停机，LoRA 风格只更新可训练子集 |
| **量化** | 微调后重量化：FP16 → Q4_K_M / Q8_0 / NF4（`infer_train quantize`），量化文件兼容 GGUF 工具链 |
| **存储** | 私有 `.mmdl` 检查点：权重 + 梯度 + AdamW m/v + 训练元数据；增量更新、delta 追加、断点续训；`strip_to_gguf` 导出纯权重 |
| **分词** | `Tokenizer` trait + Qwen/Llama/Hunyuan 三实现，按 `tokenizer.ggml.model`/`tokenizer.ggml.pre` 自动选择，`register_tokenizer` 注册自定义分词器 |
| **API** | C ABI + Python 绑定 + `torch.compile` 后端；OpenAI 兼容 HTTP 端点（`/v1/models`、`/v1/chat/completions`、`/v1/completions` SSE、`/v1/finetune`、`/v1/finetune/status`），`INFERTRAIN_API_KEY` 鉴权 |
| **CLI** | `infer_train`：llama.cpp 核心参数（`-m -c -np --host --port --temp --top-p --top-k --repeat-penalty -t --no-cnv`）+ `--infer-train-*` 参数组 |

## 性能基准（Apple M1 Max，64 GB；InferTrain 为 CPU 多线程）

| 模型 | 大小 | 预填充 | 解码 | 峰值内存 | llama.cpp（CPU）解码 | 对比 |
|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B（Q5_K_M） | 1.28 GB | 4.4 t/s | 4.2 t/s | ~5 GB | 60.3 t/s | 7% |
| Hy-MT2-7B（Q4_K_M, hunyuan-dense） | 4.6 GB | 0.82 t/s | 0.82 t/s | ~15 GB | 19.6 t/s | 4% |
| Qwen3.8-27B（UD-Q5_K_M, qwen35 hybrid） | 19.7 GB | 0.23 t/s | 0.22 t/s | ~55 GB | 4.4 t/s | 5% |

> ⚠️ **性能目标未达成**：M7 的「推理速度达 llama.cpp 85%」目标当前为 4–7%（CPU 基线）。数值正确性不受影响（与 llama.cpp 逐 token 一致），差距在 内核效率（标量 DeltaNet 递推、无 AMX 权重重排、逐元素注意力），完整分析、 32K 上下文内存数据与 v1.1 优化路线见 `docs/M7_PERFORMANCE_REPORT.md`； M5/M6 数据见 `docs/M5_PERFORMANCE_REPORT.md` 与 `docs/M6_TRAINING_REPORT.md`。

## 已验证的数值正确性

* **Hy-MT2-7B（hunyuan-dense）**：与 llama.cpp 逐 token 对比——连续 4 步贪心解码 token 完全一致，全 logits 向量相关系数 ≥ 0.9995。
* **Qwen3.8-27B（qwen35 混合）**：单 token / 双 token / 6 步贪心解码与 llama.cpp 逐 token 一致（单 token logits 相关系数 0.9991）。
* **1.5B（qwen2）**：M3 回归基准 `reference_logits_5.npy` 逐位一致（含新架构重构后）。
* 所有 GGUF 反量化格式对 gguf-py（llama.cpp 官方 Python 实现）逐一校验。
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
make test-m5     # IR 优化器 + JIT + torch.compile 后端
make test-m6     # 训练套件
make test-m7     # 所有测试 + M7 功能套件
make cli         # infer_train 二进制
```

测试组成：Mojo 可执行文件（`tests/*.mojo`，`pixi run mojo build -I .` 编译）+ Python 验收套件（`tests/python/`）。Mojo 1.0 约束：统一 `def`、无运行时全局变量、 `fn` 已弃用；C 侧辅助库经 `-Xlinker` 链接（`tools/thread_pool.c`）。

## 已知限制 / TODO

* **Qwen3.8 的 MTP（nextn_predict）模块未启用**——与 llama.cpp 默认单模型解码一致， MTP 投机解码留待 v1.1。
* **NF4 为私有 GGML 扩展类型（30）**，llama.cpp 无法读取 NF4 权重；导出给 llama.cpp 请用 `-f Q4_K_M` / `Q8_0`。
* 混合精度训练（AMP）为 fp32 主权重 + fp16 影子；FSDP/流水并行未实现。
* GPU 后端（CUDA/ROCm）接口已就绪（见 PORTING_GUIDE），内核留待移植。
* 更多算子 / 更多模型（MoE、VL）、更多平台（Windows/Linux 打包）。
