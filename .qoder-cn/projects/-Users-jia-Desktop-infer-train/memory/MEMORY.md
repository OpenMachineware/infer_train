# Memory Index

## Model Testing Status (2026-10-15)

### Working Models (GPU Inference)
- **Qwen3-0.6B-UD-Q4_K_XL.gguf**: ✅ GPU inference working, slow but no NaN
- **Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf**: ✅ GPU inference working, 0.65 t/s

### Not Supported
- **DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L.gguf**: ❌ MoE architecture (deepseek2) not fully implemented

### Known Issues
- [GEMM N<32 Dispatch Fix](gemm-n32-dispatch-fix.md) — GEMM template requires N >= 32
