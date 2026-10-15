---
name: gemm-n32-dispatch-fix
description: GEMM template requires N >= 32, use sequential MV for smaller batches
metadata:
  type: feedback
---

GEMM template kernel requires N >= 32 batch size. For N < 32, use sequential MV kernel calls to avoid NaN output. The function constants in the GEMM template are set for specific batch configurations and cannot handle arbitrary small batch sizes.

**Why:** GEMM template was returning NaN for N=6 and other small batch sizes.

**How to apply:** In matmul_q*_k_f32 functions, check if n < 32 and fall back to sequential MV instead of GEMM template.
