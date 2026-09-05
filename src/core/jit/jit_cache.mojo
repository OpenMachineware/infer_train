# core/jit/jit_cache.mojo
#
# M5 : the JIT specialization cache.
#
# Shape-keyed cache over the comptime-specialized kernels in
# jit_compile.mojo.  A node marked `jit` in the interpreter asks the cache
# for its shape key; on a hit the cached instantiation runs, on a miss the
# generic kernel runs and the key is recorded (in a real M6 JIT the miss
# would compile new machine code at runtime; Mojo 1.0 has no runtime
# codegen, so M5 "compiles" the specializations for the exact model
# shapes - 1x1536x8960 FFN and 28/8/128 MHA - ahead of time, which is the
# documented scope).
#
# M8 : CPU shape-specialization cache for fused ops.
#
# `get_or_compile(shape_signature, compile_fn, m, n, k, width)` is the
# shape-keyed compile-or-fetch API: on a hit it returns the cached
# compiled kernel, on a miss it runs the compile function, records the
# shape, caches the result, and returns it.  `compile_fn` is a thin
# (non-capturing) function value - the closest thing to the task's
# `Function` parameter in Mojo 1.0 (a top-level function; thin functions
# cannot capture, so the shape is passed explicitly).
#
# `run_fused_jit` is the fused matmul+rmsnorm entry: it builds the shape
# signature from the runtime (M, N, K) and dispatches through the cache
# when `specialize` is on (the --jit-specialize flag; off by default so
# the default behavior is exactly the generic kernel).

from .jit_compile import jit_ffn, jit_ffn_key, jit_ffn_generic
from ..tensor import Tensor
from ..ops.cpu.matmul_cpu import (
    CompiledFusedKernel,
    compile_fused_kernel,
    fused_matmul_rms_norm_key,
)
from ..ops.fused.matmul_rms_norm import fused_matmul_rms_norm

comptime JIT_FFN_M = 1
comptime JIT_FFN_F = 8960
comptime JIT_FFN_K = 1536


struct JitCache(Movable):
    var compiled: Dict[String, Int]  # shape key -> 1 when compiled
    var fused_kernels: Dict[
        String, CompiledFusedKernel
    ]  # M8: fused shape -> kernel
    var compiles: Int  # M8: JIT compile count (misses that produced a kernel)
    var hits: Int  # M8: cache hits
    var misses: Int  # M8: cache misses

    def __init__(out self):
        self.compiled = Dict[String, Int]()
        self.compiled[jit_ffn_key(JIT_FFN_M, JIT_FFN_F, JIT_FFN_K)] = 1
        self.fused_kernels = Dict[String, CompiledFusedKernel]()
        self.compiles = 0
        self.hits = 0
        self.misses = 0

    def has(self, key: String) -> Bool:
        return self.compiled.get(key, 0) == 1

    def compile_ffn(mut self, m: Int, f: Int, k: Int) -> Bool:
        """Register (and thereby comptime-instantiate) an FFN shape."""
        if m == JIT_FFN_M and f == JIT_FFN_F and k == JIT_FFN_K:
            self.compiled[jit_ffn_key(m, f, k)] = 1
            return True
        return False  # other shapes stay on the generic path (M6 JIT)

    def run_ffn(
        mut self,
        x: Tensor[DType.float16, 2],
        gate_w: Tensor[DType.float16, 2],
        up_w: Tensor[DType.float16, 2],
        down_w: Tensor[DType.float16, 2],
    ) -> Tensor[DType.float16, 2]:
        var m = x.shape()[0]
        var k = x.shape()[1]
        var f = gate_w.shape()[0]
        if self.has(jit_ffn_key(m, f, k)):
            self.hits += 1
            return jit_ffn[JIT_FFN_M, JIT_FFN_F, JIT_FFN_K](
                x, gate_w, up_w, down_w
            )
        # cache miss: run the generic kernel and record the key
        self.misses += 1
        self.compiled[jit_ffn_key(m, f, k)] = 1
        return jit_ffn_generic(x, gate_w, up_w, down_w)

    def run_ffn_width(
        mut self,
        x: Tensor[DType.float16, 2],
        gate_w: Tensor[DType.float16, 2],
        up_w: Tensor[DType.float16, 2],
        down_w: Tensor[DType.float16, 2],
        width_bits: Int,
    ) -> Tensor[DType.float16, 2]:
        """run_ffn with the projection k-loop SIMD width chosen by the
        M8 SIMD autotuner: each branch is a comptime instantiation
        (4/8/16 lanes = 64/128/256-bit)."""
        var m = x.shape()[0]
        var k = x.shape()[1]
        var f = gate_w.shape()[0]
        if self.has(jit_ffn_key(m, f, k)):
            self.hits += 1
            if width_bits == 256:
                return jit_ffn[JIT_FFN_M, JIT_FFN_F, JIT_FFN_K, 16](
                    x, gate_w, up_w, down_w
                )
            elif width_bits == 64:
                return jit_ffn[JIT_FFN_M, JIT_FFN_F, JIT_FFN_K, 4](
                    x, gate_w, up_w, down_w
                )
            return jit_ffn[JIT_FFN_M, JIT_FFN_F, JIT_FFN_K, 8](
                x, gate_w, up_w, down_w
            )
        # cache miss: run the generic kernel and record the key
        self.misses += 1
        self.compiled[jit_ffn_key(m, f, k)] = 1
        return jit_ffn_generic(x, gate_w, up_w, down_w)

    def get_or_compile(
        mut self,
        shape_signature: String,
        compile_fn: def(Int, Int, Int, Int) thin -> CompiledFusedKernel,
        m: Int,
        n: Int,
        k: Int,
        width: Int,
    ) -> CompiledFusedKernel:
        """Return the compiled kernel for `shape_signature`, compiling it
        first when the shape is not in the cache.

        Hit:  the cached kernel is returned and `hits` increments.
        Miss: `compile_fn(m, n, k, width)` materializes the
              comptime-specialized kernel, the shape is recorded, the new
              kernel is cached, and it is returned (`compiles` and
              `misses` increment).
        """
        var cached = self.fused_kernels.get(shape_signature)
        if cached:
            self.hits += 1
            return cached.value().copy()
        self.misses += 1
        self.compiles += 1
        self.compiled[shape_signature] = 1
        var kernel = compile_fn(m, n, k, width)
        self.fused_kernels[shape_signature] = kernel.copy()
        return kernel.copy()

    def hit_rate(self) -> Float64:
        """hits / (hits + misses); 0 when nothing has been looked up."""
        var total = self.hits + self.misses
        if total == 0:
            return Float64(0.0)
        return Float64(self.hits) / Float64(total)

    def run_fused_jit(
        mut self,
        x: Tensor[DType.float16, 2],
        w: Tensor[DType.float16, 2],
        eps: Float32,
        specialize: Bool,
        width_bits: Int,
    ) -> Tensor[DType.float16, 2]:
        """JIT-specialized fused matmul+rmsnorm (CPU path only).

        specialize=False: the generic kernel, exactly as before (the
        default behavior; --jit-specialize is off).
        specialize=True: dispatch by shape signature through the cache;
        a shape without a compiled specialization falls back to the
        generic kernel and is recorded (the M5 comptime-JIT model -
        "recompiling" a new shape in Mojo 1.0 means recording it for the
        next run, since there is no runtime codegen).
        """
        if not specialize:
            return fused_matmul_rms_norm[DType.float16](x, w, eps)
        var m = x.shape()[0]
        var k = x.shape()[1]
        var n = w.shape()[0]
        var sig = fused_matmul_rms_norm_key(m, n, k)
        var kernel = self.get_or_compile(
            sig, compile_fused_kernel, m, n, k, width_bits
        )
        return kernel.run(x, w, eps)
