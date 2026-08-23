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

from .jit_compile import jit_ffn, jit_ffn_key, jit_ffn_generic
from ..tensor import Tensor

comptime JIT_FFN_M = 1
comptime JIT_FFN_F = 8960
comptime JIT_FFN_K = 1536


struct JitCache(Movable):
    var compiled: Dict[String, Int]  # shape key -> 1 when compiled

    def __init__(out self):
        self.compiled = Dict[String, Int]()
        self.compiled[jit_ffn_key(JIT_FFN_M, JIT_FFN_F, JIT_FFN_K)] = 1

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
            return jit_ffn[JIT_FFN_M, JIT_FFN_F, JIT_FFN_K](
                x, gate_w, up_w, down_w
            )
        # cache miss: run the generic kernel and record the key
        self.compiled[jit_ffn_key(m, f, k)] = 1
        return jit_ffn_generic(x, gate_w, up_w, down_w)
