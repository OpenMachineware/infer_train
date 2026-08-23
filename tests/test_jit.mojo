# tests/test_jit.mojo
#
# M5: JIT specialization - comptime-specialized FFN through the
# interpreter's jit-marked node path.
#
# Checks:
#   1. a jit-marked swiglu_ffn node executes through the JitCache and its
#      output matches the generic registry path exactly;
#   2. the cache records the compiled shape key (hit on re-run);
#   3. the specialized kernel is genuinely faster than the generic path
#      (recorded timings, not asserted - the M5 report collects them).

from src.core.graph import Graph, AttrValue
from src.runtime.interpreter import Interpreter
from src.core.ops.base.op_registry import OpRegistry
from src.core.ops.base.op_interface import AnyTensor, to_any
from src.core.tensor import Tensor, tensor_zeros
from src.core.jit.jit_cache import JitCache
from std.utils.static_tuple import StaticTuple
from std.ffi import external_call


def time_ms() -> Float64:
    var us = external_call["clock_gettime_nsec_np", Int64, Int32](Int32(0))
    return Float64(us) / 1000000.0


def _fill(mut t: Tensor[DType.float16, 2], seed: Int, scale: Float32):
    for i in range(t.numel()):
        var v = Float32((i * 7 + seed * 13) % 101) / Float32(101.0) * scale
        t.set(i, Scalar[DType.float16](v))


def build_ffn_graph(jit_mark: Bool) -> Graph:
    """A single swiglu_ffn node (optionally jit-marked)."""
    var graph = Graph()
    var attrs = Dict[String, AttrValue]()
    attrs["layer"] = AttrValue(0)
    if jit_mark:
        attrs["jit"] = AttrValue(1)
    _ = graph.add_node("swiglu_ffn", List[Int](), attrs)
    return graph^


def ffn_inputs() -> List[AnyTensor]:
    var M = 1
    var K = 1536
    var F = 8960
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    var g = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](F, K))
    var u = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](F, K))
    var d = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](K, F))
    _fill(x, 3, 1)
    _fill(g, 4, 0.1)
    _fill(u, 5, 0.1)
    _fill(d, 6, 0.1)
    var inputs = List[AnyTensor]()
    inputs.append(to_any[DType.float16, 2](x))
    inputs.append(to_any[DType.float16, 2](g))
    inputs.append(to_any[DType.float16, 2](u))
    inputs.append(to_any[DType.float16, 2](d))
    return inputs^


def _max_diff(a: AnyTensor, b: AnyTensor) -> Float32:
    var pa = a.data.unsafe_bitcast[Scalar[DType.float16]]()
    var pb = b.data.unsafe_bitcast[Scalar[DType.float16]]()
    var mx = Float32(0)
    for i in range(a.numel):
        var d = Float32(pa.unsafe_load(offset=i)) - Float32(
            pb.unsafe_load(offset=i)
        )
        if d < Float32(0):
            d = -d
        if d > mx:
            mx = d
    return mx


def check(label: String, ok: Bool):
    if ok:
        print("[ok]   ", label)
    else:
        print("[FAIL] ", label)
        std.os.os.abort()


def main():
    print("== M5 JIT tests ==")

    # generic path (jit disabled)
    var generic_graph = build_ffn_graph(False)
    var registry = OpRegistry()
    registry.register_default_ops()
    var generic_interp = Interpreter(generic_graph^, registry^)
    generic_interp.jit_enabled = False

    # jit path
    var jit_graph = build_ffn_graph(True)
    var registry2 = OpRegistry()
    registry2.register_default_ops()
    var jit_interp = Interpreter(jit_graph^, registry2^)

    var inputs = ffn_inputs()
    var ref_out = generic_interp.run(inputs)
    var got = jit_interp.run(inputs)

    check("jit-marked node matches generic execution",
          len(ref_out) == 1 and len(got) == 1
          and _max_diff(ref_out[0], got[0]) < Float32(1e-3))
    check(
        "cache holds the compiled FFN shape key",
        jit_interp.jit_cache.has("ffn/1/8960/1536"),
    )

    # timing: jit vs generic over a few iterations (recorded)
    var n = 3
    var t0 = time_ms()
    for i in range(n):
        _ = generic_interp.run(inputs)
    var t1 = time_ms()
    for i in range(n):
        _ = jit_interp.run(inputs)
    var t2 = time_ms()
    print(
        "  timing: generic ", (t1 - t0) / Float64(n), " ms/iter vs jit ",
        (t2 - t1) / Float64(n), " ms/iter",
    )

    # cache miss path: a different shape registers + falls back generically
    var cache = JitCache()
    var x2 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](2, 128))
    var g2 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](256, 128))
    var u2 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](256, 128))
    var d2 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](128, 256))
    var out2 = cache.run_ffn(x2, g2, u2, d2)
    check("cache miss falls back and records the key",
          out2.numel() == 256
          and cache.has("ffn/2/256/128"))

    print("== JIT tests passed ==")
