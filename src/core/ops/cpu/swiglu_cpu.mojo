# core/ops/cpu/swiglu_cpu.mojo
#
# SwiGLU activation for the Qwen2 FFN: out = silu(gate) * up, where
# silu(x) = x * sigmoid(x).  Computed in f32 and cast back to `dtype`.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from ...simd_utils import W_F16, W_F32
from std.math import exp
from std.utils.static_tuple import StaticTuple


def _silu(x: Float32) -> Float32:
    if x < Float32(-20.0):
        return Float32(0.0)
    if x > Float32(20.0):
        return x
    return x / (Float32(1.0) + exp(-x))


def _swiglu_cpu_kernel[
    dtype: DType, simd_width: Int = 0
](gate: Tensor[dtype, 2], up: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """out = silu(gate) * up, computed in f32.  `simd_width` is the SIMD
    lane count (comptime); 0 selects the legacy per-dtype width."""
    if gate.shape() != up.shape():
        unimplemented("swiglu_cpu: shape mismatch")
    var out = tensor_zeros[dtype, 2](gate.shape())
    var n = gate.numel()
    comptime W = simd_width if simd_width > 0 else (
        W_F16 if dtype == DType.float16 else W_F32
    )
    var n_main = (n // W) * W
    var i = 0
    while i < n_main:
        var gv = (
            gate.data().unsafe_load[width=W](offset=i).cast[DType.float32]()
        )
        var uv = up.data().unsafe_load[width=W](offset=i).cast[DType.float32]()
        # per-lane silu: vector loads, scalar exp, scalar stores (SIMD
        # lane inserts are miscompiled in Mojo 1.0)
        for lane in range(W):
            var v = _silu(gv[lane]) * uv[lane]
            out.set(i + lane, Scalar[dtype](v))
        i += W
    while i < n:
        var g = Float32(gate.get(i))
        var u = Float32(up.get(i))
        out.set(i, Scalar[dtype](_silu(g) * u))
        i += 1
    return out


def swiglu_cpu[
    dtype: DType, rows: Int, cols: Int, simd_width: Int = 0
](gate: Tensor[dtype, 2], up: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Comptime-shaped SwiGLU."""
    if gate.shape() != StaticTuple[Int, 2](rows, cols):
        unimplemented("swiglu_cpu: static shape mismatch")
    return _swiglu_cpu_kernel[dtype, simd_width](gate, up)


def swiglu_cpu_dynamic[
    dtype: DType, simd_width: Int = 0
](gate: Tensor[dtype, 2], up: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Runtime-shaped SwiGLU."""
    return _swiglu_cpu_kernel[dtype, simd_width](gate, up)


def swiglu_cpu_autotuned[
    dtype: DType
](gate: Tensor[dtype, 2], up: Tensor[dtype, 2], width_bits: Int) -> Tensor[
    dtype, 2
]:
    """SwiGLU specialized for `width_bits` (64/128/256).

    `width_bits` is a runtime value (the autotuner's choice); each branch
    calls a comptime instantiation with a literal lane width.
    """
    comptime if dtype == DType.float16:
        if width_bits == 256:
            return _swiglu_cpu_kernel[dtype, 16](gate, up)
        elif width_bits == 64:
            return _swiglu_cpu_kernel[dtype, 4](gate, up)
        return _swiglu_cpu_kernel[dtype, 8](gate, up)
    else:
        if width_bits == 256:
            return _swiglu_cpu_kernel[dtype, 8](gate, up)
        elif width_bits == 64:
            return _swiglu_cpu_kernel[dtype, 2](gate, up)
        return _swiglu_cpu_kernel[dtype, 4](gate, up)


def swiglu_cpu_forward_with_saved[
    dtype: DType, rows: Int, cols: Int
](gate: Tensor[dtype, 2], up: Tensor[dtype, 2]) -> Tuple[
    Tensor[dtype, 2], List[Tensor[dtype, 2]]
]:
    var out = swiglu_cpu[dtype, rows, cols](gate, up)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(gate)
    saved.append(up)
    return (out, saved^)


def _silu_deriv(x: Float32, silu_x: Float32) -> Float32:
    """d/dx silu(x) = sigmoid(x) * (1 + x - silu(x)), with overflow guards."""
    if x < Float32(-20.0):
        return Float32(0.0)
    if x > Float32(20.0):
        return Float32(1.0)
    var sig = Float32(1.0) / (Float32(1.0) + exp(-x))
    return sig * (Float32(1.0) + x - silu_x)


def swiglu_cpu_backward[
    dtype: DType, rows: Int, cols: Int
](grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]) -> List[
    Tensor[dtype, 2]
]:
    """Backward for SwiGLU: out = silu(gate) * up.

    grad_up   = grad_out * silu(gate)
    grad_gate = grad_out * up * silu'(gate)
    `saved` = [gate, up].
    """
    var gate = saved[0]
    var up = saved[1]
    var n = grad_out.numel()
    var grad_gate = tensor_zeros[dtype, 2](grad_out.shape())
    var grad_up = tensor_zeros[dtype, 2](grad_out.shape())
    for i in range(n):
        var g = Float32(gate.get(i))
        var u = Float32(up.get(i))
        var go = Float32(grad_out.get(i))
        var silu_g = _silu(g)
        grad_gate.set(i, Scalar[dtype](go * u * _silu_deriv(g, silu_g)))
        grad_up.set(i, Scalar[dtype](go * silu_g))
    var result = List[Tensor[dtype, 2]]()
    result.append(grad_gate)
    result.append(grad_up)
    return result^
