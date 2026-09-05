# core/ops/cpu/rope_cpu.mojo
#
# Rotary position embedding, GPT-NeoX pairing (the layout Qwen2 uses - see
# llama.cpp `llama_model_rope_type`: LLM_ARCH_QWEN2 -> NEOX).
#
# Pairs are (d, d + head_dim/2) for d in [0, head_dim/2); the frequency for
# pair `d` is theta^(-2d/head_dim), identical to HF `Qwen2RotaryEmbedding`
# (inv_freq = 1.0 / theta ** (arange(0, dim, 2) / dim)).
#
# `x` is [n_heads, T, head_dim]; positions are start_pos + row index.  All
# math is f32; f16 tensors are widened element-wise and cast back on store.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.math import cos, sin, exp, log


def _rope_cpu_kernel[
    dtype: DType
](x: Tensor[dtype, 3], start_pos: Int, theta: Float32) -> Tensor[dtype, 3]:
    var n_heads = x.shape()[0]
    var n_tokens = x.shape()[1]
    var head_dim = x.shape()[2]
    var half = head_dim // 2
    var out = tensor_zeros[dtype, 3](x.shape())

    var ln_theta = log(theta)
    for h in range(n_heads):
        for t in range(n_tokens):
            var pos = Float32(start_pos + t)
            for d in range(half):
                var freq = exp(Float32(-2 * d) / Float32(head_dim) * ln_theta)
                var angle = pos * freq
                var c = cos(angle)
                var s = sin(angle)
                var x0 = Float32(x.get((h * n_tokens + t) * head_dim + d))
                var x1 = Float32(
                    x.get((h * n_tokens + t) * head_dim + d + half)
                )
                out.set(
                    (h * n_tokens + t) * head_dim + d,
                    Scalar[dtype](x0 * c - x1 * s),
                )
                out.set(
                    (h * n_tokens + t) * head_dim + d + half,
                    Scalar[dtype](x0 * s + x1 * c),
                )
    return out


def rope_cpu[
    dtype: DType, n_heads: Int, head_dim: Int
](
    x: Tensor[dtype, 3], start_pos: Int, theta: Float32 = Float32(10000.0)
) -> Tensor[dtype, 3]:
    """Comptime-shaped RoPE (n_heads/head_dim are constants)."""
    if x.shape()[0] != n_heads or x.shape()[2] != head_dim:
        unimplemented("rope_cpu: static shape mismatch")
    return _rope_cpu_kernel[dtype](x, start_pos, theta)


def rope_cpu_dynamic[
    dtype: DType
](
    x: Tensor[dtype, 3], start_pos: Int, theta: Float32 = Float32(10000.0)
) -> Tensor[dtype, 3]:
    """Runtime-shaped RoPE."""
    return _rope_cpu_kernel[dtype](x, start_pos, theta)


def rope_cpu_rot[
    dtype: DType
](
    x: Tensor[dtype, 3],
    start_pos: Int,
    theta: Float32,
    n_rot: Int,
) -> Tensor[
    dtype, 3
]:
    """Partial RoPE: rotate the first `n_rot` dims, pass the rest through.

    qwen35's MRoPE rotates only `rope.dimension_count` (64) of the 256
    head dims; for text-only input every section uses the same position,
    so MRoPE degenerates to plain NeoX RoPE over `n_rot` dims.
    """
    var n_heads = x.shape()[0]
    var n_tokens = x.shape()[1]
    var head_dim = x.shape()[2]
    if n_rot <= 0 or n_rot >= head_dim:
        return _rope_cpu_kernel[dtype](x, start_pos, theta)
    var half = n_rot // 2
    var out = tensor_zeros[dtype, 3](x.shape())
    var ln_theta = log(theta)
    for h in range(n_heads):
        for t in range(n_tokens):
            var pos = Float32(start_pos + t)
            var base = (h * n_tokens + t) * head_dim
            # copy the unrotated tail
            for d in range(n_rot, head_dim):
                out.set(base + d, x.get(base + d))
            for d in range(half):
                var freq = exp(Float32(-2 * d) / Float32(n_rot) * ln_theta)
                var angle = pos * freq
                var c = cos(angle)
                var s = sin(angle)
                var x0 = Float32(x.get(base + d))
                var x1 = Float32(x.get(base + d + half))
                out.set(base + d, Scalar[dtype](x0 * c - x1 * s))
                out.set(base + d + half, Scalar[dtype](x0 * s + x1 * c))
    return out


def rope_cpu_forward_with_saved[
    dtype: DType, n_heads: Int, head_dim: Int
](
    x: Tensor[dtype, 3],
    start_pos: Int,
    theta: Float32 = Float32(10000.0),
) -> Tuple[Tensor[dtype, 3], List[Tensor[dtype, 3]]]:
    var out = rope_cpu[dtype, n_heads, head_dim](x, start_pos, theta)
    var saved = List[Tensor[dtype, 3]]()
    saved.append(x)
    return (out, saved^)


def rope_cpu_backward[
    dtype: DType, n_heads: Int, head_dim: Int
](
    grad_out: Tensor[dtype, 3],
    x: Tensor[dtype, 3],
    start_pos: Int,
    theta: Float32 = Float32(10000.0),
) -> Tensor[dtype, 3]:
    """Backward for RoPE: apply the inverse (transpose) rotation to
    `grad_out`.

    Forward pairs (d, d+half) as  y0 = x0*c - x1*s,  y1 = x0*s + x1*c, so
        grad_x0 =  grad_y0*c + grad_y1*s
        grad_x1 = -grad_y0*s + grad_y1*c
    """
    var n_tokens = grad_out.shape()[1]
    var hd = grad_out.shape()[2]
    var half = hd // 2
    var grad_x = tensor_zeros[dtype, 3](grad_out.shape())
    var ln_theta = log(theta)
    for h in range(grad_out.shape()[0]):
        for t in range(n_tokens):
            var pos = Float32(start_pos + t)
            var base = (h * n_tokens + t) * hd
            for d in range(half):
                var freq = exp(Float32(-2 * d) / Float32(hd) * ln_theta)
                var angle = pos * freq
                var c = cos(angle)
                var s = sin(angle)
                var g0 = Float32(grad_out.get(base + d))
                var g1 = Float32(grad_out.get(base + d + half))
                grad_x.set(base + d, Scalar[dtype](g0 * c + g1 * s))
                grad_x.set(base + d + half, Scalar[dtype](-g0 * s + g1 * c))
    _ = x
    return grad_x
