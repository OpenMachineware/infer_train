# core/ops/gpu/embedding_gpu.mojo
#
# GPU token embedding lookup (gather).  The table is [vocab, hidden]
# (vocab-major, the GGUF `token_embd.weight` layout); the embedding of token
# `t` is row `t`.  Out is [T, hidden] for a [T] token vector.
#
# The forward is a clean gather and runs on the GPU.  The backward is a sparse
# scatter-add over the (usually huge) table; it stays on the CPU kernel, which
# only touches the rows of tokens actually present in the batch.

from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.embedding_cpu import (
    embedding_cpu,
    embedding_cpu_dynamic,
    embedding_cpu_backward,
)
from .gpu_runtime import (
    download2,
    get_gpu_context,
    grid1d,
    gpu_available,
    upload,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import global_idx, grid_dim, block_dim
from std.memory import Pointer
from std.origin import MutAnyOrigin
from std.utils.static_tuple import StaticTuple

comptime BLOCK = 256


# -- kernels ------------------------------------------------------------------


def _embedding_kernel_f32(
    tokens: Pointer[Int32, MutAnyOrigin],
    table: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    hidden: Int32,
    vocab: Int32,
    n_tokens: Int32,
):
    var hidden_i = Int(hidden)
    var vocab_i = Int(vocab)
    var n_tok_i = Int(n_tokens)
    var n = n_tok_i * hidden_i
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n:
        var t = i // hidden_i
        var k = i % hidden_i
        var v: Float32 = Float32(0.0)
        if t < n_tok_i:
            var token = Int(tokens[unsafe_offset=t])
            if token >= 0 and token < vocab_i:
                v = table[unsafe_offset=token * hidden_i + k]
        dst[unsafe_offset=i] = v
        i += stride


def _embedding_kernel_f16(
    tokens: Pointer[Int32, MutAnyOrigin],
    table: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    hidden: Int32,
    vocab: Int32,
    n_tokens: Int32,
):
    var hidden_i = Int(hidden)
    var vocab_i = Int(vocab)
    var n_tok_i = Int(n_tokens)
    var n = n_tok_i * hidden_i
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n:
        var t = i // hidden_i
        var k = i % hidden_i
        var v: Scalar[DType.float16] = Scalar[DType.float16](0.0)
        if t < n_tok_i:
            var token = Int(tokens[unsafe_offset=t])
            if token >= 0 and token < vocab_i:
                v = table[unsafe_offset=token * hidden_i + k]
        dst[unsafe_offset=i] = v
        i += stride


# -- launch helpers -----------------------------------------------------------


def _embedding_gpu_launch[
    dtype: DType
](
    ctx: DeviceContext, tokens: Tensor[DType.int32, 1], table: Tensor[dtype, 2]
) raises -> Tensor[dtype, 2]:
    var n_tokens = tokens.shape()[0]
    var vocab = table.shape()[0]
    var hidden = table.shape()[1]
    var tok_buf = upload[DType.int32, 1](ctx, tokens)
    var table_buf = upload[dtype, 2](ctx, table)
    var n = n_tokens * hidden
    var dst_buf = ctx.enqueue_create_buffer[dtype](n)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_embedding_kernel_f16](
            tok_buf,
            table_buf,
            dst_buf,
            Int32(hidden),
            Int32(vocab),
            Int32(n_tokens),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_embedding_kernel_f32](
            tok_buf,
            table_buf,
            dst_buf,
            Int32(hidden),
            Int32(vocab),
            Int32(n_tokens),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    var out = download2[dtype](
        ctx, dst_buf, StaticTuple[Int, 2](n_tokens, hidden)
    )
    ctx.synchronize()
    return out


# -- public entry points ------------------------------------------------------


def embedding_gpu[
    dtype: DType, hidden: Int, vocab: Int
](tokens: Tensor[DType.int32, 1], table: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Comptime-shaped GPU embedding lookup (CPU fallback on any GPU error)."""
    if table.shape() != StaticTuple[Int, 2](vocab, hidden):
        unimplemented("embedding_gpu: static shape mismatch")
    if not gpu_available[dtype]():
        return embedding_cpu[dtype, hidden, vocab](tokens, table)
    try:
        var ctx = get_gpu_context()
        return _embedding_gpu_launch[dtype](ctx, tokens, table)
    except:
        return embedding_cpu[dtype, hidden, vocab](tokens, table)


def embedding_gpu_dynamic[
    dtype: DType
](tokens: Tensor[DType.int32, 1], table: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Runtime-shaped GPU embedding lookup (CPU fallback on any GPU error)."""
    if not gpu_available[dtype]():
        return embedding_cpu_dynamic[dtype](tokens, table)
    try:
        var ctx = get_gpu_context()
        return _embedding_gpu_launch[dtype](ctx, tokens, table)
    except:
        return embedding_cpu_dynamic[dtype](tokens, table)


def embedding_gpu_forward_with_saved[
    dtype: DType, hidden: Int, vocab: Int
](tokens: Tensor[DType.int32, 1], table: Tensor[dtype, 2]) -> Tuple[
    Tensor[dtype, 2], List[Tensor[dtype, 2]]
]:
    var out = embedding_gpu[dtype, hidden, vocab](tokens, table)
    # only the table is rank-2; the tokens ride along in the registry's
    # erased saved list (see op_registry.embedding).
    var saved = List[Tensor[dtype, 2]]()
    saved.append(table)
    return (out, saved^)


def embedding_gpu_backward[
    dtype: DType, hidden: Int, vocab: Int
](grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]) -> List[
    Tensor[dtype, 2]
]:
    """Backward for the token embedding lookup (sparse update).

    Kept on the CPU kernel: it is a scatter-add over the (usually huge) table
    that only touches the rows of tokens present in the batch.  `saved`
    holds [table].
    """
    var table = saved[0]
    # The token ids are not part of the rank-2 saved list; reconstruct them is
    # the registry's job (see op_autograd.embedding_bwd_*), so this typed entry
    # only handles the table gradient via the CPU kernel.
    _ = grad_out
    _ = hidden
    _ = vocab
    unimplemented("embedding_gpu_backward: use the erased dispatcher")
    return List[Tensor[dtype, 2]]()
