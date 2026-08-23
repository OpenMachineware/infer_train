# core/ops/cpu/embedding_cpu.mojo
#
# Token embedding lookup.  GGUF stores `token_embd.weight` as
# [hidden, vocab] (transposed vs. HF), so the embedding of token `t` is
# *column* `t` of the table.  Out is [T, hidden] for a [T] token vector.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.utils.static_tuple import StaticTuple


def _embedding_cpu_kernel[dtype: DType](
    tokens: Tensor[DType.int32, 1], table: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Look up token embeddings from a [vocab, hidden] table (row = token).

    GGUF `token_embd.weight` is vocab-major on disk (dims are ggml-ordered),
    so the dequantized table is [vocab, hidden] and the embedding of token
    `t` is row `t`.
    """
    var n_tokens = tokens.shape()[0]
    var vocab = table.shape()[0]
    var hidden = table.shape()[1]
    var out = tensor_zeros[dtype, 2](
        StaticTuple[Int, 2](n_tokens, hidden)
    )
    for t in range(n_tokens):
        var token = Int(tokens.get(t))
        if token < 0 or token >= vocab:
            continue  # out-of-range: leave zeros (caller logs)
        for k in range(hidden):
            out.set(t * hidden + k, table.get(token * hidden + k))
    return out


def embedding_cpu[dtype: DType, hidden: Int, vocab: Int](
    tokens: Tensor[DType.int32, 1], table: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Comptime-shaped embedding lookup (hidden/vocab are constants)."""
    if table.shape()[0] != hidden or table.shape()[1] != vocab:
        unimplemented("embedding_cpu: static shape mismatch")
    return _embedding_cpu_kernel[dtype](tokens, table)


def embedding_cpu_dynamic[dtype: DType](
    tokens: Tensor[DType.int32, 1], table: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Runtime-shaped embedding lookup."""
    return _embedding_cpu_kernel[dtype](tokens, table)


def embedding_cpu_forward_with_saved[dtype: DType, hidden: Int, vocab: Int](
    tokens: Tensor[DType.int32, 1], table: Tensor[dtype, 2]
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = embedding_cpu[dtype, hidden, vocab](tokens, table)
    # only the table is rank-2; the tokens ride along in the registry's
    # erased saved list (see op_registry.embedding).
    var saved = List[Tensor[dtype, 2]]()
    saved.append(table)
    return (out, saved^)


def embedding_cpu_backward[dtype: DType, hidden: Int, vocab: Int](
    grad_out: Tensor[dtype, 2],
    tokens: Tensor[DType.int32, 1],
    table: Tensor[dtype, 2],
) -> Tensor[dtype, 2]:
    """Backward for the token embedding lookup (sparse update).

    grad_table[token_t, :] += grad_out[t, :]; there is no gradient w.r.t.
    the integer token ids.  Out-of-range tokens contribute nothing.
    """
    var n_tokens = tokens.shape()[0]
    var vocab_sz = table.shape()[0]
    var hidden_sz = table.shape()[1]
    var grad_table = tensor_zeros[dtype, 2](table.shape())
    for t in range(n_tokens):
        var token = Int(tokens.get(t))
        if token < 0 or token >= vocab_sz:
            continue
        for k in range(hidden_sz):
            grad_table.set(
                token * hidden_sz + k,
                Scalar[dtype](
                    Float32(grad_table.get(token * hidden_sz + k))
                    + Float32(grad_out.get(t * hidden_sz + k))
                ),
            )
    _ = vocab
    return grad_table
