# core/ops/loss/cross_entropy.mojo
#
# Cross-entropy loss over logits (M6): forward and backward.
#
#   forward: loss = mean_i -log(softmax(logits[i])[target_i])
#   backward: grad_logits[i, j] = (softmax[i, j] - onehot(target_i)[j]) / B
#
# Both are computed with the log-sum-exp trick for numerical stability.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.utils.static_tuple import StaticTuple
from std.math import log, exp


def cross_entropy_loss[
    dtype: DType, vocab_size: Int
](logits: Tensor[dtype, 2], targets: Tensor[DType.int32, 1]) -> Scalar[dtype]:
    """Mean cross-entropy loss over the batch (scalar return)."""
    var batch = logits.shape()[0]
    var vocab = logits.shape()[1]
    if batch < 1 or batch != targets.shape()[0]:
        unimplemented("cross_entropy_loss: batch mismatch")
    var total = Float32(0)
    for i in range(batch):
        var mx = Float32(logits.get(i * vocab))
        for j in range(vocab):
            var v = Float32(logits.get(i * vocab + j))
            if v > mx:
                mx = v
        var sum_exp = Float32(0)
        for j in range(vocab):
            sum_exp += exp(Float32(logits.get(i * vocab + j)) - mx)
        var lse = mx + log(sum_exp)
        var target = Int(targets.get(i))
        var lp = Float32(0)
        if target >= 0 and target < vocab:
            lp = Float32(logits.get(i * vocab + target))
        total += lse - lp
    return Scalar[dtype](total / Float32(batch))


def cross_entropy_forward[
    dtype: DType
](logits: Tensor[dtype, 2], targets: Tensor[DType.int32, 1]) -> Tensor[
    dtype, 1
]:
    """Loss as a rank-1 [1] tensor (the registry/autograd shape)."""
    var out = tensor_zeros[dtype, 1](StaticTuple[Int, 1](1))
    out.set(0, cross_entropy_loss[dtype, 0](logits, targets))
    return out


def cross_entropy_backward[
    dtype: DType
](
    grad_loss: Tensor[dtype, 1],
    logits: Tensor[dtype, 2],
    targets: Tensor[DType.int32, 1],
) -> Tensor[dtype, 2]:
    """grad_logits = (softmax(logits) - onehot(targets)) / batch * grad_loss.

    Stable softmax (subtract the row max).  `grad_loss` is the scalar
    upstream gradient (rank-1 [1] tensor).
    """
    var batch = logits.shape()[0]
    var vocab = logits.shape()[1]
    var grad_logits = tensor_zeros[dtype, 2](logits.shape())
    var g = Float32(grad_loss.get(0)) / Float32(batch)
    for i in range(batch):
        var base = i * vocab
        var mx = Float32(logits.get(base))
        for j in range(vocab):
            var v = Float32(logits.get(base + j))
            if v > mx:
                mx = v
        var sum_exp = Float32(0)
        for j in range(vocab):
            sum_exp += exp(Float32(logits.get(base + j)) - mx)
        var target = Int(targets.get(i))
        for j in range(vocab):
            var p = exp(Float32(logits.get(base + j)) - mx) / sum_exp
            var onehot = Float32(0)
            if j == target:
                onehot = Float32(1.0)
            grad_logits.set(base + j, Scalar[dtype]((p - onehot) * g))
    return grad_logits
