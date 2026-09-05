# core/sampler.mojo
#
# Token sampling strategies for generation.

from .tensor import Tensor
from std.math import exp
from std.ffi import external_call
import std.random


def seed_sampler(seed: Optional[Int] = None):
    """Seed the global RNG.

    With an explicit seed the draw is reproducible; without one the seed
    comes from wall-clock time (`time(2)` via libc) so independent runs
    sample differently - Mojo's default RNG seed is fixed, which would
    otherwise replay the same token stream on every run.
    """
    if seed:
        std.random.seed(seed.value())
    else:
        var t = external_call["time", Int](0)
        std.random.seed(t)


struct Sampler(Copyable, ImplicitlyCopyable, Movable):
    var temperature: Float32
    var top_k: Int
    var top_p: Float32
    var min_p: Float32
    var repetition_penalty: Float32

    def __init__(
        out self,
        temperature: Float32 = Float32(1.0),
        top_k: Int = 0,
        top_p: Float32 = Float32(1.0),
        min_p: Float32 = Float32(0.0),
        repetition_penalty: Float32 = Float32(1.0),
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty


def _topk_partial(probs: List[Float32], limit: Int) -> List[Int]:
    """Return up to `limit` token indices in descending probability order.

    Partial selection (O(n * limit)); the sampler only ever needs the head
    of the distribution (top_k entries, or enough mass for top_p), so the
    M1 full O(n^2) sort was replaced in M3 - sorting 151936 entries per
    token was the generation-loop bottleneck.
    """
    var n = len(probs)
    var order = List[Int]()
    var used = List[Bool](length=n, fill=False)
    var count = 0
    while count < limit and count < n:
        var best = -1
        var best_v = Float32(-3.0e38)
        for i in range(n):
            if not used[i] and probs[i] > best_v:
                best_v = probs[i]
                best = i
        if best < 0:
            break
        used[best] = True
        order.append(best)
        count += 1
    return order^


def sample[
    dtype: DType, vocab_size: Int
](
    logits: Tensor[dtype, 1],
    sampler: Sampler,
    generated_tokens: List[Int],
) -> Int where dtype.is_floating_point():
    """Sample one token from `logits` after applying the sampler's filters."""
    var probs = List[Float32]()

    # temperature + repetition penalty, then softmax.
    for i in range(vocab_size):
        var value = Float32(logits.get(i))
        if sampler.temperature > 0:
            value = value / sampler.temperature
        if sampler.repetition_penalty != Float32(1.0):
            var repeated = False
            for token in generated_tokens:
                if token == i:
                    repeated = True
                    break
            if repeated:
                if value < 0:
                    value = value * sampler.repetition_penalty
                else:
                    value = value / sampler.repetition_penalty
        probs.append(value)

    return _sample_from_probs(probs, sampler)


def sample_dynamic[
    dtype: DType
](
    logits: Tensor[dtype, 1],
    sampler: Sampler,
    generated_tokens: List[Int],
) -> Int where dtype.is_floating_point():
    """Runtime-vocab sampling (vocab size read from the logits tensor)."""
    var vocab_size = logits.shape()[0]
    var probs = List[Float32]()
    for i in range(vocab_size):
        var value = Float32(logits.get(i))
        if sampler.temperature > 0:
            value = value / sampler.temperature
        if sampler.repetition_penalty != Float32(1.0):
            var repeated = False
            for token in generated_tokens:
                if token == i:
                    repeated = True
                    break
            if repeated:
                if value < 0:
                    value = value * sampler.repetition_penalty
                else:
                    value = value / sampler.repetition_penalty
        probs.append(value)
    return _sample_from_probs(probs, sampler)


def _sample_from_probs(mut probs: List[Float32], sampler: Sampler) -> Int:
    """Shared softmax + filters + CDF sampling over a probability list."""
    var vocab_size = len(probs)
    var mx = probs[0]
    for i in range(1, vocab_size):
        if probs[i] > mx:
            mx = probs[i]
    var total = Float32(0)
    for i in range(vocab_size):
        var p = exp(probs[i] - mx)
        probs[i] = p
        total += p
    for i in range(vocab_size):
        probs[i] = probs[i] / total

    # min_p filter.
    if sampler.min_p > 0:
        var pmax = probs[0]
        for i in range(1, vocab_size):
            if probs[i] > pmax:
                pmax = probs[i]
        var threshold = sampler.min_p * pmax
        for i in range(vocab_size):
            if probs[i] < threshold:
                probs[i] = 0

    # top_k and top_p filters over a descending index order.
    if (
        sampler.top_k > 0 and sampler.top_k < vocab_size
    ) or sampler.top_p < Float32(1.0):
        # Only the head of the distribution matters: with top_k set we never
        # need more than top_k ordered entries; without it, top_p mass on a
        # real LM concentrates in a few hundred tokens (2048 is the safety
        # cap for pathological distributions).
        var limit = vocab_size
        if sampler.top_k > 0 and sampler.top_k < vocab_size:
            limit = sampler.top_k
        elif limit > 2048:
            limit = 2048
        var order = _topk_partial(probs, limit)
        var kept = List[Bool]()
        for _ in range(vocab_size):
            kept.append(False)
        var cumulative = Float32(0)
        for rank in range(len(order)):
            var token = order[rank]
            var keep = True
            if sampler.top_k > 0 and rank >= sampler.top_k:
                keep = False
            if keep and sampler.top_p < Float32(1.0):
                cumulative += probs[token]
                # llama.cpp semantics: the top-1 token is always kept even
                # when its mass alone exceeds top_p; the token that pushes
                # the cumulative past top_p (rank > 0) is dropped along
                # with everything after it.
                if cumulative > sampler.top_p and rank > 0:
                    keep = False
            if keep:
                kept[token] = True
        for i in range(vocab_size):
            if not kept[i]:
                probs[i] = 0

    # renormalize and sample from the CDF.
    var norm = Float32(0)
    for i in range(vocab_size):
        norm += probs[i]
    if norm <= 0:
        var fallback = 0
        for i in range(1, vocab_size):
            if probs[i] > probs[fallback]:
                fallback = i
        return fallback
    var draw = Float32(std.random.random_float64())
    var cumulative = Float32(0)
    for i in range(vocab_size):
        cumulative += probs[i] / norm
        if draw < cumulative:
            return i
    return vocab_size - 1


def greedy_sample[
    dtype: DType, vocab_size: Int
](logits: Tensor[dtype, 1]) -> Int:
    """Argmax over the logits (no randomness)."""
    var best = 0
    var best_value = logits.get(0)
    for i in range(1, vocab_size):
        var value = logits.get(i)
        if value > best_value:
            best_value = value
            best = i
    return best
