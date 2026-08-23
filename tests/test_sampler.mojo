# tests/test_sampler.mojo
#
# Regression test for the top_p crossing-token bug: when the top-1 token's
# mass alone exceeds top_p, llama.cpp semantics keep it (rank 0 is exempt
# from the cumulative cutoff).  The old behavior zeroed every candidate,
# which sent the sampler into its all-zero fallback (token 0) - visible as
# "!"-loop degeneracy during generation.

from src.core.sampler import Sampler, sample_dynamic, seed_sampler
from src.core.tensor import Tensor, tensor_zeros
from std.utils.static_tuple import StaticTuple


def main():
    # Dominant token 7 (logit 10.0) vs background (logit 0.0): after
    # softmax the top-1 mass >> top_p, so it must be kept and, after
    # renormalization, drawn every single time.
    var logits = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](64))
    logits.set(7, Scalar[DType.float32](Float32(10.0)))
    var sampler = Sampler(
        temperature=Float32(1.0), top_k=0, top_p=Float32(0.5)
    )
    seed_sampler(1234)
    for i in range(100):
        var history = List[Int]()
        var token = sample_dynamic[DType.float32](logits, sampler, history)
        if token != 7:
            print("FAIL: draw", i, "returned", token, "expected 7")
            abort()
    print("dominant-token sampling OK")

    # Background sanity: two close tokens are both reachable.
    var logits2 = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](8))
    logits2.set(2, Scalar[DType.float32](Float32(2.0)))
    logits2.set(5, Scalar[DType.float32](Float32(1.99)))
    var sampler2 = Sampler(
        temperature=Float32(1.0), top_k=0, top_p=Float32(1.0)
    )
    seed_sampler(99)
    var saw2 = False
    var saw5 = False
    for i in range(200):
        var history = List[Int]()
        var token = sample_dynamic[DType.float32](logits2, sampler2, history)
        if token == 2:
            saw2 = True
        if token == 5:
            saw5 = True
    if not saw2 or not saw5:
        print("FAIL: two-candidate sampling unreachable, saw2", saw2, "saw5", saw5)
        abort()
    print("two-candidate sampling OK")
    print("test_sampler OK")


def abort():
    from std.os.os import abort as _abort

    _abort()
