# tests/test_backward.mojo
#
# M6 Phase 8: gradient checks for every core operator's backward.
#
# Each op is checked with central finite differences: loss = sum(grad_out *
# forward(x)), and the analytic backward must match
# (loss(x+eps) - loss(x-eps)) / (2*eps) per element.  The weighted norm,
# SwiGLU-FFN, cross-entropy and stateless MHA are exercised through the
# *registry* erased path (forward_with_saved + backward), which also covers
# the Interpreter's autograd plumbing; `test_interpreter_run_with_grad`
# checks the reverse traversal + gradient accumulation directly.

from src.core.tensor import Tensor, tensor_zeros, tensor_copy
from src.core.ops.cpu.matmul_cpu import (
    matmul_cpu_dynamic,
    matmul_cpu_forward_with_saved,
    matmul_cpu_backward,
    matmul_weight_cpu,
    matmul_weight_cpu_backward,
)
from src.core.ops.cpu.add_cpu import (
    add_cpu_dynamic,
    add_cpu_backward,
    add_row_cpu,
    add_row_cpu_backward,
)
from src.core.ops.cpu.rms_norm_cpu import (
    rms_norm_cpu_dynamic,
    rms_norm_cpu_forward_with_saved,
    rms_norm_cpu_backward,
    rms_norm_weight_cpu_forward_with_saved,
    rms_norm_weight_cpu_backward,
)
from src.core.ops.cpu.softmax_cpu import (
    softmax_cpu_dynamic,
    softmax_cpu_forward_with_saved,
    softmax_cpu_backward,
)
from src.core.ops.cpu.rope_cpu import rope_cpu_dynamic, rope_cpu_backward
from src.core.ops.cpu.swiglu_cpu import (
    swiglu_cpu_dynamic,
    swiglu_cpu_backward,
)
from src.core.ops.cpu.embedding_cpu import (
    embedding_cpu_dynamic,
    embedding_cpu_backward,
)
from src.core.ops.loss.cross_entropy import (
    cross_entropy_loss,
    cross_entropy_forward,
    cross_entropy_backward,
)
from src.core.ops.base.op_registry import OpRegistry
from src.core.ops.base.op_interface import AnyTensor, to_any, from_any
from src.core.graph import Graph, AttrValue
from src.runtime.interpreter import Interpreter
from src.core.ops.base.op_autograd import ones_like_any, no_grad_any
from std.utils.static_tuple import StaticTuple
from std.os.os import abort

comptime EPS = Float32(1e-3)


# -- helpers ------------------------------------------------------------------


def _rnd(mut state: Int) -> Float32:
    state = (state * 1103515245 + 12345) % 2147483648
    return (Float32(state) / Float32(2147483648.0)) * 2.0 - 1.0


def _fill2(mut t: Tensor[DType.float32, 2], mut state: Int, scale: Float32):
    for i in range(t.numel()):
        t.set(i, Scalar[DType.float32](_rnd(state) * scale))


def _fill1(mut t: Tensor[DType.float32, 1], mut state: Int, scale: Float32):
    for i in range(t.numel()):
        t.set(i, Scalar[DType.float32](_rnd(state) * scale))


def _fill3(mut t: Tensor[DType.float32, 3], mut state: Int, scale: Float32):
    for i in range(t.numel()):
        t.set(i, Scalar[DType.float32](_rnd(state) * scale))


def _dot2(a: Tensor[DType.float32, 2], b: Tensor[DType.float32, 2]) -> Float32:
    var total = Float32(0)
    for i in range(a.numel()):
        total += Float32(a.get(i)) * Float32(b.get(i))
    return total


def _dot1(a: Tensor[DType.float32, 1], b: Tensor[DType.float32, 1]) -> Float32:
    var total = Float32(0)
    for i in range(a.numel()):
        total += Float32(a.get(i)) * Float32(b.get(i))
    return total


def _dot3(a: Tensor[DType.float32, 3], b: Tensor[DType.float32, 3]) -> Float32:
    var total = Float32(0)
    for i in range(a.numel()):
        total += Float32(a.get(i)) * Float32(b.get(i))
    return total


def check_close(
    actual: Float32, expected: Float32, tol: Float32, label: String
):
    var diff = actual - expected
    if diff < 0:
        diff = -diff
    if diff > tol:
        print(
            "FAIL",
            label,
            "actual:",
            actual,
            "expected:",
            expected,
            "diff:",
            diff,
        )
        abort()


def _one(a: Int) -> List[Int]:
    var l = List[Int]()
    l.append(a)
    return l^


def _two(a: Int, b: Int) -> List[Int]:
    var l = List[Int]()
    l.append(a)
    l.append(b)
    return l^


def main():
    test_matmul()
    test_matmul_weight()
    test_add()
    test_add_bias()
    test_rms_norm()
    test_rms_norm_weight()
    test_softmax()
    test_rope()
    test_swiglu()
    test_embedding()
    test_swiglu_ffn_registry()
    test_cross_entropy()
    test_mha_seq_registry()
    test_interpreter_run_with_grad()
    print("test_backward OK")


# -- matmul -------------------------------------------------------------------


def test_matmul():
    var state = 1
    var a = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    var b = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](3, 2))
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 2))
    _fill2(a, state, 0.5)
    _fill2(b, state, 0.5)
    _fill2(go, state, 0.5)
    var fws = matmul_cpu_forward_with_saved[DType.float32](a, b)
    var grads = matmul_cpu_backward[DType.float32](go, fws[1])
    # numeric check on a
    for i in range(a.numel()):
        var ap = tensor_copy(a)
        var am = tensor_copy(a)
        ap.set(i, Scalar[DType.float32](Float32(a.get(i)) + EPS))
        am.set(i, Scalar[DType.float32](Float32(a.get(i)) - EPS))
        var numeric = (
            _dot2(matmul_cpu_dynamic[DType.float32](ap, b), go)
            - _dot2(matmul_cpu_dynamic[DType.float32](am, b), go)
        ) / (2.0 * EPS)
        check_close(
            Float32(grads[0].get(i)), numeric, Float32(1e-2), "matmul grad_a"
        )
    # numeric check on b
    for i in range(b.numel()):
        var bp = tensor_copy(b)
        var bm = tensor_copy(b)
        bp.set(i, Scalar[DType.float32](Float32(b.get(i)) + EPS))
        bm.set(i, Scalar[DType.float32](Float32(b.get(i)) - EPS))
        var numeric = (
            _dot2(matmul_cpu_dynamic[DType.float32](a, bp), go)
            - _dot2(matmul_cpu_dynamic[DType.float32](a, bm), go)
        ) / (2.0 * EPS)
        check_close(
            Float32(grads[1].get(i)), numeric, Float32(1e-2), "matmul grad_b"
        )


def test_matmul_weight():
    var state = 2
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    var w = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](4, 3))
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 4))
    _fill2(x, state, 0.5)
    _fill2(w, state, 0.5)
    _fill2(go, state, 0.5)
    var saved = List[Tensor[DType.float32, 2]]()
    saved.append(x)
    saved.append(w)
    var grads = matmul_weight_cpu_backward[DType.float32](go, saved)
    for i in range(x.numel()):
        var xp = tensor_copy(x)
        var xm = tensor_copy(x)
        xp.set(i, Scalar[DType.float32](Float32(x.get(i)) + EPS))
        xm.set(i, Scalar[DType.float32](Float32(x.get(i)) - EPS))
        var numeric = (
            _dot2(matmul_weight_cpu[DType.float32](xp, w), go)
            - _dot2(matmul_weight_cpu[DType.float32](xm, w), go)
        ) / (2.0 * EPS)
        check_close(
            Float32(grads[0].get(i)), numeric, Float32(1e-2), "mw grad_x"
        )
    for i in range(w.numel()):
        var wp = tensor_copy(w)
        var wm = tensor_copy(w)
        wp.set(i, Scalar[DType.float32](Float32(w.get(i)) + EPS))
        wm.set(i, Scalar[DType.float32](Float32(w.get(i)) - EPS))
        var numeric = (
            _dot2(matmul_weight_cpu[DType.float32](x, wp), go)
            - _dot2(matmul_weight_cpu[DType.float32](x, wm), go)
        ) / (2.0 * EPS)
        check_close(
            Float32(grads[1].get(i)), numeric, Float32(1e-2), "mw grad_w"
        )


def test_add():
    var state = 3
    var a = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    var b = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    _fill2(a, state, 0.5)
    _fill2(b, state, 0.5)
    _fill2(go, state, 0.5)
    var saved = List[Tensor[DType.float32, 2]]()
    saved.append(a)
    saved.append(b)
    var grads = add_cpu_backward[DType.float32, 2, 3](go, saved)
    for i in range(go.numel()):
        check_close(
            Float32(grads[0].get(i)),
            Float32(go.get(i)),
            Float32(1e-6),
            "add gx",
        )
        check_close(
            Float32(grads[1].get(i)),
            Float32(go.get(i)),
            Float32(1e-6),
            "add gy",
        )


def test_add_bias():
    var state = 4
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    var bias = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](3))
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    _fill2(x, state, 0.5)
    _fill1(bias, state, 0.5)
    _fill2(go, state, 0.5)
    var grads = add_row_cpu_backward[DType.float32](go, bias)
    for i in range(x.numel()):
        check_close(
            Float32(grads[0].get(i)), Float32(go.get(i)), Float32(1e-6), "ab gx"
        )
    for j in range(3):
        var expect = Float32(go.get(j)) + Float32(go.get(3 + j))
        check_close(Float32(grads[1].get(j)), expect, Float32(1e-6), "ab gbias")


def test_rms_norm():
    var state = 5
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 4))
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 4))
    _fill2(x, state, 0.7)
    _fill2(go, state, 0.5)
    var fws = rms_norm_cpu_forward_with_saved[DType.float32, 4](x)
    var grads = rms_norm_cpu_backward[DType.float32, 4](go, fws[1])
    for i in range(x.numel()):
        var xp = tensor_copy(x)
        var xm = tensor_copy(x)
        xp.set(i, Scalar[DType.float32](Float32(x.get(i)) + EPS))
        xm.set(i, Scalar[DType.float32](Float32(x.get(i)) - EPS))
        var numeric = (
            _dot2(rms_norm_cpu_dynamic[DType.float32](xp), go)
            - _dot2(rms_norm_cpu_dynamic[DType.float32](xm), go)
        ) / (2.0 * EPS)
        check_close(
            Float32(grads[0].get(i)), numeric, Float32(1e-2), "rms grad_x"
        )


def test_rms_norm_weight():
    var state = 6
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 4))
    var w = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](4))
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 4))
    _fill2(x, state, 0.7)
    _fill1(w, state, 0.7)
    _fill2(go, state, 0.5)
    var fws = rms_norm_weight_cpu_forward_with_saved[DType.float32](x, w)
    var grads = rms_norm_weight_cpu_backward[DType.float32](go, fws[1], w)
    for i in range(x.numel()):
        var xp = tensor_copy(x)
        var xm = tensor_copy(x)
        xp.set(i, Scalar[DType.float32](Float32(x.get(i)) + EPS))
        xm.set(i, Scalar[DType.float32](Float32(x.get(i)) - EPS))
        var numeric = (
            _dot2(
                rms_norm_weight_cpu_forward_with_saved[DType.float32](xp, w)[0],
                go,
            )
            - _dot2(
                rms_norm_weight_cpu_forward_with_saved[DType.float32](xm, w)[0],
                go,
            )
        ) / (2.0 * EPS)
        check_close(
            Float32(grads[0].get(i)), numeric, Float32(1e-2), "rnw grad_x"
        )
    for i in range(w.numel()):
        var wp = tensor_copy(w)
        var wm = tensor_copy(w)
        wp.set(i, Scalar[DType.float32](Float32(w.get(i)) + EPS))
        wm.set(i, Scalar[DType.float32](Float32(w.get(i)) - EPS))
        var numeric = (
            _dot2(
                rms_norm_weight_cpu_forward_with_saved[DType.float32](x, wp)[0],
                go,
            )
            - _dot2(
                rms_norm_weight_cpu_forward_with_saved[DType.float32](x, wm)[0],
                go,
            )
        ) / (2.0 * EPS)
        check_close(
            Float32(grads[1].get(i)), numeric, Float32(1e-2), "rnw grad_w"
        )


def test_softmax():
    var state = 7
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 4))
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 4))
    _fill2(x, state, 0.7)
    _fill2(go, state, 0.5)
    var fws = softmax_cpu_forward_with_saved[DType.float32, 4](x)
    var grads = softmax_cpu_backward[DType.float32, 4](go, fws[1])
    for i in range(x.numel()):
        var xp = tensor_copy(x)
        var xm = tensor_copy(x)
        xp.set(i, Scalar[DType.float32](Float32(x.get(i)) + EPS))
        xm.set(i, Scalar[DType.float32](Float32(x.get(i)) - EPS))
        var numeric = (
            _dot2(softmax_cpu_dynamic[DType.float32](xp), go)
            - _dot2(softmax_cpu_dynamic[DType.float32](xm), go)
        ) / (2.0 * EPS)
        check_close(
            Float32(grads[0].get(i)), numeric, Float32(1e-2), "softmax grad_x"
        )


def test_rope():
    var state = 8
    var x = tensor_zeros[DType.float32, 3](StaticTuple[Int, 3](1, 2, 4))
    var go = tensor_zeros[DType.float32, 3](StaticTuple[Int, 3](1, 2, 4))
    _fill3(x, state, 0.7)
    _fill3(go, state, 0.5)
    var grads = rope_cpu_backward[DType.float32, 0, 0](
        go, x, 1, Float32(10000.0)
    )
    for i in range(x.numel()):
        var xp = tensor_copy(x)
        var xm = tensor_copy(x)
        xp.set(i, Scalar[DType.float32](Float32(x.get(i)) + EPS))
        xm.set(i, Scalar[DType.float32](Float32(x.get(i)) - EPS))
        var numeric = (
            _dot3(rope_cpu_dynamic[DType.float32](xp, 1), go)
            - _dot3(rope_cpu_dynamic[DType.float32](xm, 1), go)
        ) / (2.0 * EPS)
        check_close(
            Float32(grads.get(i)), numeric, Float32(1e-2), "rope grad_x"
        )


def test_swiglu():
    var state = 9
    var gate = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](1, 5))
    var up = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](1, 5))
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](1, 5))
    _fill2(gate, state, 0.7)
    _fill2(up, state, 0.7)
    _fill2(go, state, 0.5)
    var saved = List[Tensor[DType.float32, 2]]()
    saved.append(gate)
    saved.append(up)
    var grads = swiglu_cpu_backward[DType.float32, 1, 5](go, saved)
    for i in range(gate.numel()):
        var gp = tensor_copy(gate)
        var gm = tensor_copy(gate)
        gp.set(i, Scalar[DType.float32](Float32(gate.get(i)) + EPS))
        gm.set(i, Scalar[DType.float32](Float32(gate.get(i)) - EPS))
        var numeric = (
            _dot2(swiglu_cpu_dynamic[DType.float32](gp, up), go)
            - _dot2(swiglu_cpu_dynamic[DType.float32](gm, up), go)
        ) / (2.0 * EPS)
        check_close(
            Float32(grads[0].get(i)), numeric, Float32(1e-2), "swiglu g_gate"
        )
    for i in range(up.numel()):
        var up2 = tensor_copy(up)
        var um = tensor_copy(up)
        up2.set(i, Scalar[DType.float32](Float32(up.get(i)) + EPS))
        um.set(i, Scalar[DType.float32](Float32(up.get(i)) - EPS))
        var numeric = (
            _dot2(swiglu_cpu_dynamic[DType.float32](gate, up2), go)
            - _dot2(swiglu_cpu_dynamic[DType.float32](gate, um), go)
        ) / (2.0 * EPS)
        check_close(
            Float32(grads[1].get(i)), numeric, Float32(1e-2), "swiglu g_up"
        )


def test_embedding():
    var state = 10
    var tokens = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](3))
    tokens.set(0, Scalar[DType.int32](2))
    tokens.set(1, Scalar[DType.int32](0))
    tokens.set(2, Scalar[DType.int32](1))
    var table = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](3, 2))
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](3, 2))
    _fill2(table, state, 0.7)
    _fill2(go, state, 0.5)
    var grad_table = embedding_cpu_backward[DType.float32, 0, 0](
        go, tokens, table
    )
    for i in range(table.numel()):
        var tp = tensor_copy(table)
        var tm = tensor_copy(table)
        tp.set(i, Scalar[DType.float32](Float32(table.get(i)) + EPS))
        tm.set(i, Scalar[DType.float32](Float32(table.get(i)) - EPS))
        var numeric = (
            _dot2(embedding_cpu_dynamic[DType.float32](tokens, tp), go)
            - _dot2(embedding_cpu_dynamic[DType.float32](tokens, tm), go)
        ) / (2.0 * EPS)
        check_close(
            Float32(grad_table.get(i)), numeric, Float32(1e-2), "emb grad"
        )


# -- registry-path checks -----------------------------------------------------


def _registry_fws_bwd(
    op_name: String, inputs: List[AnyTensor], grad_outs: List[AnyTensor]
) -> List[AnyTensor]:
    var registry = OpRegistry()
    registry.register_default_ops()
    var op = registry.get(op_name, None)
    var fws = op.forward_with_saved(inputs)
    return op.backward(grad_outs, fws[1])


def test_swiglu_ffn_registry():
    var state = 11
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    var gw = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](5, 3))
    var uw = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](5, 3))
    var dw = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](3, 5))
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    _fill2(x, state, 0.5)
    _fill2(gw, state, 0.5)
    _fill2(uw, state, 0.5)
    _fill2(dw, state, 0.5)
    _fill2(go, state, 0.5)
    var inputs = List[AnyTensor]()
    inputs.append(to_any[DType.float32, 2](x))
    inputs.append(to_any[DType.float32, 2](gw))
    inputs.append(to_any[DType.float32, 2](uw))
    inputs.append(to_any[DType.float32, 2](dw))
    var gs = List[AnyTensor]()
    gs.append(to_any[DType.float32, 2](go))
    var grads = _registry_fws_bwd("swiglu_ffn", inputs, gs)
    var gx = from_any[DType.float32, 2](grads[0])
    # numeric check on x (the composed FFN)
    var registry = OpRegistry()
    registry.register_default_ops()
    var op = registry.get("swiglu_ffn", None)
    for i in range(x.numel()):
        var xp = tensor_copy(x)
        var xm = tensor_copy(x)
        xp.set(i, Scalar[DType.float32](Float32(x.get(i)) + EPS))
        xm.set(i, Scalar[DType.float32](Float32(x.get(i)) - EPS))
        var ip = List[AnyTensor]()
        ip.append(to_any[DType.float32, 2](xp))
        ip.append(to_any[DType.float32, 2](gw))
        ip.append(to_any[DType.float32, 2](uw))
        ip.append(to_any[DType.float32, 2](dw))
        var im = List[AnyTensor]()
        im.append(to_any[DType.float32, 2](xm))
        im.append(to_any[DType.float32, 2](gw))
        im.append(to_any[DType.float32, 2](uw))
        im.append(to_any[DType.float32, 2](dw))
        var op_p = from_any[DType.float32, 2](op.forward(ip)[0])
        var op_m = from_any[DType.float32, 2](op.forward(im)[0])
        var numeric = (_dot2(op_p, go) - _dot2(op_m, go)) / (2.0 * EPS)
        check_close(Float32(gx.get(i)), numeric, Float32(1e-2), "ffn grad_x")


def test_cross_entropy():
    var state = 12
    var logits = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](3, 4))
    var targets = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](3))
    targets.set(0, Scalar[DType.int32](1))
    targets.set(1, Scalar[DType.int32](3))
    targets.set(2, Scalar[DType.int32](0))
    _fill2(logits, state, 0.7)
    _ = cross_entropy_forward[DType.float32](logits, targets)
    var ones = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](1))
    ones.set(0, Scalar[DType.float32](Float32(1.0)))
    var grad = cross_entropy_backward[DType.float32](ones, logits, targets)
    # numeric check
    for i in range(logits.numel()):
        var lp = tensor_copy(logits)
        var lm = tensor_copy(logits)
        lp.set(i, Scalar[DType.float32](Float32(logits.get(i)) + EPS))
        lm.set(i, Scalar[DType.float32](Float32(logits.get(i)) - EPS))
        var numeric = (
            Float32(cross_entropy_loss[DType.float32, 4](lp, targets))
            - Float32(cross_entropy_loss[DType.float32, 4](lm, targets))
        ) / (2.0 * EPS)
        check_close(Float32(grad.get(i)), numeric, Float32(1e-2), "ce grad")
    # rows must sum to ~0
    for r in range(3):
        var s = Float32(0)
        for j in range(4):
            s += Float32(grad.get(r * 4 + j))
        check_close(s, Float32(0), Float32(1e-5), "ce grad row sum")


def test_mha_seq_registry():
    # tiny attention: 1 head, T=2, head_dim=2 (hidden=2)
    var state = 13
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 2))
    var wq = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 2))
    var wk = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 2))
    var wv = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 2))
    var wo = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 2))
    var bq = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](2))
    var bk = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](2))
    var bv = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](2))
    var cfg = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](3))
    cfg.set(0, Scalar[DType.int32](1))
    cfg.set(1, Scalar[DType.int32](1))
    cfg.set(2, Scalar[DType.int32](2))
    var pos = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](2))
    pos.set(0, Scalar[DType.float32](0))
    pos.set(1, Scalar[DType.float32](10000.0))
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 2))
    _fill2(x, state, 0.5)
    _fill2(wq, state, 0.5)
    _fill2(wk, state, 0.5)
    _fill2(wv, state, 0.5)
    _fill2(wo, state, 0.5)
    _fill1(bq, state, 0.5)
    _fill1(bk, state, 0.5)
    _fill1(bv, state, 0.5)
    _fill2(go, state, 0.5)
    var inputs = List[AnyTensor]()
    inputs.append(to_any[DType.float32, 2](x))
    inputs.append(to_any[DType.float32, 2](wq))
    inputs.append(to_any[DType.float32, 2](wk))
    inputs.append(to_any[DType.float32, 2](wv))
    inputs.append(to_any[DType.float32, 2](wo))
    inputs.append(to_any[DType.float32, 1](bq))
    inputs.append(to_any[DType.float32, 1](bk))
    inputs.append(to_any[DType.float32, 1](bv))
    inputs.append(to_any[DType.int32, 1](cfg))
    inputs.append(to_any[DType.float32, 1](pos))
    var gs = List[AnyTensor]()
    gs.append(to_any[DType.float32, 2](go))
    var grads = _registry_fws_bwd("mha_seq", inputs, gs)
    var gx = from_any[DType.float32, 2](grads[0])
    var registry = OpRegistry()
    registry.register_default_ops()
    var op = registry.get("mha_seq", None)
    for i in range(x.numel()):
        var xp = tensor_copy(x)
        var xm = tensor_copy(x)
        xp.set(i, Scalar[DType.float32](Float32(x.get(i)) + EPS))
        xm.set(i, Scalar[DType.float32](Float32(x.get(i)) - EPS))
        var ip = List[AnyTensor]()
        ip.append(to_any[DType.float32, 2](xp))
        ip.append(to_any[DType.float32, 2](wq))
        ip.append(to_any[DType.float32, 2](wk))
        ip.append(to_any[DType.float32, 2](wv))
        ip.append(to_any[DType.float32, 2](wo))
        ip.append(to_any[DType.float32, 1](bq))
        ip.append(to_any[DType.float32, 1](bk))
        ip.append(to_any[DType.float32, 1](bv))
        ip.append(to_any[DType.int32, 1](cfg))
        ip.append(to_any[DType.float32, 1](pos))
        var im = List[AnyTensor]()
        im.append(to_any[DType.float32, 2](xm))
        im.append(to_any[DType.float32, 2](wq))
        im.append(to_any[DType.float32, 2](wk))
        im.append(to_any[DType.float32, 2](wv))
        im.append(to_any[DType.float32, 2](wo))
        im.append(to_any[DType.float32, 1](bq))
        im.append(to_any[DType.float32, 1](bk))
        im.append(to_any[DType.float32, 1](bv))
        im.append(to_any[DType.int32, 1](cfg))
        im.append(to_any[DType.float32, 1](pos))
        var op_p = from_any[DType.float32, 2](op.forward(ip)[0])
        var op_m = from_any[DType.float32, 2](op.forward(im)[0])
        var numeric = (_dot2(op_p, go) - _dot2(op_m, go)) / (2.0 * EPS)
        check_close(Float32(gx.get(i)), numeric, Float32(1e-2), "mha grad_x")


# -- interpreter run_with_grad -----------------------------------------------


def test_interpreter_run_with_grad():
    # graph: identity(x) -> rms_norm -> add(n, n)  (gradient accumulates at
    # the rms_norm output twice); loss node = add, seeded with ones.
    var graph = Graph()
    var a0 = Dict[String, AttrValue]()
    a0["n_inputs"] = AttrValue(1)
    var x_id = graph.add_node("identity", List[Int](), a0)
    var norm = graph.add_node("rms_norm", _one(x_id), Dict[String, AttrValue]())
    var add_id = graph.add_node(
        "add", _two(norm, norm), Dict[String, AttrValue]()
    )
    _ = add_id

    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](1, 4))
    var state = 14
    _fill2(x, state, 0.7)
    var inputs = List[AnyTensor]()
    inputs.append(to_any[DType.float32, 2](x))

    var registry = OpRegistry()
    registry.register_default_ops()
    var interp = Interpreter(graph^, registry^)
    var result = interp.run_with_grad(inputs, 2)
    # outputs = add result; grads = [grad w.r.t. x]
    var gx = from_any[DType.float32, 2](result[1][0])
    # reference: rms_norm backward with grad = 2 * ones
    var twice = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](1, 4))
    for i in range(4):
        twice.set(i, Scalar[DType.float32](Float32(2.0)))
    var fws = rms_norm_cpu_forward_with_saved[DType.float32, 4](x)
    var expected = rms_norm_cpu_backward[DType.float32, 4](twice, fws[1])
    for i in range(4):
        check_close(
            Float32(gx.get(i)),
            Float32(expected[0].get(i)),
            Float32(1e-4),
            "run_with_grad gx",
        )
