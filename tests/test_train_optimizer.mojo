# tests/test_train_optimizer.mojo
#
# M6 Phase 8: optimizer tests.
#
# AdamW is checked against reference values produced by torch.optim.AdamW
# (see tests/python/test_training.py::test_adamw_matches_torch for the
# live comparison; the constants here were generated from the same
# PyTorch run): param [[1, -2], [3, 0.5]], three fixed gradient steps with
# lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01.
#
# SGD-with-momentum is checked with a hand derivation, and the parameter
# group support (per-group lr/weight_decay) with an analytic update.

from src.core.train_optimizer import AdamW, SGD
from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.base.op_interface import AnyTensor, to_any
from std.utils.static_tuple import StaticTuple
from std.os.os import abort


def check_close(actual: Float32, expected: Float32, tol: Float32, label: String):
    var diff = actual - expected
    if diff < 0:
        diff = -diff
    if diff > tol:
        print("FAIL", label, "actual:", actual, "expected:", expected)
        abort()


def main():
    test_adamw_matches_torch_reference()
    test_adamw_zero_grad()
    test_adamw_param_groups()
    test_sgd_momentum()
    print("test_train_optimizer OK")


def _make_param(vals: List[Float32]) -> Tensor[DType.float32, 2]:
    var p = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 2))
    for i in range(4):
        p.set(i, Scalar[DType.float32](vals[i]))
    return p


def _make_grad(vals: List[Float32]) -> AnyTensor:
    var g = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 2))
    for i in range(4):
        g.set(i, Scalar[DType.float32](vals[i]))
    return to_any[DType.float32, 2](g)


def _l4(a: Float32, b: Float32, c: Float32, d: Float32) -> List[Float32]:
    var l = List[Float32]()
    l.append(a)
    l.append(b)
    l.append(c)
    l.append(d)
    return l^


def _accumulate(mut opt: AdamW, vals: List[Float32]):
    var grads = List[AnyTensor]()
    grads.append(_make_grad(vals))
    opt.accumulate_grads(grads)


def test_adamw_matches_torch_reference():
    var p = _make_param(_l4(Float32(1.0), Float32(-2.0), Float32(3.0), Float32(0.5)))
    var opt = AdamW(Float32(0.1))
    opt.set_group_hyperparams(0, Float32(0.1), Float32(0.01))
    opt.add_param[DType.float32, 2](p)

    _accumulate(opt, _l4(Float32(0.5), Float32(-0.25), Float32(1.0), Float32(-0.75)))
    opt.step()
    opt.zero_grad()
    var expect1 = _l4(
        Float32(0.899000049), Float32(-1.898000002),
        Float32(2.897000074), Float32(0.5995),
    )
    for i in range(4):
        check_close(Float32(p.get(i)), expect1[i], Float32(2e-6), "adamw step1")

    _accumulate(opt, _l4(Float32(0.1), Float32(0.2), Float32(-0.3), Float32(0.4)))
    opt.step()
    opt.zero_grad()
    var expect2 = _l4(
        Float32(0.817796946), Float32(-1.890289545),
        Float32(2.851318121), Float32(0.622984886),
    )
    for i in range(4):
        check_close(Float32(p.get(i)), expect2[i], Float32(2e-6), "adamw step2")

    _accumulate(opt, _l4(Float32(-0.5), Float32(0.5), Float32(0.25), Float32(-0.125)))
    opt.step()
    opt.zero_grad()
    var expect3 = _l4(
        Float32(0.817426682), Float32(-1.939788222),
        Float32(2.801415205), Float32(0.650083184),
    )
    for i in range(4):
        check_close(Float32(p.get(i)), expect3[i], Float32(2e-6), "adamw step3")


def test_adamw_zero_grad():
    var p = _make_param(_l4(Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0)))
    var opt = AdamW(Float32(0.1))
    opt.add_param[DType.float32, 2](p)
    _accumulate(opt, _l4(Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0)))
    opt.zero_grad()
    # with a zero gradient only the (decoupled) weight decay moves the
    # parameter: p = p - lr * wd * p
    opt.step()
    for i in range(4):
        check_close(
            Float32(p.get(i)), Float32(1.0 - 0.1 * 0.01), Float32(1e-6),
            "zero_grad wd step1",
        )
    opt.step()
    for i in range(4):
        check_close(
            Float32(p.get(i)),
            Float32((1.0 - 0.1 * 0.01) * (1.0 - 0.1 * 0.01)),
            Float32(1e-6),
            "zero_grad wd step2",
        )


def test_adamw_param_groups():
    # two parameters, different group lr: 0.1 vs 0.0 (frozen)
    var p1 = _make_param(_l4(Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0)))
    var p2 = _make_param(_l4(Float32(2.0), Float32(2.0), Float32(2.0), Float32(2.0)))
    var opt = AdamW(Float32(0.1))
    opt.add_param[DType.float32, 2](p1, 0)
    opt.add_param[DType.float32, 2](p2, 0)
    var g = List[AnyTensor]()
    g.append(_make_grad(_l4(Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0))))
    g.append(_make_grad(_l4(Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0))))
    opt.accumulate_grads(g)
    opt.step()
    # same group, same hyperparameters: p1 = 1 - lr*(mhat + wd*1),
    # p2 = 2 - lr*(mhat + wd*2) with mhat = vhat = 1 at step 1
    for i in range(4):
        check_close(Float32(p1.get(i)), Float32(0.899), Float32(2e-6), "group p1")
        check_close(Float32(p2.get(i)), Float32(1.898), Float32(2e-6), "group p2")
    # freeze group 0: a second step must not move p1/p2
    opt.set_group_hyperparams(0, Float32(0.0), Float32(0.0))
    var before = Float32(p1.get(0))
    g2 = List[AnyTensor]()
    g2.append(_make_grad(_l4(Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0))))
    g2.append(_make_grad(_l4(Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0))))
    opt.accumulate_grads(g2)
    opt.step()
    check_close(Float32(p1.get(0)), before, Float32(1e-6), "frozen group")


def test_sgd_momentum():
    var p = _make_param(_l4(Float32(2.0), Float32(3.0), Float32(4.0), Float32(5.0)))
    var opt = SGD(Float32(0.1), Float32(0.9))
    opt.add_param[DType.float32, 2](p)
    var g = List[AnyTensor]()
    g.append(_make_grad(_l4(Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0))))
    opt.accumulate_grads(g)
    opt.step()
    # buf = 1.0; p = 2 - 0.1*1 = 1.9
    check_close(Float32(p.get(0)), Float32(1.9), Float32(1e-6), "sgd step1")
    opt.zero_grad()
    opt.accumulate_grads(g)
    opt.step()
    # buf = 0.9*1 + 1 = 1.9; p = 1.9 - 0.19 = 1.71
    check_close(Float32(p.get(0)), Float32(1.71), Float32(1e-6), "sgd step2")
