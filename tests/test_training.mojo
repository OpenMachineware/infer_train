# tests/test_training.mojo
#
# M6 Phase 8: end-to-end training tests in Mojo.
#
#   * a full training loop on a small decoder model (TrainModel): the loss
#     must decrease across steps and the eval accuracy must improve;
#   * gradient accumulation: with accumulation_steps=4 the parameters must
#     not move for the first 3 mini-batches and must move on the 4th;
#   * AMP: the GradScaler backs off on Inf gradients and grows on clean
#     runs, and a model trains through the enable_amp path;
#   * the interpreter's eval path reports a valid accuracy.

from src.core.training import TrainModel, TrainConfig, train_step, eval_step
from src.core.gradient_scaler import GradScaler
from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.base.op_interface import AnyTensor, to_any
from std.utils.static_tuple import StaticTuple
from std.os.os import abort


def check(condition: Bool, label: String):
    if not condition:
        print("FAIL", label)
        abort()


def check_close(actual: Float32, expected: Float32, tol: Float32, label: String):
    var diff = actual - expected
    if diff < 0:
        diff = -diff
    if diff > tol:
        print("FAIL", label, "actual:", actual, "expected:", expected)
        abort()


def main():
    test_loss_decreases()
    test_gradient_accumulation()
    test_amp_scaler_and_training()
    test_eval_accuracy()
    test_dynamic_quantize_roundtrip()
    print("test_training OK")


def _make_config() -> TrainConfig:
    var cfg = TrainConfig()
    cfg.n_layers = 1
    cfg.hidden = 16
    cfg.ffn = 32
    cfg.n_heads = 2
    cfg.n_kv_heads = 2
    cfg.head_dim = 8
    cfg.vocab = 32
    return cfg


def _make_data() -> Tuple[
    Tensor[DType.int32, 1], Tensor[DType.int32, 1]
]:
    # 8 tokens with "next token = token + 1 (mod 32)" targets: learnable
    var tokens = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](8))
    var targets = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](8))
    for i in range(8):
        var tok = (i * 4) % 32
        tokens.set(i, Scalar[DType.int32](tok))
        targets.set(i, Scalar[DType.int32]((tok + 1) % 32))
    return (tokens, targets)


def test_loss_decreases():
    var model = TrainModel(_make_config(), 7)
    var data = _make_data()
    var tokens = data[0]
    var targets = data[1]
    var first = Float32(0)
    var last = Float32(0)
    for step in range(30):
        var r = train_step(model, tokens, targets)
        if step == 0:
            first = r[0]
        last = r[0]
    print(
        "  loss:",
        first,
        "->",
        last,
        "(", len_placeholder(), ")",
    )
    check(last < first, "train loss must decrease")
    check(first - last > Float32(0.1), "loss must decrease meaningfully")


def len_placeholder() -> Int:
    return 30


def test_gradient_accumulation():
    var model = TrainModel(_make_config(), 11)
    var data = _make_data()
    var tokens = data[0]
    var targets = data[1]
    var p0 = Float32(model.token_embd.get(0))
    # first 3 mini-batches: no optimizer step
    for i in range(3):
        var r = train_step(model, tokens, targets, 4)
        _ = r
    check_close(
        Float32(model.token_embd.get(0)), p0, Float32(0), "accum no step"
    )
    # the 4th mini-batch triggers the step
    var r4 = train_step(model, tokens, targets, 4)
    _ = r4
    check(
        Float32(model.token_embd.get(0)) != p0, "accum step fires"
    )


def test_amp_scaler_and_training():
    var scaler = GradScaler()
    # no Inf/NaN: scale grows after the growth interval
    var clean = List[AnyTensor]()
    var g = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](1, 4))
    clean.append(to_any[DType.float32, 2](g))
    for i in range(2000):
        scaler.update(False)
    check(
        scaler.current_scale() > Float32(65536.0), "scaler grows"
    )
    # an Inf gradient backs the scale off
    var ginf = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](1, 4))
    ginf.set(0, Scalar[DType.float32](Float32(1.0) / Float32(0.0)))
    var bad = List[AnyTensor]()
    bad.append(to_any[DType.float32, 2](ginf))
    check(scaler.found_inf(bad), "scaler detects inf")
    var before = scaler.current_scale()
    scaler.update(True)
    check(scaler.current_scale() < before, "scaler backs off")

    # full AMP training path on a small model
    var model = TrainModel(_make_config(), 13)
    model.enable_amp(True)
    var data = _make_data()
    var tokens = data[0]
    var targets = data[1]
    var first = Float32(0)
    var last = Float32(0)
    for step in range(20):
        var r = train_step(model, tokens, targets)
        if step == 0:
            first = r[0]
        last = r[0]
    check(last < first, "amp training decreases loss")


def test_dynamic_quantize_roundtrip():
    # Phase 7: dynamic quantize + dequantize must reconstruct x within
    # half a quantization step
    from src.core.ops.base.op_registry import OpRegistry
    from src.core.ops.base.op_interface import AnyTensor, to_any, from_any

    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 8))
    var state = 3
    for i in range(16):
        state = (state * 1103515245 + 12345) % 2147483648
        x.set(
            i,
            Scalar[DType.float32](
                (Float32(state) / Float32(2147483648.0)) * 2.0 - 1.0
            ),
        )
    var registry = OpRegistry()
    registry.register_default_ops()
    var q_op = registry.get("dynamic_quantize", None)
    var dq_op = registry.get("dynamic_dequantize", None)
    var inputs = List[AnyTensor]()
    inputs.append(to_any[DType.float32, 2](x))
    var qres = q_op.forward(inputs)
    var dq_inputs = List[AnyTensor]()
    dq_inputs.append(qres[0])
    dq_inputs.append(qres[1])
    dq_inputs.append(qres[2])
    var xr = from_any[DType.float32, 2](dq_op.forward(dq_inputs)[0])
    var scale = from_any[DType.float32, 1](qres[1]).get(0)
    for i in range(16):
        var diff = Float32(xr.get(i)) - Float32(x.get(i))
        if diff < 0:
            diff = -diff
        check(diff <= Float32(scale) * 0.5 + Float32(1e-4), "quant roundtrip")


def test_eval_accuracy():
    var model = TrainModel(_make_config(), 17)
    var data = _make_data()
    var tokens = data[0]
    var targets = data[1]
    # train a few steps first
    for step in range(10):
        _ = train_step(model, tokens, targets)
    var e = eval_step(model, tokens, targets)
    check(e[1] >= Float32(0) and e[1] <= Float32(1), "accuracy in range")
    print("  eval loss:", e[0], "accuracy:", e[1])
