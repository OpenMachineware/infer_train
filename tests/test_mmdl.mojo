# tests/test_mmdl.mojo
#
# M7 private-format tests: save -> load round-trip (weights + grads +
# AdamW m/v + metadata), incremental update, delta append, and GGUF-core
# compatibility.

from src.core.training import TrainModel, TrainConfig
from src.core.mmdl_storage import (
    save_checkpoint,
    save_checkpoint_incremental,
    load_checkpoint,
    append_delta,
    strip_to_gguf,
    CheckpointMeta,
)
from src.core.gguf_loader import load_gguf
from src.core.tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple

comptime CKPT_PATH = "/tmp/test_mmdl_checkpoint.mmdl"
comptime STRIP_PATH = "/tmp/test_mmdl_stripped.gguf"


def main() raises:
    var cfg = TrainConfig()
    cfg.n_layers = 2
    cfg.hidden = 8
    cfg.ffn = 16
    cfg.vocab = 24
    cfg.n_heads = 2
    cfg.n_kv_heads = 2
    cfg.head_dim = 4
    var model = TrainModel(cfg, seed=3)
    # run one optimizer step so m/v state exists
    var tokens = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](4))
    var targets = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](4))
    for i in range(4):
        tokens.set(i, Scalar[DType.int32](i % cfg.vocab))
        targets.set(i, Scalar[DType.int32]((i + 1) % cfg.vocab))
    var losses = List[Float32]()
    losses.append(Float32(1.5))
    losses.append(Float32(1.25))
    _ = model.train_step(tokens, targets)
    # move one parameter so the save/load is observable (after the step)
    model.token_embd.set(0, Scalar[DType.float32](Float32(0.5)))
    model.layers[0].q_w.set(2, Scalar[DType.float32](Float32(-0.25)))

    save_checkpoint(model, CKPT_PATH, 2, losses)
    print("saved")

    # GGUF structure: the file must parse as GGUF and hold the tensors
    var ctx = load_gguf(CKPT_PATH)
    print("tensors:", len(ctx.tensors), "meta:", len(ctx.metadata))
    check(len(ctx.tensors) >= 14, "tensor count")
    var has_grad = False
    var has_m = False
    for t in ctx.tensors:
        if t.name == "grad.blk.0.attn_q.weight":
            has_grad = True
        if t.name == "opt.m.output.weight":
            has_m = True
    check(has_grad, "grad tensors present")
    check(has_m, "adam m tensors present")

    # round-trip into a fresh model
    var model2 = TrainModel(cfg, seed=999)
    var meta = load_checkpoint(model2, CKPT_PATH)
    check(meta.step == 2, "step restored")
    check(len(meta.loss_history) == 2, "loss history restored")
    check(
        Float32(meta.loss_history[0]) == Float32(1.5)
        and Float32(meta.loss_history[1]) == Float32(1.25),
        "loss values",
    )
    check(
        Float32(model2.token_embd.get(0)) == Float32(0.5),
        "weight restored",
    )
    check(
        Float32(model2.layers[0].q_w.get(2)) == Float32(-0.25),
        "layer weight restored",
    )

    # incremental: only one tensor changes
    model.token_embd.set(1, Scalar[DType.float32](Float32(0.75)))
    var changed = List[String]()
    changed.append("token_embd.weight")
    save_checkpoint_incremental(model, CKPT_PATH, 3, changed, losses)
    var model3 = TrainModel(cfg, seed=999)
    _ = load_checkpoint(model3, CKPT_PATH)
    check(
        Float32(model3.token_embd.get(1)) == Float32(0.75),
        "incremental update applied",
    )
    check(
        Float32(model3.token_embd.get(0)) == Float32(0.5),
        "unchanged weight preserved",
    )

    # delta append (in-place update of one tensor)
    var delta = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](8, 24))
    delta.set(0, Scalar[DType.float32](Float32(0.125)))
    append_delta(CKPT_PATH, "token_embd.weight", delta)
    var ctx2 = load_gguf(CKPT_PATH)
    check(len(ctx2.tensors) >= 14, "delta keeps GGUF parseable")

    # strip to a weights-only GGUF for llama.cpp
    strip_to_gguf(CKPT_PATH, STRIP_PATH)
    var ctx3 = load_gguf(STRIP_PATH)
    var has_grad2 = False
    for t in ctx3.tensors:
        if t.name == "grad.blk.0.attn_q.weight":
            has_grad2 = True
    check(not has_grad2, "strip drops grad tensors")
    check(len(ctx3.tensors) == 27, "strip keeps weights only")

    print("test_mmdl OK")


def check(cond: Bool, name: String):
    if not cond:
        print("FAIL:", name)
        abort()


def abort():
    from std.os.os import abort as _abort

    _abort()
