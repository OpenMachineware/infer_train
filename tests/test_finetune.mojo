# tests/test_finetune.mojo
#
# M7 inference-time fine-tuning: finetune_step updates parameters, LoRA
# mode (set_trainable) restricts updates to the selected subset, and the
# loss decreases over a short adaptation run.

from src.core.training import (
    TrainModel,
    TrainConfig,
    train_step,
    eval_step,
    finetune_step,
    set_trainable,
    param_names,
    FinetuneMode,
)
from src.core.train_optimizer import AdamW
from src.core.tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple


def main():
    var cfg = TrainConfig()
    cfg.n_layers = 2
    cfg.hidden = 8
    cfg.ffn = 16
    cfg.vocab = 24
    cfg.n_heads = 2
    cfg.n_kv_heads = 2
    cfg.head_dim = 4
    var model = TrainModel(cfg, seed=3)

    # a repeating pattern: 0 1 2 3 -> 1 2 3 0
    var tokens = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](4))
    var targets = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](4))
    for i in range(4):
        tokens.set(i, Scalar[DType.int32](i))
        targets.set(i, Scalar[DType.int32]((i + 1) % 4))

    var (loss0, _) = eval_step(model, tokens, targets)
    print("loss0:", loss0)

    # full fine-tuning: 30 steps, loss must drop
    var (l1, _) = finetune_step(model, tokens, targets)
    print("step1 loss:", l1)
    for _ in range(29):
        _ = finetune_step(model, tokens, targets)
    var (loss1, _) = eval_step(model, tokens, targets)
    print("loss after full finetune:", loss1)
    check(loss1 < loss0, "full finetune lowers loss")

    # LoRA mode: freeze everything except the output head
    var model2 = TrainModel(cfg, seed=3)
    var trainable = List[String]()
    trainable.append("output.weight")
    set_trainable(model2, trainable)
    # record pre-step values of a frozen param and the trainable one
    var frozen_before = Float32(model2.token_embd.get(0))
    var head_before = Float32(model2.output_w.get(0))
    _ = finetune_step(model2, tokens, targets)
    check(
        Float32(model2.token_embd.get(0)) == frozen_before,
        "frozen param unchanged in LoRA mode",
    )
    check(
        Float32(model2.output_w.get(0)) != head_before,
        "trainable head updated in LoRA mode",
    )

    # mode struct + names API sanity
    check(FinetuneMode.Lora != FinetuneMode.Full, "mode tags")
    var names = param_names(model2)
    check(len(names) == 1 + 2 * 12 + 2, "param name count")
    check(
        names[0] == "token_embd.weight"
        and names[len(names) - 1] == "output.weight",
        "name order",
    )

    print("test_finetune OK")


def check(cond: Bool, name: String):
    if not cond:
        print("FAIL:", name)
        abort()


def abort():
    from std.os.os import abort as _abort

    _abort()
