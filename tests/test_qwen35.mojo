# tests/test_qwen35.mojo
#
# Qwen3.8-27B (arch `qwen35`) load + config + forward smoke test.
#
# The dense hybrid-SSM case: Gated DeltaNet recurrent layers interleaved
# with full-attention layers (fused Q+gate, QK-norm, MRoPE), no MoE.  This
# exercises the config-driven capability flags (has_ssm / has_gate /
# has_post_attn_norm) and the unified attention + SSM forward paths.
#
# Skips (SKIP) if the model file is not present.

from src.core.gguf_loader import load_gguf
from src.core.transformer import (
    TransformerModel,
    load_config,
    collect_weights,
    ARCH_QWEN,
    arch_name,
)
from src.core.tokenizers import make_tokenizer
from src.core.tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple

comptime MODEL_PATH = "Qwen3.8-27B-UD-Q5_K_M.gguf"


def _file_exists(path: String) -> Bool:
    try:
        var f = FileHandle(path, "r")
        f.close()
        return True
    except:
        return False


def main() raises:
    if not _file_exists(MODEL_PATH):
        print("SKIP: " + MODEL_PATH + " not present")
        return

    var ctx = load_gguf(MODEL_PATH)
    var config = load_config(ctx)
    print("arch:", arch_name(config.arch), "arch_str:", config.arch_str)
    print(
        "layers:", config.n_layers, "hidden:", config.hidden, "ffn:", config.ffn
    )
    print(
        "heads:",
        config.n_heads,
        "kv:",
        config.n_kv_heads,
        "head_dim:",
        config.head_dim,
    )
    print("vocab:", config.vocab, "rope:", config.rope_theta)
    print(
        "ssm:",
        config.has_ssm,
        "interval:",
        config.full_attn_interval,
        "n_rot:",
        config.n_rot,
    )
    print("moe:", config.is_moe)
    print(
        "qk_norm:",
        config.has_qk_norm,
        "gate:",
        config.has_gate,
        "post_attn_norm:",
        config.has_post_attn_norm,
    )

    # -- config checks (config-driven capability flags) ----------------------
    check(config.arch == ARCH_QWEN, "unified qwen arch")
    check(config.arch_str == "qwen35", "arch_str preserved")
    # block_count=65 includes the MTP block (nextn=1) -> 64 main layers.
    check(config.n_layers == 64, "layer count (65 - 1 MTP)")
    check(config.hidden == 5120, "hidden")
    check(config.ffn == 17408, "ffn")
    check(config.n_heads == 24 and config.n_kv_heads == 4, "heads")
    check(config.head_dim == 256, "head_dim from key_length")
    check(config.vocab == 248320, "vocab")
    # Hybrid SSM (dense, no MoE).
    check(config.has_ssm, "has_ssm")
    check(config.full_attn_interval == 4, "full_attention_interval")
    check(config.ssm_d_state == 128, "ssm.state_size")
    check(config.ssm_d_inner == 6144, "ssm.inner_size")
    check(config.n_rot == 64, "rope.dimension_count")
    check(not config.is_moe, "not MoE (dense)")
    # Attention-layer capability flags.
    check(config.has_qk_norm, "has_qk_norm")
    check(config.has_gate, "has_gate (fused Q+gate)")
    check(config.has_post_attn_norm, "has_post_attn_norm")
    check(config.norm_before_rope, "norm_before_rope (qwen family)")

    var weights = collect_weights(ctx)
    var model = TransformerModel(config, ctx^, 512)
    model.weights = weights^

    var tokenizer = make_tokenizer(model.ctx, String(""))
    print(
        "tokenizer:", tokenizer.flavor_name(), "vocab:", tokenizer.vocab_size()
    )

    # -- forward smoke: prefill a few prompt tokens -------------------------
    var tokens = tokenizer.encode_with_bos("What is 1+1?")
    print("prompt tokens:", len(tokens))
    var n_prefill = len(tokens)
    if n_prefill > 8:
        n_prefill = 8
    var logits = tensor_zeros[DType.float32, 1](
        StaticTuple[Int, 1](config.vocab)
    )
    for i in range(n_prefill):
        logits = model.forward(tokens[i], i)
    var mx = Float32(-3.0e38)
    var arg = 0
    for i in range(config.vocab):
        var v = Float32(logits.get(i))
        if v > mx:
            mx = v
            arg = i
    var one = List[Int]()
    one.append(arg)
    print("argmax:", arg, "max:", mx, "decoded:", tokenizer.decode(one))
    check(mx > Float32(-1.0e6) and mx < Float32(1.0e6), "logits finite")

    print("test_qwen35 OK")


def check(cond: Bool, name: String):
    if not cond:
        print("FAIL:", name)
        abort()


def abort():
    from std.os.os import abort as _abort

    _abort()
