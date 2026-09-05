# tests/test_qwen35moe.mojo
#
# Qwen3.6-35B-A3B (arch `qwen35moe`) load + config + forward smoke test.
#
# This is the MoE + hybrid-SSM case that motivated the config-driven arch
# redesign: `general.architecture = qwen35moe` used to fall through to the
# dense qwen2 path (reading `qwen2.*` keys -> n_layers=0 -> KV cache
# capacity 0 -> error).  Now any `qwen*` arch maps to the unified Qwen path
# and the MoE / SSM / QK-norm / gate behavior is read from the metadata.
#
# Checks the config is populated correctly (the primary bug) and that a
# short prefill produces finite logits (the MoE FFN runs end to end).
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

comptime MODEL_PATH = "Qwen3.6-35B-A3B-DSV4Pro-Distill-MTP-Q5_K_M-imatrix.gguf"


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
    print(
        "vocab:",
        config.vocab,
        "rope:",
        config.rope_theta,
        "eps:",
        config.norm_eps,
    )
    print(
        "ssm:",
        config.has_ssm,
        "interval:",
        config.full_attn_interval,
        "n_rot:",
        config.n_rot,
    )
    print(
        "moe:",
        config.is_moe,
        "experts:",
        config.n_experts,
        "top_k:",
        config.n_experts_used,
    )
    print(
        "qk_norm:",
        config.has_qk_norm,
        "gate:",
        config.has_gate,
        "post_attn_norm:",
        config.has_post_attn_norm,
    )

    # -- config checks (the primary bug: these used to all read 0) ----------
    check(config.arch == ARCH_QWEN, "unified qwen arch")
    check(config.arch_str == "qwen35moe", "arch_str preserved")
    # block_count=41 includes the MTP block (nextn=1) -> 40 main layers.
    check(config.n_layers == 40, "layer count (41 - 1 MTP)")
    check(config.hidden == 2048, "hidden")
    check(config.n_heads == 16 and config.n_kv_heads == 2, "heads")
    check(config.head_dim == 256, "head_dim from key_length")
    check(config.vocab == 248320, "vocab")
    # MoE capability flags (config-driven, from the metadata keys).
    check(config.is_moe, "is_moe")
    check(config.n_experts == 256, "expert_count")
    check(config.n_experts_used == 8, "expert_used_count")
    check(config.expert_ffn == 512, "expert_feed_forward_length")
    check(config.ffn == 512, "ffn falls back to expert_ffn")
    # Hybrid SSM capability flags.
    check(config.has_ssm, "has_ssm")
    check(config.full_attn_interval == 4, "full_attention_interval")
    check(config.ssm_d_state == 128, "ssm.state_size")
    check(config.ssm_d_inner == 4096, "ssm.inner_size")
    check(config.n_rot == 64, "rope.dimension_count")
    # Attention-layer capability flags (probed from the tensor layout).
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
    if n_prefill > 16:
        n_prefill = 16
    var logits = tensor_zeros[DType.float32, 1](
        StaticTuple[Int, 1](config.vocab)
    )
    for i in range(n_prefill):
        logits = model.forward(tokens[i], i)
    var mx = Float32(-3.0e38)
    var mn = Float32(3.0e38)
    var arg = 0
    for i in range(config.vocab):
        var v = Float32(logits.get(i))
        if v > mx:
            mx = v
            arg = i
        if v < mn:
            mn = v
    var one = List[Int]()
    one.append(arg)
    print(
        "argmax:",
        arg,
        "max:",
        mx,
        "min:",
        mn,
        "decoded:",
        tokenizer.decode(one),
    )
    check(mx > Float32(-1.0e6) and mx < Float32(1.0e6), "logits finite")
    check(mn > Float32(-1.0e6) and mn < Float32(1.0e6), "logits bounded")

    print("test_qwen35moe OK")


def check(cond: Bool, name: String):
    if not cond:
        print("FAIL:", name)
        abort()


def abort():
    from std.os.os import abort as _abort

    _abort()
