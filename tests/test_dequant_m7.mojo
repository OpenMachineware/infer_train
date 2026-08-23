# tests/test_dequant_m7.mojo
#
# M7 dequantizer validation: Q4_K / Q8_0 / IQ4_NL / IQ4_XS (+ Q5_K / Q6_K
# regression) against a numpy reference computed from llama.cpp's
# ggml-quants.c formulas (/tmp/dequant_ref.py generates the .bin files).

from src.core.gguf_loader import load_gguf, find_tensor
from src.core.ops.quantized.dequantize import dequantize_into
from src.core.tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple
from std.io.file import FileHandle
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.collections import Span


comptime MODEL_HY = "Hy-MT2-7B-Q4_K_M.gguf"
comptime MODEL_Q35 = "Qwen3.8-27B-UD-Q5_K_M.gguf"


def main() raises:
    # (model, tensor, type, ref bin, compare count)
    check_model(
        MODEL_HY, "blk.0.attn_k.weight", 12, "/tmp/dequant_ref_hy.npz_0.bin"
    )
    check_model(
        MODEL_HY,
        "blk.0.attn_output.weight",
        12,
        "/tmp/dequant_ref_hy.npz_1.bin",
    )
    check_model(
        MODEL_HY, "token_embd.weight", 14, "/tmp/dequant_ref_hy.npz_2.bin"
    )
    check_model(
        MODEL_HY, "output_norm.weight", 0, "/tmp/dequant_ref_hy.npz_3.bin"
    )
    check_model(
        MODEL_Q35,
        "blk.0.attn_norm.weight",
        0,
        "/tmp/dequant_ref_q35.npz_0.bin",
    )
    check_model(
        MODEL_Q35,
        "blk.0.ssm_alpha.weight",
        8,
        "/tmp/dequant_ref_q35.npz_1.bin",
    )
    check_model(
        MODEL_Q35,
        "blk.2.ffn_up.weight",
        20,
        "/tmp/dequant_ref_q35.npz_2.bin",
    )
    check_model(
        MODEL_Q35,
        "blk.0.ffn_gate.weight",
        23,
        "/tmp/dequant_ref_q35.npz_3.bin",
    )
    check_model(
        MODEL_Q35,
        "blk.0.attn_qkv.weight",
        13,
        "/tmp/dequant_ref_q35.npz_4.bin",
    )
    print("test_dequant_m7 OK")


def check_model(
    model_path: String, name: String, ggml_type: Int, ref_path: String
) raises:
    var ctx = load_gguf(model_path)
    var tensor = find_tensor(ctx, name)
    if not tensor:
        print("FAIL: tensor not found:", name)
        abort()
    if tensor.value().ggml_type != ggml_type:
        print(
            "FAIL: type mismatch for", name, "actual:",
            tensor.value().ggml_type, "expected:", ggml_type,
        )
        abort()
    var numel = 1024
    var dst = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, numel))
    dequantize_into(
        ggml_type, ctx.data, ctx.data_offset + tensor.value().offset, dst, numel
    )
    # load the reference floats
    var handle = FileHandle(ref_path, "r")
    var ref_size = Int(handle.seek(0, UInt8(2)))
    _ = handle.seek(0, UInt8(0))
    var raw = unsafe_alloc[UInt8](ref_size)
    var span = Span[UInt8, MutUntrackedOrigin](
        unsafe_ptr=raw, length=ref_size
    )
    var read_count = handle.read[DType.uint8](span)
    handle.close()
    if read_count != ref_size:
        print("FAIL: short ref read", ref_path)
        abort()
    var ref_data = raw.unsafe_bitcast[Scalar[DType.float32]]()
    var n_refs = ref_size // 4
    var n = 1024
    if n > n_refs:
        n = n_refs
    var max_rel = Float32(0)
    for i in range(n):
        var ref_val = Float32(ref_data.unsafe_load[width=1](offset=i))
        if ref_val != ref_val:  # NaN reference: skip (dynamic-quant outliers)
            continue
        var got = Float32(dst.get(i))
        var diff = got - ref_val
        if diff < 0:
            diff = -diff
        var denom = ref_val if ref_val > 0 else -ref_val
        if denom < Float32(1e-6):
            denom = Float32(1e-6)
        var rel = diff / denom
        if rel > max_rel:
            max_rel = rel
        if rel > Float32(5e-3):
            print(
                "FAIL:", name, "index", i, "got:", got, "ref:", ref_val,
                "rel:", rel,
            )
            abort()
    print("  ok:", name, "type", ggml_type, "n", n, "max_rel:", max_rel)


def abort():
    from std.os.os import abort as _abort

    _abort()
