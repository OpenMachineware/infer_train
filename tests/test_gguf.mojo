# tests/test_gguf.mojo
#
# Q4-resident (M11) validation for the GGUF weight pipeline:
#
#   1. Q4_0 kernel: the new Q4_0 dequantizer (whole-tensor `dequantize_into`
#      and the block-mode `dequantize_blocks`) against hand-computed values.
#   2. Memory: the 27B model loads with its weights in their on-disk
#      (quantized) format - after faulting in every mapped page the process
#      footprint stays < 25 GB.  The legacy full-dequantize path materializes
#      ~54 GB of fp16 for this model and swaps on a 64 GB machine.
#   3. Correctness: on the 7B Q4_K_M and 1.5B Q5_K_M models, the
#      quantized-resident forward (fused per-block-dequant matmul, the M11
#      default) matches the legacy dequantized forward within fp16
#      tolerance.
#
# Skips (SKIP) a section when its model file is not present.

from src.core.tensor import Tensor, tensor_zeros
from src.core.gguf_loader import (
    GGUFContext,
    find_tensor,
    ggml_quant_info,
    load_gguf,
)
from src.core.transformer import TransformerModel, load_config
from src.core.memory import process_resident_bytes, process_rss_bytes
from src.core.ops.quantized.dequantize import (
    dequantize_into,
    dequantize_blocks,
)
from src.core.ops.quantized.quant_types import QuantType
from std.utils.static_tuple import StaticTuple
from std.io.file import FileHandle
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin


comptime MODEL_15B = "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"
comptime MODEL_7B = "Hy-MT2-7B-Q4_K_M.gguf"
comptime MODEL_27B = "Qwen3.8-27B-UD-Q5_K_M.gguf"


def _file_exists(path: String) -> Bool:
    try:
        var f = FileHandle(path, "r")
        f.close()
        return True
    except:
        return False


def _gib(x: Int) -> Float64:
    return Float64(x) / (1024.0 * 1024.0 * 1024.0)


def _max_logit_diff(
    a: Tensor[DType.float32, 1], b: Tensor[DType.float32, 1]
) -> Float32:
    var n = a.numel()
    var m = Float32(0.0)
    for i in range(n):
        var d = abs(Float32(a.get(i)) - Float32(b.get(i)))
        if d > m:
            m = d
    return m


def _touch_all(ctx: GGUFContext) -> Tuple[Int, Int]:
    """Fault in every mapped tensor page (mmap is lazy); returns
    (payload bytes touched, checksum).

    The checksum MUST escape the function (it is printed by the caller):
    with a per-tensor checksum that is discarded inside the loop, the
    compiler dead-code-eliminates the whole touch loop (verified: the
    resident delta then stays at the KV-cache size instead of growing by
    the payload), which makes the memory measurement vacuous.
    """
    var total = 0
    var checksum = 0
    for t in ctx.tensors:
        var numel = 1
        for d in range(t.n_dims):
            numel *= t.dims[d]
        var (be, bb, gs, tag) = ggml_quant_info(t.ggml_type)
        _ = gs
        _ = tag
        var n_bytes = numel * bb // be
        var (base, off) = ctx.tensor_data(t)
        var p = base.unsafe_offset(off)
        var i = 0
        while i + 16 <= n_bytes:
            var v = p.unsafe_load[width=16](offset=i)
            checksum += Int(v.reduce_add())
            i += 16
        while i < n_bytes:
            checksum += Int(p.unsafe_load[width=1](offset=i))
            i += 1
        total += n_bytes
    return (total, checksum)


# -- 1. Q4_0 dequantizer (whole-tensor + block mode) -------------------------


def check_q4_0_kernel() raises:
    # Two synthetic Q4_0 blocks (18 bytes each): d0 = 1.5, d1 = -0.5.
    var data = unsafe_alloc[UInt8](36)
    data.unsafe_store(0, UInt8(0x00))  # fp16 1.5 = 0x3E00 (LE)
    data.unsafe_store(1, UInt8(0x3E))
    for j in range(16):
        data.unsafe_store(2 + j, UInt8((j & 0xF) | ((j & 0xF) << 4)))
    data.unsafe_store(18, UInt8(0x00))  # fp16 -0.5 = 0xB800 (LE)
    data.unsafe_store(19, UInt8(0xB8))
    for j in range(16):
        data.unsafe_store(
            20 + j, UInt8(((15 - j) & 0xF) | (((15 - j) & 0xF) << 4))
        )

    # whole-tensor path
    var dst = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, 64))
    dequantize_into(2, data, 0, dst, 64)
    for j in range(16):
        var e0 = Float32(1.5) * (Float32(j) - Float32(8))
        var e1 = Float32(-0.5) * (Float32(15 - j) - Float32(8))
        if abs(Float32(dst.get(j)) - e0) > Float32(1e-6):
            print("FAIL: Q4_0 block0 low nibble at", j)
            raise Error("Q4_0 dequant mismatch (block0 low)")
        if abs(Float32(dst.get(16 + j)) - e0) > Float32(1e-6):
            print("FAIL: Q4_0 block0 high nibble at", j)
            raise Error("Q4_0 dequant mismatch (block0 high)")
        if abs(Float32(dst.get(32 + j)) - e1) > Float32(1e-6):
            print("FAIL: Q4_0 block1 low nibble at", j)
            raise Error("Q4_0 dequant mismatch (block1 low)")
        if abs(Float32(dst.get(48 + j)) - e1) > Float32(1e-6):
            print("FAIL: Q4_0 block1 high nibble at", j)
            raise Error("Q4_0 dequant mismatch (block1 high)")

    # block mode: one block at a time into a scratch buffer - must agree
    # element for element with the whole-tensor path.
    var scratch = unsafe_alloc[Scalar[DType.float16]](64)
    dequantize_blocks[DType.float16, QuantType.Q4_0](data, 0, scratch, 1)
    dequantize_blocks[DType.float16, QuantType.Q4_0](
        data, 18, scratch.unsafe_offset(32), 1
    )
    for i in range(64):
        if Float32(scratch.unsafe_load[width=1](offset=i)) != Float32(
            dst.get(i)
        ):
            print("FAIL: Q4_0 block mode differs at", i)
            raise Error("Q4_0 block-mode mismatch")
    data.unsafe_free()
    scratch.unsafe_free()
    print("Q4_0 kernel OK (whole-tensor + block mode)")


# -- 2. 27B memory budget (quantized-resident load) --------------------------


def check_27b_memory() raises:
    if not _file_exists(MODEL_27B):
        print("SKIP: 27B model not present")
        return
    var rss0 = process_rss_bytes()
    var res0 = process_resident_bytes()
    var ctx = load_gguf(MODEL_27B)
    var config = load_config(ctx)
    # M11 default: quantized-resident (no flag).  The legacy path would
    # dequantize every weight to fp16 here (~54 GB for this model).
    var model = TransformerModel(config, ctx^, 512)
    var (touched, checksum) = _touch_all(model.ctx)
    var rss1 = process_rss_bytes()
    var res1 = process_resident_bytes()
    print("27B quantized payload touched:", _gib(touched), "GiB")
    print("touch checksum (non-zero = pages really faulted in):", checksum)
    print("footprint before load:      ", _gib(rss0), "GiB")
    print("footprint after load+touch: ", _gib(rss1), "GiB")
    print("resident after load+touch:  ", _gib(res1), "GiB (incl. file pages)")
    # The committed (anonymous) footprint must stay small: the legacy
    # full-dequantize path materializes ~2x the payload as fp16 and would
    # blow this budget (and swap on a 64 GB machine).  The quantized
    # weights are file-backed, so they do NOT count against it.
    if rss1 >= 25 * 1024 * 1024 * 1024:
        print("FAIL: footprint after 27B load", _gib(rss1), "GiB >= 25 GiB")
        raise Error("27B memory budget exceeded")
    # The TOTAL physical RAM (footprint + the mmap'd file's resident pages)
    # must stay at the on-disk payload size, not 2x it.  Under memory
    # pressure the OS evicts clean file pages, so the resident delta can be
    # BELOW the payload - that is the reclaimable behavior we want, not a
    # failure.  It must never EXCEED payload + overhead.
    var res_delta = res1 - res0
    var budget = touched + 2 * 1024 * 1024 * 1024
    if res_delta > budget:
        print(
            "FAIL: resident delta",
            _gib(res_delta),
            "GiB exceeds payload + 2 GiB = ",
            _gib(budget),
            "GiB (weights NOT staying quantized-resident)",
        )
        raise Error("27B resident budget exceeded")
    print("27B memory OK (footprint < 25 GiB, resident ~= payload)")


# -- 3. quantized-resident vs dequantized forward ----------------------------


def check_forward_match(model_path: String, n_tokens: Int, tol: Float32) raises:
    if not _file_exists(model_path):
        print("SKIP:", model_path)
        return
    # quantized-resident (the M11 default)
    var ctx1 = load_gguf(model_path)
    var config1 = load_config(ctx1)
    var m1 = TransformerModel(config1, ctx1^, 512)
    var logits1 = m1.forward(100, 0)
    for i in range(1, n_tokens):
        logits1 = m1.forward(100, i)
    # legacy full-dequantize path
    var ctx2 = load_gguf(model_path)
    var config2 = load_config(ctx2)
    var m2 = TransformerModel(config2, ctx2^, 512, quant_resident=False)
    var logits2 = m2.forward(100, 0)
    for i in range(1, n_tokens):
        logits2 = m2.forward(100, i)
    var d = _max_logit_diff(logits1, logits2)
    print(model_path, "max |logit diff| quant vs dequant:", d)
    if d > tol:
        print("FAIL: quantized-resident forward diverges (", d, " > ", tol, ")")
        raise Error("quantized vs dequantized logit mismatch")
    print(model_path, "forward match OK")


def main() raises:
    check_q4_0_kernel()
    check_27b_memory()
    check_forward_match(MODEL_7B, 1, Float32(1e-2))
    check_forward_match(MODEL_15B, 2, Float32(1e-3))
    print("test_gguf OK")
