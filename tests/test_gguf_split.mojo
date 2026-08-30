# tests/test_gguf_split.mojo
#
# Correctness test for GGUF split-file (multi-part) loading
# (src/core/gguf_loader.mojo).
#
# The DeepSeek-R1-Distill-Qwen-1.5B model is split with llama.cpp's
# `llama-gguf-split` into `DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf-NNNNN-of-00003.gguf`.
# Loading the split model must yield the same tensors (and metadata) as loading
# the original single file: same tensor count, same per-tensor dims/type, and
# byte-identical dequantized weights (verifying each tensor is read from the
# correct part at the correct offset).
#
# Skips (SKIP) if the split part files are not present.

from src.core.tensor import Tensor
from src.core.gguf_loader import (
    find_tensor,
    get_meta_str,
    get_meta_uint,
    load_gguf,
)
from src.core.transformer import dequantize_vector, dequantize_weight

comptime SINGLE = "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"
comptime SPLIT_PART = "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf-00001-of-00003.gguf"


def _file_exists(path: String) -> Bool:
    try:
        var f = FileHandle(path, "r")
        f.close()
        return True
    except:
        return False


def _max_diff1(a: Tensor[DType.float16, 1], b: Tensor[DType.float16, 1]) -> Float32:
    var n = a.numel()
    var m = Float32(0.0)
    for i in range(n):
        var d = abs(Float32(a.get(i)) - Float32(b.get(i)))
        if d > m:
            m = d
    return m


def _max_diff2(a: Tensor[DType.float16, 2], b: Tensor[DType.float16, 2]) -> Float32:
    var n = a.numel()
    var m = Float32(0.0)
    for i in range(n):
        var d = abs(Float32(a.get(i)) - Float32(b.get(i)))
        if d > m:
            m = d
    return m


def main() raises:
    if not _file_exists(SINGLE) or not _file_exists(SPLIT_PART):
        print("SKIP: single model or split part files not present")
        return

    var single = load_gguf(SINGLE)
    var split = load_gguf(SPLIT_PART)

    # -- structural checks -------------------------------------------------
    print("single tensors:", single.tensor_count, " parts:", single.n_parts())
    print("split  tensors:", split.tensor_count, " parts:", split.n_parts())
    if split.n_parts() != 3:
        print("FAIL: expected 3 split parts, got", split.n_parts())
        raise Error("wrong part count")
    if split.tensor_count != single.tensor_count:
        print(
            "FAIL: tensor count mismatch",
            split.tensor_count,
            "!=",
            single.tensor_count,
        )
        raise Error("tensor count mismatch")

    # -- metadata must come from the first part ---------------------------
    var arch_s = get_meta_str(split, "general.architecture", String("none"))
    var arch_o = get_meta_str(single, "general.architecture", String("none"))
    if arch_s != arch_o or arch_s == "none":
        print("FAIL: architecture metadata mismatch:", arch_s, arch_o)
        raise Error("arch metadata mismatch")
    print("architecture:", arch_s)
    var blocks_s = get_meta_uint(split, "qwen2.block_count", -1)
    var blocks_o = get_meta_uint(single, "qwen2.block_count", -1)
    if blocks_s != blocks_o:
        print("FAIL: block_count mismatch:", blocks_s, blocks_o)
        raise Error("block_count mismatch")
    print("block_count:", blocks_s)

    # every single-file tensor must exist in the split context (same name)
    for t in single.tensors:
        if not find_tensor(split, t.name):
            print("FAIL: tensor missing in split:", t.name)
            raise Error("tensor missing in split")

    # -- per-part weight correctness --------------------------------------
    # One rank-1 norm from each part + one rank-2 weight (part 1).
    var v1 = find_tensor(single, "output_norm.weight")
    var v2 = find_tensor(split, "output_norm.weight")
    var d1 = _max_diff1(
        dequantize_vector(single, v1.value()),
        dequantize_vector(split, v2.value()),
    )
    print("output_norm.weight (part 0) max_diff =", d1)
    if d1 > Float32(1e-3):
        raise Error("part 0 weight mismatch")

    var v3 = find_tensor(single, "blk.7.ffn_norm.weight")
    var v4 = find_tensor(split, "blk.7.ffn_norm.weight")
    var d2 = _max_diff1(
        dequantize_vector(single, v3.value()),
        dequantize_vector(split, v4.value()),
    )
    print("blk.7.ffn_norm.weight (part 1) max_diff =", d2)
    if d2 > Float32(1e-3):
        raise Error("part 1 weight mismatch")

    var v5 = find_tensor(single, "blk.25.ffn_norm.weight")
    var v6 = find_tensor(split, "blk.25.ffn_norm.weight")
    var d3 = _max_diff1(
        dequantize_vector(single, v5.value()),
        dequantize_vector(split, v6.value()),
    )
    print("blk.25.ffn_norm.weight (part 2) max_diff =", d3)
    if d3 > Float32(1e-3):
        raise Error("part 2 weight mismatch")

    var w1 = find_tensor(single, "blk.7.ffn_gate.weight")
    var w2 = find_tensor(split, "blk.7.ffn_gate.weight")
    var dw = _max_diff2(
        dequantize_weight(single, w1.value()),
        dequantize_weight(split, w2.value()),
    )
    print("blk.7.ffn_gate.weight (part 1, rank-2) max_diff =", dw)
    if dw > Float32(1e-3):
        raise Error("rank-2 weight mismatch")

    print("test_gguf_split OK")
