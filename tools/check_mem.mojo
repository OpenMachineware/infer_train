# tools/check_mem.mojo
#
# Q4-resident (M11) memory verification for a single GGUF model.
#
# Usage: check_mem <model.gguf>
#
# Loads the model with the DEFAULT load mode (quantized-resident, no flag),
# faults in every mapped tensor page (mmap is lazy, so pages only become
# resident when touched), and reports the memory breakdown:
#
#   * quant payload: the sum of every tensor's on-disk (quantized) bytes
#   * phys_footprint: committed memory (anonymous heap + purgeable) - the
#     number that CANNOT be reclaimed without swap; must stay small
#     (KV cache + small fp16 vectors + process base).  NOTE: macOS
#     phys_footprint EXCLUDES clean file-backed pages, so it does NOT see
#     the mmap'd weights at all.
#   * resident (mach_task_basic_info.resident_size): TOTAL physical RAM
#     the process occupies, INCLUDING the mmap'd model file's resident
#     pages (reclaimable page cache); after a full touch it is ~= the
#     payload size
#
# Budgets asserted:
#   * footprint delta < 2 GiB  (no fp16 weight materialization; the legacy
#     full-dequantize path would add ~2x the payload of ANONYMOUS bytes)
#   * resident delta < payload + 2 GiB  (weights stay at on-disk size)
#
# Run each model in its OWN process: file-backed pages stay resident while
# the mapping is alive, so loading several models in one process would
# stack their footprints.

from src.core.gguf_loader import GGUFContext, load_gguf, ggml_quant_info
from src.core.transformer import TransformerModel, load_config
from src.core.memory import process_resident_bytes, process_rss_bytes
from std.sys import argv

comptime GIB: Int = 1024 * 1024 * 1024


def _gib(x: Int) -> Float64:
    return Float64(x) / (1024.0 * 1024.0 * 1024.0)


def _parts_size(ctx: GGUFContext) -> Int:
    """Total mapped bytes across every split part (the model's file size)."""
    var total = 0
    for part in ctx.parts:
        total += part.size
    return total


def _touch_all(ctx: GGUFContext) -> Tuple[Int, Int]:
    """Fault in every mapped tensor page; returns (payload bytes, checksum).

    The checksum is printed so dead-code elimination of the touch loops is
    visible (a zero checksum means the compiler optimized the reads away).
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


def main() raises:
    var arg_list = List[String]()
    for a in argv():
        arg_list.append(String(a))
    if len(arg_list) < 2:
        print("usage: check_mem <model.gguf>")
        return
    var path = arg_list[1]

    var fp0 = process_rss_bytes()
    var res0 = process_resident_bytes()
    var ctx = load_gguf(path)
    var fsize = _parts_size(ctx)
    var config = load_config(ctx)
    var model = TransformerModel(config, ctx^, 512)
    var (touched, checksum) = _touch_all(model.ctx)
    var fp1 = process_rss_bytes()
    var res1 = process_resident_bytes()

    var fp_delta = fp1 - fp0
    var res_delta = res1 - res0

    print("model:            ", path)
    print("arch:             ", config.arch_str, " layers:", config.n_layers)
    print("file size:        ", _gib(fsize), "GiB")
    print("quant payload:    ", _gib(touched), "GiB")
    print("touch checksum:   ", checksum)
    print("footprint before: ", _gib(fp0), "GiB")
    print("footprint after:  ", _gib(fp1), "GiB (delta ", _gib(fp_delta), "GiB)")
    print("resident after:   ", _gib(res1), "GiB (delta ", _gib(res_delta), "GiB)")
    print("fp16 would be:    ", _gib(2 * touched), "GiB (legacy full-dequantize)")

    var ok = True
    if fp_delta > 2 * GIB:
        print(
            "FAIL: footprint delta",
            _gib(fp_delta),
            "GiB > 2 GiB - weights were MATERIALIZED (not Q4-resident)",
        )
        ok = False
    var budget = touched + 2 * GIB
    if res_delta > budget:
        print(
            "FAIL: resident delta",
            _gib(res_delta),
            "GiB exceeds payload + 2 GiB = ",
            _gib(budget),
            "GiB",
        )
        ok = False
    if res_delta < touched // 2:
        print(
            "WARN: only",
            _gib(res_delta),
            "GiB resident after touch (expected ~",
            _gib(touched), "GiB) - touch loop may have been optimized away",
        )
    if ok:
        print("OK: quantized-resident (footprint small, resident ~= payload)")
    else:
        raise Error("memory budget exceeded")
