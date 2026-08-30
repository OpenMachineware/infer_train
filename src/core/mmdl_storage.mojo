# core/mmdl_storage.mojo
#
# M7: the private `.mmdl` checkpoint format.
#
# Design (llama.cpp-compatible core):
#   * The file IS a valid GGUF v3 file: same magic, same typed KV metadata
#     section, same tensor table and 32-byte-aligned tensor data.  llama.cpp
#     (or any GGUF reader) can load the *weights* directly.
#   * Weights keep their GGUF names ("token_embd.weight",
#     "blk.N.attn_q.weight", ...) and are written in standard ggml types
#     (F32 / F16), so the weight section round-trips through llama.cpp.
#   * Training state rides along as extra tensors with reserved name
#     prefixes - "grad." (gradients), "opt.m." / "opt.v." (AdamW moments) -
#     plus checkpoint metadata KV pairs (infer_train.step,
#     infer_train.loss_history, infer_train.format, ...).  GGUF readers
#     ignore unknown tensors/KVs, so this stays compatible.
#   * Incremental updates:
#       - `save_checkpoint_incremental` re-serializes only the tensors
#         named in `changed`; unchanged tensors are bulk-copied
#         byte-for-byte from the previous file's mapping.
#       - `append_delta` supports in-place appends via MMDL delta chunks
#         (magic "MMDT") chained by an "MMDX" index trailer; readers apply
#         deltas on top of the GGUF core (last write wins).
#   * `load_checkpoint` restores weights + gradients + AdamW m/v state +
#     the step counter and loss history (resume training).
#   * `strip_to_gguf` exports the weights-only GGUF core for llama.cpp.
#
# Mojo 1.0 notes: `FileHandle.write` accepts Strings only, so binary data
# is staged into byte buffers and handed over as `String(unsafe_from_utf8=)`
# spans (the bytes pass through untouched).

from .tensor import Tensor, tensor_zeros
from .training import TrainModel
from .train_optimizer import AdamState
from .ops.base.op_interface import AnyTensor
from .gguf_loader import (
    load_gguf,
    GGUFContext,
    GGUFTensor,
    GGUFMetaValue,
    Reader,
)
from .utils import align_up, unimplemented
from std.io.file import FileHandle
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.collections import Span
from std.utils.static_tuple import StaticTuple
from std.memory.unsafe import bitcast

comptime MMDL_FORMAT_VERSION = 1
comptime GGUF_MAGIC: String = "GGUF"
comptime GGML_F32 = 0
comptime GGML_F16 = 1
comptime KV_UINT32 = 4
comptime KV_FLOAT32 = 6
comptime KV_STRING = 8
comptime KV_ARRAY = 9


struct CheckpointMeta:
    var step: Int
    var lr: Float32
    var loss_history: List[Float32]

    def __init__(out self):
        self.step = 0
        self.lr = Float32(1e-3)
        self.loss_history = List[Float32]()


# -- binary staging helpers ---------------------------------------------------


struct ByteBuf(Movable):
    var data: Pointer[UInt8, MutUntrackedOrigin]
    var length: Int
    var capacity: Int

    def __init__(out self, cap: Int = 256):
        var alloc_size = cap
        if alloc_size < 1:
            alloc_size = 1
        self.capacity = alloc_size
        self.data = unsafe_alloc[UInt8](alloc_size, alignment=64)
        self.length = 0

    def reserve(mut self, extra: Int):
        if self.length + extra <= self.capacity:
            return
        var new_cap = self.capacity * 2
        while new_cap < self.length + extra:
            new_cap *= 2
        var fresh = unsafe_alloc[UInt8](new_cap, alignment=64)
        for i in range(self.length):
            fresh.unsafe_store(i, self.data.unsafe_load(offset=i))
        self.data.unsafe_free()
        self.data = fresh
        self.capacity = new_cap

    def append_u8(mut self, v: UInt8):
        self.reserve(1)
        self.data.unsafe_store(self.length, v)
        self.length += 1

    def append_u32(mut self, v: UInt32):
        self.reserve(4)
        for i in range(4):
            self.data.unsafe_store(
                self.length + i, UInt8((v >> UInt32(8 * i)) & 0xFF)
            )
        self.length += 4

    def append_u64(mut self, v: UInt64):
        self.reserve(8)
        for i in range(8):
            self.data.unsafe_store(
                self.length + i, UInt8((v >> UInt64(8 * i)) & 0xFF)
            )
        self.length += 8

    def append_bytes[O: Origin](mut self, bytes: Span[UInt8, O]):
        self.reserve(len(bytes))
        for i in range(len(bytes)):
            self.data.unsafe_store(self.length + i, bytes[i])
        self.length += len(bytes)

    def append_string(mut self, s: String):
        var bytes = s.as_bytes()
        self.append_u64(UInt64(len(bytes)))
        self.append_bytes(bytes)

    def pad_to(mut self, alignment: Int):
        var target = align_up(self.length, alignment)
        self.reserve(target - self.length)
        for i in range(target - self.length):
            self.data.unsafe_store(self.length + i, UInt8(0))
        self.length = target

    def as_string(self) -> String:
        var span = Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=self.data, length=self.length
        )
        return String(unsafe_from_utf8=span)


def _write_to_file(mut handle: FileHandle, buf: ByteBuf) raises:
    _ = handle.write(buf.as_string())


def _append_f32_2d(mut buf: ByteBuf, t: Tensor[DType.float32, 2]):
    var n = t.numel()
    buf.reserve(n * 4)
    var out = buf.data.unsafe_offset(buf.length).unsafe_bitcast[
        Scalar[DType.float32]
    ]()
    for i in range(n):
        out.unsafe_store(i, t.get(i))
    buf.length += n * 4


def _append_f32_1d(mut buf: ByteBuf, t: Tensor[DType.float32, 1]):
    var n = t.numel()
    buf.reserve(n * 4)
    var out = buf.data.unsafe_offset(buf.length).unsafe_bitcast[
        Scalar[DType.float32]
    ]()
    for i in range(n):
        out.unsafe_store(i, t.get(i))
    buf.length += n * 4


def _append_f32_any(mut buf: ByteBuf, any: AnyTensor):
    var n = any.numel
    buf.reserve(n * 4)
    var out = buf.data.unsafe_offset(buf.length).unsafe_bitcast[
        Scalar[DType.float32]
    ]()
    var src = any.data.unsafe_bitcast[Scalar[DType.float32]]()
    for i in range(n):
        out.unsafe_store(i, src.unsafe_load[width=1](offset=i))
    buf.length += n * 4


# -- GGUF writer --------------------------------------------------------------


def _write_gguf(
    mut handle: FileHandle,
    kv_buf: ByteBuf,
    tensor_names: List[String],
    tensor_dims: List[List[Int]],
    tensor_types: List[Int],
    tensor_bytes: List[ByteBuf],
) raises:
    """Assemble a GGUF v3 file from pre-staged KV/tensor buffers."""
    var header = ByteBuf(1024)
    var magic = GGUF_MAGIC.as_bytes()
    header.append_bytes(magic)
    header.append_u32(UInt32(3))
    header.append_u64(UInt64(len(tensor_names)))
    header.append_u64(UInt64(_count_kvs(kv_buf)))
    header.append_bytes(
        Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=kv_buf.data, length=kv_buf.length
        )
    )

    var table = ByteBuf(1024)
    var running = 0
    for i in range(len(tensor_names)):
        table.append_string(tensor_names[i])
        table.append_u32(UInt32(len(tensor_dims[i])))
        for d in tensor_dims[i]:
            table.append_u64(UInt64(d))
        table.append_u32(UInt32(tensor_types[i]))
        table.append_u64(UInt64(running))
        running += tensor_bytes[i].length
        running = align_up(running, 32)
    header.append_bytes(
        Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=table.data, length=table.length
        )
    )
    header.pad_to(32)

    for i in range(len(tensor_names)):
        header.append_bytes(
            Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=tensor_bytes[i].data,
                length=tensor_bytes[i].length,
            )
        )
        header.pad_to(32)
    _write_to_file(handle, header)


def _count_kvs(buf: ByteBuf) -> Int:
    var offset = 0
    var count = 0
    while offset < buf.length:
        var key_len = Int(_read_u64_at(buf, offset))
        offset += 8 + key_len
        var value_type = Int(_read_u32_at(buf, offset))
        offset += 4
        if value_type == KV_UINT32 or value_type == KV_FLOAT32:
            offset += 4
        elif value_type == KV_STRING:
            var s_len = Int(_read_u64_at(buf, offset))
            offset += 8 + s_len
        elif value_type == KV_ARRAY:
            var arr_type = Int(_read_u32_at(buf, offset))
            offset += 4
            var arr_len = Int(_read_u64_at(buf, offset))
            offset += 8
            if arr_type == KV_FLOAT32 or arr_type == KV_UINT32:
                offset += arr_len * 4
            elif arr_type == KV_STRING:
                for _ in range(arr_len):
                    var s2_len = Int(_read_u64_at(buf, offset))
                    offset += 8 + s2_len
            else:
                unimplemented("_count_kvs: unsupported array type")
        else:
            unimplemented("_count_kvs: unsupported kv type")
        count += 1
    return count


def _read_u32_at(buf: ByteBuf, offset: Int) -> UInt32:
    var v = UInt32(0)
    for i in range(4):
        v |= UInt32(buf.data.unsafe_load(offset=offset + i)) << UInt32(8 * i)
    return v


def _read_u64_at(buf: ByteBuf, offset: Int) -> UInt64:
    var v = UInt64(0)
    for i in range(8):
        v |= UInt64(buf.data.unsafe_load(offset=offset + i)) << UInt64(8 * i)
    return v


def _kv_uint32(mut buf: ByteBuf, key: String, v: Int):
    buf.append_string(key)
    buf.append_u32(UInt32(KV_UINT32))
    buf.append_u32(UInt32(v))


def _kv_float32(mut buf: ByteBuf, key: String, v: Float32):
    buf.append_string(key)
    buf.append_u32(UInt32(KV_FLOAT32))
    buf.append_u32(bitcast[DType.uint32](v))


def _kv_string(mut buf: ByteBuf, key: String, v: String):
    buf.append_string(key)
    buf.append_u32(UInt32(KV_STRING))
    buf.append_string(v)


def _kv_f32_array(mut buf: ByteBuf, key: String, values: List[Float32]):
    buf.append_string(key)
    buf.append_u32(UInt32(KV_ARRAY))
    buf.append_u32(UInt32(KV_FLOAT32))
    buf.append_u64(UInt64(len(values)))
    for v in values:
        buf.append_u32(bitcast[DType.uint32](v))


# -- TrainModel parameter tables ---------------------------------------------


def _collect_params(
    model: TrainModel,
    mut names: List[String],
    mut dims: List[List[Int]],
    mut tensors: List[Tensor[DType.float32, 2]],
):
    """Flatten every fp32 master parameter with a GGUF-style name.

    Rank-1 parameters are widened to [1, n] for serialization (GGUF tensor
    ranks are arbitrary; the loader restores the rank from the config).
    """
    var cfg = model.cfg
    names.append("token_embd.weight")
    dims.append(_dims2(cfg.hidden, cfg.vocab))
    tensors.append(model.token_embd)
    for l in range(cfg.n_layers):
        var base = "blk." + String(l) + "."
        names.append(base + "attn_norm.weight")
        dims.append(_dims1(cfg.hidden))
        tensors.append(_widen(model.layers[l].attn_norm_w, cfg.hidden))
        names.append(base + "attn_q.weight")
        dims.append(_dims2(cfg.hidden, cfg.n_heads * cfg.head_dim))
        tensors.append(model.layers[l].q_w)
        names.append(base + "attn_k.weight")
        dims.append(_dims2(cfg.hidden, cfg.n_kv_heads * cfg.head_dim))
        tensors.append(model.layers[l].k_w)
        names.append(base + "attn_v.weight")
        dims.append(_dims2(cfg.hidden, cfg.n_kv_heads * cfg.head_dim))
        tensors.append(model.layers[l].v_w)
        names.append(base + "attn_output.weight")
        dims.append(_dims2(cfg.n_heads * cfg.head_dim, cfg.hidden))
        tensors.append(model.layers[l].o_w)
        names.append(base + "attn_q.bias")
        dims.append(_dims1(cfg.n_heads * cfg.head_dim))
        tensors.append(_widen(model.layers[l].q_b, cfg.n_heads * cfg.head_dim))
        names.append(base + "attn_k.bias")
        dims.append(_dims1(cfg.n_kv_heads * cfg.head_dim))
        tensors.append(
            _widen(model.layers[l].k_b, cfg.n_kv_heads * cfg.head_dim)
        )
        names.append(base + "attn_v.bias")
        dims.append(_dims1(cfg.n_kv_heads * cfg.head_dim))
        tensors.append(
            _widen(model.layers[l].v_b, cfg.n_kv_heads * cfg.head_dim)
        )
        names.append(base + "ffn_norm.weight")
        dims.append(_dims1(cfg.hidden))
        tensors.append(_widen(model.layers[l].ffn_norm_w, cfg.hidden))
        names.append(base + "ffn_gate.weight")
        dims.append(_dims2(cfg.hidden, cfg.ffn))
        tensors.append(model.layers[l].gate_w)
        names.append(base + "ffn_up.weight")
        dims.append(_dims2(cfg.hidden, cfg.ffn))
        tensors.append(model.layers[l].up_w)
        names.append(base + "ffn_down.weight")
        dims.append(_dims2(cfg.ffn, cfg.hidden))
        tensors.append(model.layers[l].down_w)
    names.append("output_norm.weight")
    dims.append(_dims1(cfg.hidden))
    tensors.append(_widen(model.output_norm_w, cfg.hidden))
    names.append("output.weight")
    dims.append(_dims2(cfg.hidden, cfg.vocab))
    tensors.append(model.output_w)


def _widen(t: Tensor[DType.float32, 1], n: Int) -> Tensor[DType.float32, 2]:
    return t.reshape[2](StaticTuple[Int, 2](1, n))


def _copy_int_list(src: List[Int]) -> List[Int]:
    var out = List[Int]()
    for v in src:
        out.append(v)
    return out^


def _dims1(n: Int) -> List[Int]:
    var l = List[Int]()
    l.append(n)
    return l^


def _dims2(a: Int, b: Int) -> List[Int]:
    var l = List[Int]()
    l.append(a)
    l.append(b)
    return l^


# -- save ---------------------------------------------------------------------


def save_checkpoint(
    model: TrainModel,
    path: String,
    step: Int,
    loss_history: List[Float32] = List[Float32](),
) raises:
    """Write a full checkpoint: weights + gradients + AdamW m/v + metadata."""
    var kv = ByteBuf(1024)
    _kv_string(kv, "general.architecture", "infer-train")
    _kv_string(kv, "infer_train.format", "mmdl")
    _kv_uint32(kv, "infer_train.format_version", MMDL_FORMAT_VERSION)
    _kv_uint32(kv, "infer_train.step", step)
    _kv_float32(kv, "infer_train.lr", model.opt.groups[0].lr)
    if len(loss_history) > 0:
        _kv_f32_array(kv, "infer_train.loss_history", loss_history)

    var names = List[String]()
    var dims = List[List[Int]]()
    var types = List[Int]()
    var bytes = List[ByteBuf]()
    var pnames = List[String]()
    var pdims = List[List[Int]]()
    var ptensors = List[Tensor[DType.float32, 2]]()
    _collect_params(model, pnames, pdims, ptensors)
    for i in range(len(pnames)):
        names.append(pnames[i])
        dims.append(_copy_int_list(pdims[i]))
        types.append(GGML_F32)
        var b = ByteBuf(64)
        _append_f32_2d(b, ptensors[i])
        bytes.append(b^)

    for i in range(len(model.opt.groups[0].entries)):
        var g = model.opt.groups[0].entries[i].grad
        if g.numel > 0 and g.dtype == DType.float32:
            names.append("grad." + pnames[i])
            dims.append(_copy_int_list(pdims[i]))
            types.append(GGML_F32)
            var b = ByteBuf(64)
            _append_f32_any(b, g)
            bytes.append(b^)
        var st_ptr = model.opt.groups[0].entries[i].state.unsafe_bitcast[AdamState]()
        if st_ptr[0].m.numel > 0:
            names.append("opt.m." + pnames[i])
            dims.append(_copy_int_list(pdims[i]))
            types.append(GGML_F32)
            var bm = ByteBuf(64)
            _append_f32_any(bm, st_ptr[0].m)
            bytes.append(bm^)
        if st_ptr[0].v.numel > 0:
            names.append("opt.v." + pnames[i])
            dims.append(_copy_int_list(pdims[i]))
            types.append(GGML_F32)
            var bv = ByteBuf(64)
            _append_f32_any(bv, st_ptr[0].v)
            bytes.append(bv^)

    var handle = FileHandle(path, "w")
    _write_gguf(handle, kv, names, dims, types, bytes)
    handle.close()


def save_checkpoint_incremental(
    model: TrainModel,
    path: String,
    step: Int,
    changed: List[String],
    loss_history: List[Float32] = List[Float32](),
) raises:
    """Incremental checkpoint: only the tensors whose names appear in
    `changed` are re-serialized; every other tensor is copied byte-for-byte
    from the previous file through its memory mapping (no re-serialization).
    """
    var old = load_gguf(path)
    var kv = ByteBuf(1024)
    _kv_string(kv, "general.architecture", "infer-train")
    _kv_string(kv, "infer_train.format", "mmdl")
    _kv_uint32(kv, "infer_train.format_version", MMDL_FORMAT_VERSION)
    _kv_uint32(kv, "infer_train.step", step)
    _kv_float32(kv, "infer_train.lr", model.opt.groups[0].lr)
    if len(loss_history) > 0:
        _kv_f32_array(kv, "infer_train.loss_history", loss_history)

    var names = List[String]()
    var dims = List[List[Int]]()
    var types = List[Int]()
    var bytes = List[ByteBuf]()
    var pnames = List[String]()
    var pdims = List[List[Int]]()
    var ptensors = List[Tensor[DType.float32, 2]]()
    _collect_params(model, pnames, pdims, ptensors)
    var changed_set = Dict[String, Bool]()
    for c in changed:
        changed_set[c] = True
    for i in range(len(pnames)):
        names.append(pnames[i])
        dims.append(_copy_int_list(pdims[i]))
        var b = ByteBuf(64)
        if changed_set.get(pnames[i], False):
            types.append(GGML_F32)
            _append_f32_2d(b, ptensors[i])
        else:
            # bulk-copy the previous file's raw bytes for this tensor
            var old_t = _find_old(old, pnames[i])
            if old_t:
                types.append(old_t.value().ggml_type)
                var numel = _tensor_numel(old_t.value())
                var raw = old.tensor_data_ptr(old_t.value())
                b.append_bytes(
                    Span[UInt8, MutUntrackedOrigin](
                        unsafe_ptr=raw, length=_ggml_bytes(
                            old_t.value().ggml_type, numel
                        )
                    )
                )
            else:
                types.append(GGML_F32)
                _append_f32_2d(b, ptensors[i])
        bytes.append(b^)

    for i in range(len(model.opt.groups[0].entries)):
        var g = model.opt.groups[0].entries[i].grad
        if g.numel > 0 and g.dtype == DType.float32:
            names.append("grad." + pnames[i])
            dims.append(_copy_int_list(pdims[i]))
            types.append(GGML_F32)
            var b = ByteBuf(64)
            _append_f32_any(b, g)
            bytes.append(b^)
        var st_ptr2 = model.opt.groups[0].entries[i].state.unsafe_bitcast[AdamState]()
        if st_ptr2[0].m.numel > 0:
            names.append("opt.m." + pnames[i])
            dims.append(_copy_int_list(pdims[i]))
            types.append(GGML_F32)
            var bm = ByteBuf(64)
            _append_f32_any(bm, st_ptr2[0].m)
            bytes.append(bm^)
        if st_ptr2[0].v.numel > 0:
            names.append("opt.v." + pnames[i])
            dims.append(_copy_int_list(pdims[i]))
            types.append(GGML_F32)
            var bv = ByteBuf(64)
            _append_f32_any(bv, st_ptr2[0].v)
            bytes.append(bv^)

    var handle = FileHandle(path, "w")
    _write_gguf(handle, kv, names, dims, types, bytes)
    handle.close()


def _find_old(ctx: GGUFContext, name: String) -> Optional[GGUFTensor]:
    for t in ctx.tensors:
        if t.name == name:
            return t
    return None


def _tensor_numel(t: GGUFTensor) -> Int:
    var n = 1
    for i in range(t.n_dims):
        n *= t.dims[i]
    return n


def _ggml_bytes(ggml_type: Int, numel: Int) -> Int:
    if ggml_type == GGML_F32:
        return numel * 4
    if ggml_type == GGML_F16:
        return numel * 2
    # block-quantized types the loader supports (Q4_K/Q5_K/Q6_K/Q8_0/...)
    if ggml_type == 12:  # Q4_K
        return (numel // 256) * 144
    if ggml_type == 13:  # Q5_K
        return (numel // 256) * 176
    if ggml_type == 14:  # Q6_K
        return (numel // 256) * 210
    if ggml_type == 8:  # Q8_0
        return (numel // 32) * 34
    unimplemented("_ggml_bytes: unsupported type")
    return 0


# -- delta chunks (in-place append) -------------------------------------------


def append_delta(
    path: String, name: String, values: Tensor[DType.float32, 2]
) raises:
    """Append an MMDL delta chunk for one tensor to an existing .mmdl file.

    Layout: [magic "MMDT"][u32 version][u64 tensor count][tensor table]
    [32B-aligned data]["MMDX" index trailer at EOF].
    """
    var chunk = ByteBuf(64)
    var magic = String("MMDT").as_bytes()
    chunk.append_bytes(magic)
    chunk.append_u32(UInt32(1))
    chunk.append_u64(UInt64(1))
    chunk.append_string(name)
    chunk.append_u32(UInt32(2))
    chunk.append_u64(UInt64(values.shape()[0]))
    chunk.append_u64(UInt64(values.shape()[1]))
    chunk.append_u32(UInt32(GGML_F32))
    chunk.append_u64(UInt64(0))
    chunk.pad_to(32)
    _append_f32_2d(chunk, values)
    chunk.pad_to(32)

    var handle_r = FileHandle(path, "r")
    var size = Int(handle_r.seek(0, UInt8(2)))
    _ = handle_r.seek(0, UInt8(0))
    var file_buf = unsafe_alloc[UInt8](size)
    var file_bytes = Span[UInt8, MutUntrackedOrigin](
        unsafe_ptr=file_buf, length=size
    )
    _ = handle_r.read[DType.uint8](file_bytes)
    handle_r.close()

    var out = ByteBuf(size + chunk.length + 64)
    out.append_bytes(file_bytes)
    # strip the previous "MMDX" trailer (16 bytes) if present, then append
    if size >= 16:
        var tail_ok = True
        for i in range(4):
            if file_bytes[size - 16 + i] != magic[i]:
                tail_ok = False
        if tail_ok:
            out.length -= 16
    var chunk_span = Span[UInt8, MutUntrackedOrigin](
        unsafe_ptr=chunk.data, length=chunk.length
    )
    out.append_bytes(chunk_span)
    # index: [u32 chunk count][u64 chunk offset]...
    var index = ByteBuf(64)
    index.append_u32(UInt32(1))
    index.append_u64(UInt64(out.length))
    var index_span = Span[UInt8, MutUntrackedOrigin](
        unsafe_ptr=index.data, length=index.length
    )
    out.append_bytes(index_span)
    # trailer: "MMDX" + u32 index_offset + u32 version + u32 reserved
    var trail = ByteBuf(64)
    var mdx = String("MMDX").as_bytes()
    trail.append_bytes(mdx)
    trail.append_u32(UInt32(out.length - index.length))
    trail.append_u32(UInt32(1))
    trail.append_u32(UInt32(0))
    var trail_span = Span[UInt8, MutUntrackedOrigin](
        unsafe_ptr=trail.data, length=trail.length
    )
    out.append_bytes(trail_span)
    var handle2 = FileHandle(path, "w")
    _ = handle2.write(out.as_string())
    handle2.close()


def _find_delta_offsets(ctx: GGUFContext, data: Pointer[UInt8, MutUntrackedOrigin], size: Int) -> List[Int]:
    """Read the MMDX trailer and return the MMDT chunk offsets."""
    _ = ctx
    var out = List[Int]()
    if size < 16:
        return out^
    var tail = data.unsafe_offset(size - 16)
    var m = String("MMDX").as_bytes()
    for i in range(4):
        if tail.unsafe_load[width=1](offset=i) != m[i]:
            return out^
    var index_offset = Int(
        _read_u32_ptr(tail, 4)
    )
    if index_offset <= 0 or index_offset >= size:
        return out^
    var index = data.unsafe_offset(index_offset)
    var count = Int(_read_u32_ptr(index, 0))
    for i in range(count):
        out.append(Int(_read_u64_ptr(index, 4 + i * 8)))
    return out^


def _read_u32_ptr(p: Pointer[UInt8, MutUntrackedOrigin], off: Int) -> UInt32:
    var v = UInt32(0)
    for i in range(4):
        v |= UInt32(p.unsafe_load[width=1](offset=off + i)) << UInt32(8 * i)
    return v


def _read_u64_ptr(p: Pointer[UInt8, MutUntrackedOrigin], off: Int) -> UInt64:
    var v = UInt64(0)
    for i in range(8):
        v |= UInt64(p.unsafe_load[width=1](offset=off + i)) << UInt64(8 * i)
    return v


# -- load ---------------------------------------------------------------------


def load_checkpoint(
    mut model: TrainModel, path: String
) raises -> CheckpointMeta:
    """Restore weights, gradients, AdamW m/v and the training metadata."""
    var ctx = load_gguf(path)
    var meta = CheckpointMeta()
    meta.step = _meta_uint(ctx, "infer_train.step", 0)
    meta.lr = Float32(_meta_float(ctx, "infer_train.lr", 1e-3))
    meta.loss_history = _meta_f32_array(ctx, "infer_train.loss_history")

    # restore weights by name
    var cfg = model.cfg
    _restore2(model.token_embd, ctx, "token_embd.weight")
    for l in range(cfg.n_layers):
        var base = "blk." + String(l) + "."
        _restore1(model.layers[l].attn_norm_w, ctx, base + "attn_norm.weight")
        _restore2(model.layers[l].q_w, ctx, base + "attn_q.weight")
        _restore2(model.layers[l].k_w, ctx, base + "attn_k.weight")
        _restore2(model.layers[l].v_w, ctx, base + "attn_v.weight")
        _restore2(model.layers[l].o_w, ctx, base + "attn_output.weight")
        _restore1(model.layers[l].q_b, ctx, base + "attn_q.bias")
        _restore1(model.layers[l].k_b, ctx, base + "attn_k.bias")
        _restore1(model.layers[l].v_b, ctx, base + "attn_v.bias")
        _restore1(model.layers[l].ffn_norm_w, ctx, base + "ffn_norm.weight")
        _restore2(model.layers[l].gate_w, ctx, base + "ffn_gate.weight")
        _restore2(model.layers[l].up_w, ctx, base + "ffn_up.weight")
        _restore2(model.layers[l].down_w, ctx, base + "ffn_down.weight")
    _restore1(model.output_norm_w, ctx, "output_norm.weight")
    _restore2(model.output_w, ctx, "output.weight")

    # restore optimizer state (grad + m/v) by matching entry order
    var pnames = List[String]()
    var pdims = List[List[Int]]()
    var ptensors = List[Tensor[DType.float32, 2]]()
    _collect_params(model, pnames, pdims, ptensors)
    for i in range(len(model.opt.groups[0].entries)):
        if i < len(pnames):
            _restore_any(
                model.opt.groups[0].entries[i].grad, ctx, "grad." + pnames[i]
            )
            var st_ptr = model.opt.groups[0].entries[i].state.unsafe_bitcast[
                AdamState
            ]()
            _restore_any(st_ptr[0].m, ctx, "opt.m." + pnames[i])
            _restore_any(st_ptr[0].v, ctx, "opt.v." + pnames[i])
    model.opt.groups[0].lr = meta.lr
    return meta^


def _restore2(
    mut dst: Tensor[DType.float32, 2], ctx: GGUFContext, name: String
):
    var t = _find_old(ctx, name)
    if not t:
        return
    var numel = _tensor_numel(t.value())
    var src = ctx.tensor_data_ptr(t.value()).unsafe_bitcast[
        Scalar[DType.float32]
    ]()
    for i in range(numel):
        dst.set(i, src.unsafe_load[width=1](offset=i))


def _restore1(
    mut dst: Tensor[DType.float32, 1], ctx: GGUFContext, name: String
):
    var t = _find_old(ctx, name)
    if not t:
        return
    var numel = _tensor_numel(t.value())
    var src = ctx.tensor_data_ptr(t.value()).unsafe_bitcast[
        Scalar[DType.float32]
    ]()
    for i in range(numel):
        dst.set(i, src.unsafe_load[width=1](offset=i))


def _restore_any(mut dst: AnyTensor, ctx: GGUFContext, name: String):
    var t = _find_old(ctx, name)
    if not t:
        return
    var numel = _tensor_numel(t.value())
    var src = ctx.tensor_data_ptr(t.value()).unsafe_bitcast[
        Scalar[DType.float32]
    ]()
    var d = dst.data.unsafe_bitcast[Scalar[DType.float32]]()
    for i in range(numel):
        d.unsafe_store(i, src.unsafe_load[width=1](offset=i))


def _meta_uint(ctx: GGUFContext, key: String, default: Int) -> Int:
    var v = ctx.metadata.get(key, GGUFMetaValue())
    if v.kind == 0:
        return Int(v.uint_val)
    return default


def _meta_float(ctx: GGUFContext, key: String, default: Float64) -> Float64:
    var v = ctx.metadata.get(key, GGUFMetaValue())
    if v.kind == 2:
        return v.float_val
    return default


def _meta_f32_array(ctx: GGUFContext, key: String) -> List[Float32]:
    var v = ctx.metadata.get(key, GGUFMetaValue())
    if v.kind != 5 or v.arr_type != KV_FLOAT32:
        return List[Float32]()
    var reader = Reader(ctx.data)
    reader.offset = v.arr_offset
    var out = List[Float32]()
    for _ in range(v.arr_len):
        out.append(reader.read_f32())
    return out^


# -- strip --------------------------------------------------------------------


def strip_to_gguf(path_in: String, path_out: String) raises:
    """Export the weights-only GGUF core (drop grad/opt state tensors).

    The weight tensors are bulk-copied from the source mapping; the KV
    section is rebuilt from the parsed metadata (training keys dropped).
    """
    var old = load_gguf(path_in)
    var kv = ByteBuf(1024)
    # rebuild the KV section from the source metadata (skip training keys)
    for key in old.metadata.keys():
        if _str_starts(key, "infer_train."):
            continue
        _kv_from_meta(kv, key, old.metadata[key], old)
    var names = List[String]()
    var dims = List[List[Int]]()
    var types = List[Int]()
    var bytes = List[ByteBuf]()
    for t in old.tensors:
        if _str_starts(t.name, "grad.") or _str_starts(t.name, "opt."):
            continue
        names.append(t.name)
        var dl = List[Int]()
        for d in range(t.n_dims):
            dl.append(t.dims[d])
        dims.append(dl^)
        types.append(t.ggml_type)
        var numel = _tensor_numel(t)
        var b = ByteBuf(64)
        b.append_bytes(
            Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=old.tensor_data_ptr(t),
                length=_ggml_bytes(t.ggml_type, numel),
            )
        )
        bytes.append(b^)
    var handle = FileHandle(path_out, "w")
    _write_gguf(handle, kv, names, dims, types, bytes)
    handle.close()


def _kv_from_meta(
    mut buf: ByteBuf, key: String, v: GGUFMetaValue, ctx: GGUFContext
):
    _ = ctx
    if v.kind == 0:  # uint
        buf.append_string(key)
        buf.append_u32(UInt32(KV_UINT32))
        buf.append_u32(UInt32(v.uint_val))
    elif v.kind == 2:  # float
        buf.append_string(key)
        buf.append_u32(UInt32(KV_FLOAT32))
        var f = Float32(v.float_val)
        buf.append_u32(bitcast[DType.uint32](f))
    elif v.kind == 4:  # string
        buf.append_string(key)
        buf.append_u32(UInt32(KV_STRING))
        buf.append_string(v.str_val)
    else:
        return  # skip unsupported kinds (arrays/int/bool are dropped)


def _str_starts(s: String, prefix: String) -> Bool:
    var sb = s.as_bytes()
    var pb = prefix.as_bytes()
    if len(sb) < len(pb):
        return False
    for i in range(len(pb)):
        if sb[i] != pb[i]:
            return False
    return True
