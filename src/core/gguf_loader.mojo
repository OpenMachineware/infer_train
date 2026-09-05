# core/gguf_loader.mojo
#
# GGUF model file parser.
#
# Reads the GGUF header (magic / version / tensor count / metadata count),
# parses the typed metadata key-value pairs into a tagged union (the same
# "AttrValue" scheme used by the graph), extracts the weight tensor info
# (name / n_dims / dims / ggml type / data offset), and exposes the raw
# tensor bytes through the memory-mapped file buffer.
#
# The whole file is memory-mapped via `memory.mmap_file`; tensor payloads are
# addressed as offsets into that buffer (zero copy on the host).

from .memory import mmap_file, munmap_file
from .utils import align_up, unimplemented
from .tensor import Tensor
from .ops.quantized.quant_types import QuantType
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.collections import Span
from std.utils.static_tuple import StaticTuple
from std.memory.unsafe import bitcast
from std.memory.alloc import unsafe_alloc

# GGUF value types (see the GGUF spec).
comptime GGUF_UINT8 = 0
comptime GGUF_INT8 = 1
comptime GGUF_UINT16 = 2
comptime GGUF_INT16 = 3
comptime GGUF_UINT32 = 4
comptime GGUF_INT32 = 5
comptime GGUF_FLOAT32 = 6
comptime GGUF_BOOL = 7
comptime GGUF_STRING = 8
comptime GGUF_ARRAY = 9
comptime GGUF_UINT64 = 10
comptime GGUF_INT64 = 11
comptime GGUF_FLOAT64 = 12

# ggml tensor types (subset used by DeepSeek-R1-Distill-Qwen-1.5B).
comptime GGML_F32 = 0
comptime GGML_F16 = 1
comptime GGML_Q5_K = 13
comptime GGML_Q6_K = 14

# Maximum tensor rank we materialize (Qwen2 weights are rank <= 2).
comptime GGUF_MAX_DIMS = 4


struct GGUFMetaValue(Copyable, ImplicitlyCopyable, Movable):
    """Tagged union for a GGUF metadata value (see module docstring)."""

    var kind: Int8  # 0 uint, 1 int, 2 float, 3 bool, 4 string, 5 array
    var uint_val: UInt64
    var int_val: Int64
    var float_val: Float64
    var bool_val: Bool
    var str_val: String
    var arr_type: Int32
    var arr_len: Int
    var arr_offset: Int  # file offset of the array's first element

    def __init__(out self):
        self.kind = 0
        self.uint_val = 0
        self.int_val = 0
        self.float_val = 0
        self.bool_val = False
        self.str_val = String("")
        self.arr_type = 0
        self.arr_len = 0
        self.arr_offset = 0


struct GGUFTensor(Copyable, ImplicitlyCopyable, Movable):
    var name: String
    var n_dims: Int
    var dims: StaticTuple[Int, GGUF_MAX_DIMS]
    var ggml_type: Int
    var offset: Int  # byte offset into the owning file's tensor-data section
    var file_idx: Int  # index into GGUFContext.parts (0 for a single file)

    def __init__(out self):
        self.name = String("")
        self.n_dims = 0
        self.dims = StaticTuple[Int, GGUF_MAX_DIMS](fill=0)
        self.ggml_type = 0
        self.offset = 0
        self.file_idx = 0


struct GGUFQuantInfo(Copyable, ImplicitlyCopyable, Movable):
    """Per-tensor quantization metadata (the Q4-resident contract).

    Stored behind `Tensor.quantization_info` (an opaque `Pointer[UInt8]`)
    by `GGUFContext.load_tensor`.  `quant_type` is the `QuantType` tag of
    the comptime-specialized dequantizer/matmul kernel (-1 when the format
    has no block kernel: F16/F32, IQ4_NL, NF4).  For every GGUF block
    format the per-block scales (and mins) live *inside* the quantized
    bytes, so the global `scale` is 0.0 and `group_size` is the sub-block
    (scale-group) element count.
    """

    var ggml_type: Int  # raw GGUF ggml type (2 Q4_0, 12 Q4_K, 13 Q5_K, ...)
    var quant_type: Int8  # QuantType tag, or -1 (no comptime block kernel)
    var group_size: Int  # elements per scale group (32 for block formats)
    var block_bytes: Int  # bytes per block (18 Q4_0, 144 Q4_K, 176 Q5_K, ...)
    var block_elems: Int  # elements per block (32 Q4_0/Q8_0, 256 K formats)
    var scale: Float32  # global scale (0.0: scales are packed per block)

    def __init__(out self):
        self.ggml_type = 0
        self.quant_type = Int8(-1)
        self.group_size = 0
        self.block_bytes = 0
        self.block_elems = 0
        self.scale = Float32(0.0)


def ggml_quant_info(ggml_type: Int) -> Tuple[Int, Int, Int, Int8]:
    """(block_elems, block_bytes, group_size, quant_type tag) for a GGUF
    ggml type.

    Block sizes mirror llama.cpp's `ggml-quants.c`.  The `quant_type` tag
    matches `QuantType` (ops/quantized/quant_types.mojo); -1 means the
    format has no comptime quantized-matmul kernel (F16/F32, IQ4_NL, NF4)
    and must be materialized to fp16 by the caller.
    """
    if ggml_type == 2:  # Q4_0
        return (32, 18, 32, QuantType.Q4_0._tag)
    if ggml_type == 12:  # Q4_K (Q4_K_M)
        return (256, 144, 32, QuantType.Q4_K_M._tag)
    if ggml_type == 13:  # Q5_K
        return (256, 176, 32, QuantType.Q5_K._tag)
    if ggml_type == 14:  # Q6_K
        return (256, 210, 32, QuantType.Q6_K._tag)
    if ggml_type == 8:  # Q8_0
        return (32, 34, 32, QuantType.Q8_0._tag)
    if ggml_type == 23:  # IQ4_XS
        return (256, 136, 32, QuantType.IQ4_XS._tag)
    if ggml_type == 20:  # IQ4_NL
        return (32, 18, 0, Int8(-1))
    if ggml_type == 30:  # NF4
        return (64, 34, 0, Int8(-1))
    if ggml_type == 0:  # F32
        return (1, 4, 0, Int8(-1))
    if ggml_type == 1:  # F16
        return (1, 2, 0, Int8(-1))
    unimplemented("gguf: unsupported ggml type " + String(ggml_type))
    return (1, 1, 0, Int8(-1))


struct GGUFFilePart(Copyable, ImplicitlyCopyable, Movable):
    """One memory-mapped split file of a (possibly multi-part) GGUF model.

    A single-file model is a one-element list of these.  `data_offset` is the
    byte offset (within THIS file's mapping) where the tensor payloads begin;
    a tensor's bytes live at `data.unsafe_offset(data_offset + tensor.offset)`.
    """

    var path: String
    var data: Pointer[UInt8, MutUntrackedOrigin]
    var size: Int
    var data_offset: Int

    def __init__(
        out self,
        path: String,
        data: Pointer[UInt8, MutUntrackedOrigin],
        size: Int,
        data_offset: Int,
    ):
        self.path = path
        self.data = data
        self.size = size
        self.data_offset = data_offset


struct Reader(Movable):
    """Little-endian byte reader over the mapped file."""

    var data: Pointer[UInt8, MutUntrackedOrigin]
    var offset: Int

    def __init__(out self, data: Pointer[UInt8, MutUntrackedOrigin]):
        self.data = data
        self.offset = 0

    def read_u8(mut self) -> UInt8:
        var value = self.data.unsafe_load[width=1](offset=self.offset)
        self.offset += 1
        return value

    def read_u16(mut self) -> UInt16:
        var b0 = UInt16(self.read_u8())
        var b1 = UInt16(self.read_u8())
        return b0 | (b1 << 8)

    def read_u32(mut self) -> UInt32:
        var b0 = UInt32(self.read_u8())
        var b1 = UInt32(self.read_u8())
        var b2 = UInt32(self.read_u8())
        var b3 = UInt32(self.read_u8())
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)

    def read_u64(mut self) -> UInt64:
        var low = UInt64(self.read_u32())
        var high = UInt64(self.read_u32())
        return low | (high << 32)

    def read_i32(mut self) -> Int32:
        return bitcast[DType.int32](self.read_u32())

    def read_f32(mut self) -> Float32:
        return bitcast[DType.float32](self.read_u32())

    def read_string(mut self) -> String:
        var length = Int(self.read_u64())
        var span = Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=self.data.unsafe_offset(self.offset), length=length
        )
        self.offset += length
        return String(unsafe_from_utf8=span)

    def skip(mut self, count: Int):
        self.offset += count


struct GGUFContext(Movable):
    var data: Pointer[UInt8, MutUntrackedOrigin]
    var size: Int
    var version: Int
    var tensor_count: Int
    var meta_count: Int
    var metadata: Dict[String, GGUFMetaValue]
    var tensors: List[GGUFTensor]
    var data_offset: Int  # byte offset where tensor payloads begin (part 0)
    var parts: List[GGUFFilePart]  # one per split file (>= 1)

    def __init__(out self, data: Pointer[UInt8, MutUntrackedOrigin], size: Int):
        self.data = data
        self.size = size
        self.version = 0
        self.tensor_count = 0
        self.meta_count = 0
        self.metadata = Dict[String, GGUFMetaValue]()
        self.tensors = List[GGUFTensor]()
        self.data_offset = 0
        self.parts = List[GGUFFilePart]()

    def n_parts(self) -> Int:
        return len(self.parts)

    def tensor_data(
        self, t: GGUFTensor
    ) -> Tuple[Pointer[UInt8, MutUntrackedOrigin], Int]:
        """(base pointer, byte offset) of `t`'s payload in its owning part."""
        var part = self.parts[t.file_idx]
        return (part.data, part.data_offset + t.offset)

    def tensor_data_ptr(
        self, t: GGUFTensor
    ) -> Pointer[UInt8, MutUntrackedOrigin]:
        """Base pointer directly at `t`'s payload bytes (owning part aware)."""
        var (base, off) = self.tensor_data(t)
        return base.unsafe_offset(off)

    def load_tensor(self, t: GGUFTensor) -> Tensor[DType.uint8, 2]:
        """Materialize `t` as a zero-copy `Tensor[UInt8, 2]` that KEEPS its
        on-disk layout (Q4-resident).

        No dequantization happens here: the bytes keep their quantized
        block layout (Q4_K_M / Q4_0 / Q5_K / Q6_K / Q8_0 / IQ4_XS) - or,
        for F16/F32 tensors, their raw 2/4-byte-per-element layout.  The
        quantization metadata (`quant_type`, `scale`, `group_size`, block
        sizes) is stored in the returned tensor's `quantization_info`
        field (a heap `GGUFQuantInfo`), so consumers such as
        `matmul_quantized_cpu` dequantize per block at compute time and
        the weight's resident footprint stays its on-disk size instead of
        doubling into fp16.

        Shape: rank-2 tensors come back as [dims[1], bytes_per_row] - the
        [out, in] layout the weight-major kernels expect (GGUF dims are
        ggml-ordered, innermost first); rank-1 tensors as [1, total_bytes].
        The storage is a view over the memory-mapped file: it must outlive
        the owning `GGUFContext` mapping (released by `munmap_all`).
        """
        if t.n_dims > 2:
            unimplemented("load_tensor: rank-3 tensors use GGUFTensor views")
        var numel = 1
        for d in range(t.n_dims):
            numel *= t.dims[d]
        var (be, bb, gs, tag) = ggml_quant_info(t.ggml_type)
        var shape: StaticTuple[Int, 2]
        if t.n_dims >= 2:
            shape = StaticTuple[Int, 2](
                t.dims[1], (numel // t.dims[1]) * bb // be
            )
        else:
            shape = StaticTuple[Int, 2](1, numel * bb // be)
        var ptr = self.tensor_data_ptr(t).unsafe_bitcast[Scalar[DType.uint8]]()
        var out = Tensor[DType.uint8, 2](shape, ptr)
        var info = unsafe_alloc[GGUFQuantInfo](1)
        info[unsafe_offset=0].ggml_type = t.ggml_type
        info[unsafe_offset=0].quant_type = tag
        info[unsafe_offset=0].group_size = gs
        info[unsafe_offset=0].block_bytes = bb
        info[unsafe_offset=0].block_elems = be
        info[unsafe_offset=0].scale = Float32(
            0.0
        )  # block formats pack scales per block
        out.set_quantization_info(info.unsafe_bitcast[UInt8]())
        return out

    def munmap_all(mut self):
        """Release every part's memory mapping (call once when done)."""
        for part in self.parts:
            munmap_file(part.data, part.size)


def _skip_meta_array(mut reader: Reader, elem_type: Int, length: Int):
    """Advance the reader past an array of `length` values."""
    if elem_type == GGUF_STRING:
        for _ in range(length):
            _ = reader.read_string()
    elif elem_type in (GGUF_UINT8, GGUF_INT8, GGUF_BOOL):
        reader.skip(length)
    elif elem_type in (GGUF_UINT16, GGUF_INT16):
        reader.skip(length * 2)
    elif elem_type in (GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32):
        reader.skip(length * 4)
    elif elem_type in (GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64):
        reader.skip(length * 8)
    else:
        reader.skip(length)  # unknown; best effort


# -- split-file (multi-part) support ------------------------------------------
#
# llama.cpp's `llama-gguf-split` writes `<base>.gguf-NNNNN-of-NNNNN.gguf`
# (5-digit zero-padded).  Every part is a valid GGUF file: the FIRST part
# carries the full metadata, the others only `split.no` / `split.tensors.count`
# / `split.count`.  Each part lists only its OWN tensors, and every tensor's
# `offset` is relative to that part's own tensor-data section.  Loading a split
# model therefore means mapping every part, taking the metadata from part 1,
# and concatenating the per-part tensor tables (tagging each tensor with the
# part that owns its bytes).

comptime GGUF_SPLIT_MAX = 64  # max part count probed for base-name lookup


def _fmt5(n: Int) -> String:
    """Zero-pad `n` to 5 digits (GGUF split part numbering)."""
    var s = String(n)
    while s.byte_length() < 5:
        s = "0" + s
    return s^


def _str_to_int(s: String) -> Int:
    var v = 0
    var b = s.as_bytes()
    for c in range(len(b)):
        var x = Int(b[c])
        if x >= 48 and x <= 57:
            v = v * 10 + (x - 48)
    return v


def _is_split_part_path(path: String) -> Bool:
    """True if `path` is a GGUF split part: `<base>.gguf-NNNNN-of-NNNNN.gguf`.
    """
    var of = path.rfind("-of-")
    if of < 6:
        return False
    if of + 14 != path.byte_length():
        return False
    var ext = String(path[byte = of + 9 : of + 14])
    if ext != ".gguf":
        return False
    var bytes_ = path.as_bytes()
    if Int(bytes_[of - 6]) != 45:  # '-' before the part number
        return False
    for c in range(of - 5, of):
        var x = Int(bytes_[c])
        if x < 48 or x > 57:
            return False
    for c in range(of + 4, of + 9):
        var x = Int(bytes_[c])
        if x < 48 or x > 57:
            return False
    return True


def _split_part_base_and_total(path: String) -> Tuple[String, Int]:
    """(base, total) for a split part path; base is `<...>.gguf`."""
    var of = path.rfind("-of-")
    var total = _str_to_int(String(path[byte = of + 4 : of + 9]))
    var base = String(path[byte = 0 : of - 6])
    return (base, total)


def _file_exists(path: String) -> Bool:
    try:
        var f = FileHandle(path, "r")
        f.close()
        return True
    except:
        return False


def _find_split_total(base: String) -> Int:
    """Probe `base-00001-of-NNNNN.gguf` for a small range of totals."""
    for total in range(1, GGUF_SPLIT_MAX + 1):
        var p = base + "-00001-of-" + _fmt5(total) + ".gguf"
        if _file_exists(p):
            return total
    return 0


def _parse_part(
    data: Pointer[UInt8, MutUntrackedOrigin],
    mut metadata: Dict[String, GGUFMetaValue],
    mut tensors: List[GGUFTensor],
    file_idx: Int,
) raises -> Tuple[Int, Int]:
    """Parse one GGUF file's header, metadata, and tensor table.

    Fills `metadata` and `tensors` in place and returns (data_offset, version).
    `file_idx` is stamped onto every tensor so the context can later locate
    each tensor's bytes in the correct split part.
    """
    var reader = Reader(data)
    if (
        reader.read_u8() != UInt8(71)  # 'G'
        or reader.read_u8() != UInt8(71)  # 'G'
        or reader.read_u8() != UInt8(85)  # 'U'
        or reader.read_u8() != UInt8(70)  # 'F'
    ):
        unimplemented("gguf: bad magic")

    var version = Int(reader.read_u32())
    var tensor_count = Int(reader.read_u64())
    var meta_count = Int(reader.read_u64())

    for _ in range(meta_count):
        var key = reader.read_string()
        var value_type = Int(reader.read_u32())
        var value = GGUFMetaValue()
        if value_type == GGUF_UINT8:
            value.kind = 0
            value.uint_val = UInt64(reader.read_u8())
        elif value_type == GGUF_INT8:
            value.kind = 1
            value.int_val = Int64(reader.read_u8())
        elif value_type == GGUF_UINT16:
            value.kind = 0
            value.uint_val = UInt64(reader.read_u16())
        elif value_type == GGUF_INT16:
            value.kind = 1
            value.int_val = Int64(bitcast[DType.int16](reader.read_u16()))
        elif value_type == GGUF_UINT32:
            value.kind = 0
            value.uint_val = UInt64(reader.read_u32())
        elif value_type == GGUF_INT32:
            value.kind = 1
            value.int_val = Int64(reader.read_i32())
        elif value_type == GGUF_FLOAT32:
            value.kind = 2
            value.float_val = Float64(reader.read_f32())
        elif value_type == GGUF_BOOL:
            value.kind = 3
            value.bool_val = reader.read_u8() != 0
        elif value_type == GGUF_STRING:
            value.kind = 4
            value.str_val = reader.read_string()
        elif value_type == GGUF_ARRAY:
            value.kind = 5
            value.arr_type = Int32(reader.read_u32())
            value.arr_len = Int(reader.read_u64())
            value.arr_offset = reader.offset
            _skip_meta_array(reader, Int(value.arr_type), value.arr_len)
        elif value_type == GGUF_UINT64:
            value.kind = 0
            value.uint_val = reader.read_u64()
        elif value_type == GGUF_INT64:
            value.kind = 1
            value.int_val = bitcast[DType.int64](reader.read_u64())
        elif value_type == GGUF_FLOAT64:
            value.kind = 2
            value.float_val = bitcast[DType.float64](reader.read_u64())
        else:
            unimplemented("gguf: unknown metadata type")
        metadata[key] = value

    for _ in range(tensor_count):
        var tensor = GGUFTensor()
        tensor.name = reader.read_string()
        tensor.n_dims = Int(reader.read_u32())
        if tensor.n_dims > GGUF_MAX_DIMS:
            unimplemented("gguf: tensor rank too large")
        for d in range(tensor.n_dims):
            tensor.dims[d] = Int(reader.read_u64())
        tensor.ggml_type = Int(reader.read_u32())
        tensor.offset = Int(reader.read_u64())
        tensor.file_idx = file_idx
        tensors.append(tensor)

    var data_offset = align_up(reader.offset, 32)
    return (data_offset, version)


def load_gguf_single(file_path: String) raises -> GGUFContext:
    """Parse a single (non-split) GGUF file into a `GGUFContext`."""
    var (data, size) = mmap_file(file_path)
    var metadata = Dict[String, GGUFMetaValue]()
    var tensors = List[GGUFTensor]()
    var (data_offset, version) = _parse_part(data, metadata, tensors, 0)
    var context = GGUFContext(data, size)
    context.version = version
    context.tensor_count = len(tensors)
    context.meta_count = len(metadata)
    context.data_offset = data_offset
    context.parts.append(GGUFFilePart(file_path, data, size, data_offset))
    for t in tensors:
        context.tensors.append(t)  # file_idx 0 (stamped in _parse_part)
    context.metadata = metadata^  # move the local into the context field
    return context^


def load_gguf_split(base: String, total: Int) raises -> GGUFContext:
    """Parse a split GGUF model (`base-NNNNN-of-TTTTT.gguf`, NNNNN in 1..total).

    The first part carries the full metadata; each part carries only its own
    tensor subset, with offsets relative to that part's data section.
    """
    var first_path = base + "-00001-of-" + _fmt5(total) + ".gguf"
    var (first_data, first_size) = mmap_file(first_path)
    var metadata = Dict[String, GGUFMetaValue]()
    var first_tensors = List[GGUFTensor]()
    var (first_data_offset, version) = _parse_part(
        first_data, metadata, first_tensors, 0
    )
    var context = GGUFContext(first_data, first_size)
    context.version = version
    context.meta_count = len(metadata)
    context.data_offset = first_data_offset
    context.parts.append(
        GGUFFilePart(first_path, first_data, first_size, first_data_offset)
    )
    for t in first_tensors:
        context.tensors.append(t)  # file_idx 0 (stamped in _parse_part)
    context.metadata = metadata^  # move the local into the context field
    for i in range(2, total + 1):
        var path = base + "-" + _fmt5(i) + "-of-" + _fmt5(total) + ".gguf"
        var (data, size) = mmap_file(path)
        var part_meta = Dict[String, GGUFMetaValue]()  # discarded (part 1 only)
        var part_tensors = List[GGUFTensor]()
        var (data_offset, _) = _parse_part(data, part_meta, part_tensors, i - 1)
        context.parts.append(GGUFFilePart(path, data, size, data_offset))
        for t in part_tensors:
            context.tensors.append(t)
    context.tensor_count = len(context.tensors)
    return context^


def load_gguf(file_path: String) raises -> GGUFContext:
    """Parse a (possibly split) GGUF model into a `GGUFContext`.

    Accepts a single `.gguf` file, a split part path
    (`<base>.gguf-NNNNN-of-NNNNN.gguf`), or a split base name whose part files
    exist on disk.
    """
    if _is_split_part_path(file_path):
        var (base, total) = _split_part_base_and_total(file_path)
        return load_gguf_split(base, total)
    if _file_exists(file_path):
        return load_gguf_single(file_path)
    var total = _find_split_total(file_path)
    if total > 0:
        return load_gguf_split(file_path, total)
    raise Error("load_gguf: file not found: " + file_path)


# -- config helpers ---------------------------------------------------------


def get_meta_uint(context: GGUFContext, key: String, default: Int) -> Int:
    var value = context.metadata.get(key, GGUFMetaValue())
    if value.kind == 0:
        return Int(value.uint_val)
    if value.kind == 1:
        return Int(value.int_val)
    if value.kind == 3:  # BOOL is a common encoding for flag keys
        return 1 if value.bool_val else 0
    return default


def get_meta_str(context: GGUFContext, key: String, default: String) -> String:
    var value = context.metadata.get(key, GGUFMetaValue())
    if value.kind == 4:
        return value.str_val
    return default


def get_meta_float(
    context: GGUFContext, key: String, default: Float64
) -> Float64:
    var value = context.metadata.get(key, GGUFMetaValue())
    if value.kind == 2:
        return value.float_val
    return default


def find_tensor(context: GGUFContext, name: String) -> Optional[GGUFTensor]:
    for tensor in context.tensors:
        if tensor.name == name:
            return tensor
    return None
