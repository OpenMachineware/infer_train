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

from .memory import mmap_file
from .utils import align_up, unimplemented
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.collections import Span
from std.utils.static_tuple import StaticTuple
from std.memory.unsafe import bitcast

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


struct GGUFMetaValue(Copyable, Movable, ImplicitlyCopyable):
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


struct GGUFTensor(Copyable, Movable, ImplicitlyCopyable):
    var name: String
    var n_dims: Int
    var dims: StaticTuple[Int, GGUF_MAX_DIMS]
    var ggml_type: Int
    var offset: Int  # byte offset into the tensor-data section

    def __init__(out self):
        self.name = String("")
        self.n_dims = 0
        self.dims = StaticTuple[Int, GGUF_MAX_DIMS](fill=0)
        self.ggml_type = 0
        self.offset = 0


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
    var data_offset: Int  # byte offset where tensor payloads begin

    def __init__(
        out self, data: Pointer[UInt8, MutUntrackedOrigin], size: Int
    ):
        self.data = data
        self.size = size
        self.version = 0
        self.tensor_count = 0
        self.meta_count = 0
        self.metadata = Dict[String, GGUFMetaValue]()
        self.tensors = List[GGUFTensor]()
        self.data_offset = 0


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


def load_gguf(file_path: String) raises -> GGUFContext:
    """Parse a GGUF file into a `GGUFContext` (file is memory-mapped)."""
    var (data, size) = mmap_file(file_path)
    var reader = Reader(data)

    # Header.
    if (
        reader.read_u8() != UInt8(71)  # 'G'
        or reader.read_u8() != UInt8(71)  # 'G'
        or reader.read_u8() != UInt8(85)  # 'U'
        or reader.read_u8() != UInt8(70)  # 'F'
    ):
        unimplemented("load_gguf: bad magic")

    var context = GGUFContext(data, size)
    context.version = Int(reader.read_u32())
    context.tensor_count = Int(reader.read_u64())
    context.meta_count = Int(reader.read_u64())

    # Metadata key-value pairs.
    for _ in range(context.meta_count):
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
            unimplemented("load_gguf: unknown metadata type")
        context.metadata[key] = value

    # Tensor info table.
    for _ in range(context.tensor_count):
        var tensor = GGUFTensor()
        tensor.name = reader.read_string()
        tensor.n_dims = Int(reader.read_u32())
        if tensor.n_dims > GGUF_MAX_DIMS:
            unimplemented("load_gguf: tensor rank too large")
        for d in range(tensor.n_dims):
            tensor.dims[d] = Int(reader.read_u64())
        tensor.ggml_type = Int(reader.read_u32())
        tensor.offset = Int(reader.read_u64())
        context.tensors.append(tensor)

    # Tensor payloads are 32-byte aligned.
    context.data_offset = align_up(reader.offset, 32)
    return context^


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
