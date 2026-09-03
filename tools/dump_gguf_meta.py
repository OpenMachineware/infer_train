#!/usr/bin/env python3
"""Dump GGUF metadata (and optionally tensor names) for debugging arch mapping.

Usage: python3 dump_gguf_meta.py <file.gguf> [--tensors] [--arch KEY]
"""
import struct
import sys

MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION_3 = 3


def read_u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_i64(f):
    return struct.unpack("<q", f.read(8))[0]


def read_u32(f):
    return struct.unpack("<I", f.read(4))[0]


def read_i32(f):
    return struct.unpack("<i", f.read(4))[0]


def read_f32(f):
    return struct.unpack("<f", f.read(4))[0]


def read_f64(f):
    return struct.unpack("<d", f.read(8))[0]


def read_str(f):
    n = read_u64(f)
    return f.read(n).decode("utf-8", errors="replace")


# Standard GGUF value types (matches src/core/gguf_loader.mojo).
def read_value(f, val_type):
    if val_type == 0:  # UINT8
        return f.read(1)[0]
    if val_type == 1:  # INT8
        return struct.unpack("<b", f.read(1))[0]
    if val_type == 2:  # UINT16
        return struct.unpack("<H", f.read(2))[0]
    if val_type == 3:  # INT16
        return struct.unpack("<h", f.read(2))[0]
    if val_type == 4:  # UINT32
        return read_u32(f)
    if val_type == 5:  # INT32
        return read_i32(f)
    if val_type == 6:  # FLOAT32
        return read_f32(f)
    if val_type == 7:  # BOOL
        return bool(f.read(1)[0])
    if val_type == 8:  # STRING
        return read_str(f)
    if val_type == 9:  # ARRAY
        elem_type = read_u32(f)
        arr_len = read_u64(f)
        return [read_value(f, elem_type) for _ in range(arr_len)]
    if val_type == 10:  # UINT64
        return read_u64(f)
    if val_type == 11:  # INT64
        return read_i64(f)
    if val_type == 12:  # FLOAT64
        return read_f64(f)
    raise ValueError("unknown value type %d" % val_type)


def main():
    path = sys.argv[1]
    show_tensors = "--tensors" in sys.argv
    with open(path, "rb") as f:
        magic = read_u32(f)
        assert magic == MAGIC, "not a GGUF file"
        version = read_u32(f)
        n_tensors = read_u64(f)
        n_kv = read_u64(f)
        print("== %s ==" % path)
        print("version=%d tensors=%d kv=%d" % (version, n_tensors, n_kv))
        for _ in range(n_kv):
            key = read_str(f)
            val_type = read_u32(f)
            val = read_value(f, val_type)
            if isinstance(val, list) and len(val) > 16:
                val = val[:16] + ["...(%d)" % len(val)]
            print("%-55s = %r" % (key, val))
        if show_tensors:
            print("-- tensors (%d) --" % n_tensors)
            for i in range(n_tensors):
                name = read_str(f)
                n_dims = read_u32(f)
                dims = [read_u64(f) for _ in range(n_dims)]
                ggml_type = read_u32(f)
                offset = read_u64(f)
                print("%-70s dims=%s type=%d" % (name, dims, ggml_type))


if __name__ == "__main__":
    main()
