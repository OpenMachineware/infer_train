#!/usr/bin/env python3
"""
Direct call to llama.cpp IQ2_XXS quantization with dump output
"""

import ctypes
import struct
import array

# Load the library
lib_path = "/Users/jia/Desktop/infer_train/llama.cpp-0.5.0/build/bin/libggml.dylib"
lib = ctypes.CDLL(lib_path)

# Define structures
class BlockIQ2XXS(ctypes.Structure):
    _fields_ = [
        ("d", ctypes.c_uint16),
        ("qs", ctypes.c_uint16 * 8),
    ]

# Define function prototypes
lib.ggml_quantize_init.restype = None
lib.ggml_quantize_init.argtypes = [ctypes.c_int]

lib.quantize_iq2_xxs.restype = ctypes.c_size_t
lib.quantize_iq2_xxs.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # src
    ctypes.c_void_p,                 # dst
    ctypes.c_int64,                  # nrows
    ctypes.c_int64,                  # n_per_row
    ctypes.POINTER(ctypes.c_float),  # imatrix
]

# Constants
GGML_TYPE_IQ2_XXS = 21  # From ggml.h
QK_K = 256

# Initialize
print("Initializing llama.cpp quantization tables...")
lib.ggml_quantize_init(GGML_TYPE_IQ2_XXS)

# Create test data (same as Rust version)
test_values = [
    0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812, 0.934245, 0.814190,
    -0.321567, 0.456789, -0.234567, 0.876543, -0.543210, 0.123456, -0.789012, 0.345678,
    0.567890, -0.234567, 0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
    0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678, 0.567890, -0.234567,
] + [0.0] * 224

test_data = array.array('f', test_values)
quant_weights = array.array('f', [1.0] * QK_K)

# Prepare output
output = BlockIQ2XXS()

print("\nCalling llama.cpp quantize_iq2_xxs...")
print("=" * 60)

# Call quantization (this will print dump output from C code)
lib.quantize_iq2_xxs(
    (ctypes.c_float * len(test_data))(*test_data),
    ctypes.byref(output),
    1,  # nrows
    QK_K,  # n_per_row
    (ctypes.c_float * QK_K)(*quant_weights),
)

print("=" * 60)
print("\nllama.cpp output:")
print(f"  d = 0x{output.d:04x}")
print(f"  qs = ", end="")
for i in range(8):
    print(f"0x{output.qs[i]:04x} ", end="")
print()