#!/usr/bin/env python3
"""
Extract Grid Tables from llama.cpp Source

Extracts IQ grid tables from llama.cpp's ggml-common.h or gguf-py/quants.py
and exports them in various formats for use in Rust/Metal code.

Usage:
    python extract_grid_tables.py --input ../shaders/ggml-common.h --output grids.json
    python extract_grid_tables.py --input ../llama.cpp-0.5.0/gguf-py/gguf/quants.py --format metal
"""

import argparse
import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def parse_c_header(content: str) -> Dict[str, List[int]]:
    """
    Parse grid tables from C header file.

    Looks for GGML_TABLE_BEGIN patterns and extracts the values.
    """
    tables = {}

    # Find all table definitions
    pattern = r'GGML_TABLE_BEGIN\s*\(\s*\w+\s*,\s*(\w+)\s*,\s*(\d+)\s*\)\s*(.*?)\s*GGML_TABLE_END\s*\(\s*\)'

    matches = re.findall(pattern, content, re.DOTALL)

    for name, size, values_str in matches:
        # Parse hex values
        values = []
        for hex_match in re.finditer(r'0x([0-9a-fA-F]+)', values_str):
            values.append(int(hex_match.group(1), 16))

        tables[name] = values
        print(f"  Found table {name} with {len(values)} entries (expected {size})")

    return tables


def parse_python_quants(content: str) -> Dict[str, Dict]:
    """
    Parse grid tables from Python quants.py file.

    Looks for grid_hex attributes in quantization classes.
    """
    tables = {}

    # Find all class definitions
    class_pattern = r'class\s+(\w+)\s*\([^)]*GGMLQuantizationType\.(\w+)[^)]*\):'

    for class_match in re.finditer(class_pattern, content):
        class_name = class_match.group(1)
        quant_type = class_match.group(2)

        # Find the class body
        start = class_match.end()
        # Find next class or end of file
        next_class = re.search(r'\nclass\s+\w+', content[start:])
        end = start + next_class.start() if next_class else len(content)
        class_body = content[start:end]

        # Extract grid_hex
        grid_hex_match = re.search(r'grid_hex\s*=\s*\(\s*(.*?)\s*\)', class_body, re.DOTALL)
        if grid_hex_match:
            hex_data = grid_hex_match.group(1)

            # Parse bytes from hex string
            # Format: b"hex_chars"
            raw_bytes = b''
            for byte_match in re.finditer(r'"([^"]*)"', hex_data):
                raw_bytes += byte_match.group(1).encode('latin-1')

            # Extract grid_shape
            shape_match = re.search(r'grid_shape\s*=\s*\((\d+),\s*(\d+)\)', class_body)
            if shape_match:
                shape = (int(shape_match.group(1)), int(shape_match.group(2)))
            else:
                shape = (len(raw_bytes) // 2, 1)

            # Extract grid_map
            map_match = re.search(r'grid_map\s*=\s*\(([^)]+)\)', class_body)
            grid_map = None
            if map_match:
                grid_map = tuple(int(x.strip(), 0) for x in map_match.group(1).split(',') if x.strip())

            tables[quant_type.lower()] = {
                'name': quant_type,
                'shape': shape,
                'grid_map': grid_map,
                'hex_data': raw_bytes.hex(),
            }
            print(f"  Found {quant_type}: shape={shape}, {len(raw_bytes)} bytes")

    return tables


def decode_packed_grid(hex_data: str, shape: Tuple[int, int], grid_map: Optional[Tuple]) -> np.ndarray:
    """
    Decode packed grid data into actual values.

    The grid_hex format stores indices that map through grid_map to actual values.
    """
    raw_bytes = bytes.fromhex(hex_data)

    n_entries, n_values = shape

    # Each byte in hex_data encodes multiple values
    # For IQ2: each byte represents 2 indices (4 bits each)
    # For IQ3: more complex packing

    if grid_map and len(grid_map) > 0:
        # Values are encoded as indices into grid_map
        n_bits_per_value = int(np.ceil(np.log2(len(grid_map))))
        values_per_byte = 8 // n_bits_per_value

        grid = np.zeros((n_entries, n_values), dtype=np.int32)

        byte_idx = 0
        bit_idx = 0

        for entry_idx in range(n_entries):
            for val_idx in range(n_values):
                # Extract n_bits_per_value bits
                byte_val = raw_bytes[byte_idx]
                bits_left_in_byte = 8 - bit_idx

                if n_bits_per_value <= bits_left_in_byte:
                    # Value fits in current byte
                    value_idx = (byte_val >> bit_idx) & ((1 << n_bits_per_value) - 1)
                    bit_idx += n_bits_per_value
                    if bit_idx >= 8:
                        bit_idx = 0
                        byte_idx += 1
                else:
                    # Value spans two bytes (not typical for IQ formats)
                    value_idx = byte_val >> bit_idx
                    byte_idx += 1
                    remaining_bits = n_bits_per_value - bits_left_in_byte
                    value_idx |= (raw_bytes[byte_idx] & ((1 << remaining_bits) - 1)) << bits_left_in_byte
                    bit_idx = remaining_bits
                    byte_idx += 1 if bit_idx >= 8 else 0

                grid[entry_idx, val_idx] = grid_map[value_idx] if value_idx < len(grid_map) else 0

        return grid
    else:
        # Direct hex encoding - decode from hex string
        grid = np.zeros((n_entries, n_values), dtype=np.int32)

        # Parse hex string (each pair of hex digits = 1 byte)
        hex_str = hex_data
        idx = 0
        for entry_idx in range(n_entries):
            for val_idx in range(n_values):
                if idx + 2 <= len(hex_str):
                    byte_val = int(hex_str[idx:idx+2], 16)
                    grid[entry_idx, val_idx] = byte_val
                    idx += 2

        return grid


def export_json(tables: Dict, output_path: str):
    """Export tables as JSON."""
    with open(output_path, 'w') as f:
        json.dump(tables, f, indent=2)
    print(f"Saved to {output_path}")


def export_metal(tables: Dict, output_path: str):
    """Export tables as Metal shader header."""
    with open(output_path, 'w') as f:
        f.write("// Auto-generated IQ grid tables for Metal\n")
        f.write("// Extracted from llama.cpp source\n\n")

        for name, data in tables.items():
            if isinstance(data, dict):
                # Python format
                shape = data['shape']
                hex_data = data['hex_data']
                raw_bytes = bytes.fromhex(hex_data)

                total_bits = shape[1] * 8  # Approximate
                if shape[1] <= 4:
                    dtype = "uint32_t"
                else:
                    dtype = "uint64_t"

                f.write(f"constexpr constant static {dtype} {name}_grid[{shape[0]}] = {{\n")

                # Pack bytes into integers
                values = []
                if dtype == "uint64_t":
                    for i in range(0, len(raw_bytes), 8):
                        val = int.from_bytes(raw_bytes[i:i+8], 'little')
                        values.append(val)
                elif dtype == "uint32_t":
                    for i in range(0, len(raw_bytes), 4):
                        val = int.from_bytes(raw_bytes[i:i+4], 'little')
                        values.append(val)

                for i, v in enumerate(values):
                    if i % 4 == 0:
                        f.write("    ")
                    hex_width = 16 if dtype == "uint64_t" else 8
                    f.write(f"0x{v:0{hex_width}x}, ")
                    if (i + 1) % 4 == 0:
                        f.write("\n")

                if len(values) % 4 != 0:
                    f.write("\n")
                f.write("};\n\n")
            else:
                # C header format (already has values)
                f.write(f"// {name}\n")
                f.write(f"static const uint32_t {name}[] = {{\n")
                for i, v in enumerate(data):
                    if i % 8 == 0:
                        f.write("    ")
                    f.write(f"0x{v:x}, ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")
                if len(data) % 8 != 0:
                    f.write("\n")
                f.write("};\n\n")

    print(f"Saved to {output_path}")


def export_rust(tables: Dict, output_path: str):
    """Export tables as Rust source."""
    with open(output_path, 'w') as f:
        f.write("// Auto-generated IQ grid tables for Rust\n")
        f.write("// Extracted from llama.cpp source\n\n")

        for name, data in tables.items():
            if isinstance(data, dict):
                shape = data['shape']
                hex_data = data['hex_data']
                raw_bytes = bytes.fromhex(hex_data)

                f.write(f"pub const {name.upper()}_GRID_SIZE: usize = {shape[0]};\n")
                f.write(f"pub const {name.upper()}_VALUES_PER_ENTRY: usize = {shape[1]};\n")

                # Export as raw bytes
                f.write(f"pub const {name.upper()}_GRID: [u8; {len(raw_bytes)}] = [\n")
                for i, b in enumerate(raw_bytes):
                    if i % 16 == 0:
                        f.write("    ")
                    f.write(f"0x{b:02x}, ")
                    if (i + 1) % 16 == 0:
                        f.write("\n")
                if len(raw_bytes) % 16 != 0:
                    f.write("\n")
                f.write("];\n\n")
            else:
                f.write(f"pub const {name.upper()}: [u32; {len(data)}] = [\n")
                for i, v in enumerate(data):
                    if i % 8 == 0:
                        f.write("    ")
                    f.write(f"0x{v:x}, ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")
                if len(data) % 8 != 0:
                    f.write("\n")
                f.write("];\n\n")

    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract IQ grid tables from llama.cpp')
    parser.add_argument('--input', '-i', required=True,
                        help='Input file (ggml-common.h or quants.py)')
    parser.add_argument('--output', '-o', default='grids.json',
                        help='Output file path')
    parser.add_argument('--format', '-f', choices=['json', 'metal', 'rust'],
                        default='json', help='Output format')

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Parsing {input_path}...")

    content = input_path.read_text()

    if input_path.suffix == '.h':
        tables = parse_c_header(content)
    elif input_path.suffix == '.py':
        tables = parse_python_quants(content)
    else:
        print(f"Error: Unknown file type: {input_path.suffix}")
        return 1

    if not tables:
        print("Error: No grid tables found")
        return 1

    print(f"\nFound {len(tables)} grid tables")

    if args.format == 'json':
        export_json(tables, args.output)
    elif args.format == 'metal':
        export_metal(tables, args.output)
    elif args.format == 'rust':
        export_rust(tables, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
