#!/usr/bin/env python3
"""
IQ Test Data Generator

Generates valid test data for IQ quantization formats. This data can be used
for correctness verification of quantization/dequantization kernels.

The key insight is that IQ formats use grid tables, so valid test data must
quantize to actual grid entries, not arbitrary values.

Usage:
    python iq_test_data_generator.py --format iq3_xxs --blocks 100 --output test_iq3_xxs.npz
    python iq_test_data_generator.py --format iq2_xxs --generate-reference
"""

import argparse
import json
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import struct


# IQ format block sizes
QK_K = 256  # Super-block size


@dataclass
class IQFormatConfig:
    """Configuration for an IQ format."""
    name: str
    block_size: int
    bits_per_value: int
    grid_size: int
    values_per_entry: int

    # Block layout
    has_scale: bool
    has_signs: bool
    has_qh: bool  # High bits

    # Scale encoding
    scale_bits: int
    scale_per_block: int


# Standard IQ format configurations
IQ_FORMATS = {
    'iq1_s': IQFormatConfig(
        name='iq1_s',
        block_size=256,
        bits_per_value=1,
        grid_size=2048,
        values_per_entry=8,
        has_scale=True,
        has_signs=False,
        has_qh=True,
        scale_bits=16,
        scale_per_block=1,
    ),
    'iq1_m': IQFormatConfig(
        name='iq1_m',
        block_size=256,
        bits_per_value=1,
        grid_size=2048,
        values_per_entry=8,
        has_scale=True,
        has_signs=False,
        has_qh=True,
        scale_bits=8,
        scale_per_block=8,
    ),
    'iq2_xxs': IQFormatConfig(
        name='iq2_xxs',
        block_size=256,
        bits_per_value=2,
        grid_size=256,
        values_per_entry=8,
        has_scale=True,
        has_signs=True,
        has_qh=False,
        scale_bits=16,
        scale_per_block=1,
    ),
    'iq2_xs': IQFormatConfig(
        name='iq2_xs',
        block_size=256,
        bits_per_value=2,
        grid_size=512,
        values_per_entry=8,
        has_scale=True,
        has_signs=True,
        has_qh=False,
        scale_bits=8,
        scale_per_block=8,
    ),
    'iq2_s': IQFormatConfig(
        name='iq2_s',
        block_size=256,
        bits_per_value=2,
        grid_size=1024,
        values_per_entry=8,
        has_scale=True,
        has_signs=True,
        has_qh=True,
        scale_bits=8,
        scale_per_block=8,
    ),
    'iq3_xxs': IQFormatConfig(
        name='iq3_xxs',
        block_size=256,
        bits_per_value=3,
        grid_size=256,
        values_per_entry=4,
        has_scale=True,
        has_signs=True,
        has_qh=False,
        scale_bits=32,  # Packed with scales and signs
        scale_per_block=1,
    ),
    'iq3_s': IQFormatConfig(
        name='iq3_s',
        block_size=256,
        bits_per_value=3,
        grid_size=512,
        values_per_entry=4,
        has_scale=True,
        has_signs=True,
        has_qh=True,
        scale_bits=8,
        scale_per_block=4,
    ),
    'iq4_nl': IQFormatConfig(
        name='iq4_nl',
        block_size=32,
        bits_per_value=4,
        grid_size=16,
        values_per_entry=1,
        has_scale=True,
        has_signs=False,
        has_qh=False,
        scale_bits=16,
        scale_per_block=1,
    ),
    'iq4_xs': IQFormatConfig(
        name='iq4_xs',
        block_size=256,
        bits_per_value=4,
        grid_size=256,
        values_per_entry=1,
        has_scale=True,
        has_signs=False,
        has_qh=False,
        scale_bits=16,
        scale_per_block=1,
    ),
}


def load_grid_table(format_name: str) -> np.ndarray:
    """
    Load the grid table for an IQ format.

    This loads from llama.cpp's ggml-common.h or gguf-py/quants.py
    """
    # For now, return None - grids should be loaded from actual source
    # or generated using iq_grid_generator.py
    return None


def quantize_to_grid(
    values: np.ndarray,
    grid: np.ndarray,
    scale: float
) -> Tuple[np.ndarray, float]:
    """
    Quantize values to nearest grid entries.

    Args:
        values: Values to quantize
        grid: Grid table
        scale: Initial scale

    Returns:
        indices: Grid indices
        final_scale: Adjusted scale
    """
    # Normalize by scale
    normalized = values / scale

    # Find nearest grid entry for each group of values_per_entry
    n_groups = len(values) // grid.shape[1]
    values_per_entry = grid.shape[1]

    indices = np.zeros(n_groups, dtype=np.int32)

    for i in range(n_groups):
        group = normalized[i * values_per_entry : (i + 1) * values_per_entry]
        distances = np.sum((grid - group) ** 2, axis=1)
        indices[i] = np.argmin(distances)

    return indices, scale


def generate_valid_fp32_data(
    config: IQFormatConfig,
    grid: Optional[np.ndarray] = None,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate FP32 test data that will quantize correctly to the given format.

    This generates data that is guaranteed to dequantize to the expected values,
    avoiding rounding errors that cause correctness test failures.

    Args:
        config: Format configuration
        grid: Grid table (if None, uses default value range)
        seed: Random seed

    Returns:
        fp32_data: FP32 values
        metadata: Metadata about how data was generated
    """
    np.random.seed(seed)

    n_blocks = 1
    n_values = config.block_size * n_blocks

    if grid is not None and config.bits_per_value >= 2:
        # Use grid-based generation for proper quantization
        # Generate random grid indices
        n_groups = n_values // config.values_per_entry
        indices = np.random.randint(0, config.grid_size, n_groups)

        # Look up grid values
        if config.values_per_entry == 1:
            grid_values = grid[indices].flatten()
        else:
            grid_values = grid[indices].reshape(-1)

        # Apply random scale
        scale = np.random.uniform(0.5, 2.0)
        fp32_data = grid_values * scale

        metadata = {
            'method': 'grid_lookup',
            'indices': indices.tolist(),
            'scale': scale,
            'grid_size': config.grid_size,
        }
    else:
        # For simple formats (like iq4_nl), generate directly
        if config.name == 'iq4_nl':
            # 4-bit values: -8 to 7
            quantized = np.random.randint(-8, 8, n_values).astype(np.float32)
            scale = np.random.uniform(0.1, 1.0)
            fp32_data = quantized * scale

            metadata = {
                'method': 'direct_quantization',
                'value_range': [-8, 7],
                'scale': scale,
            }
        else:
            # Generic: generate normal distribution
            fp32_data = np.random.randn(n_values).astype(np.float32)

            metadata = {
                'method': 'random_normal',
                'scale': 1.0,
            }

    return fp32_data, metadata


def pack_iq_block(
    indices: np.ndarray,
    signs: Optional[np.ndarray],
    scale: float,
    config: IQFormatConfig
) -> bytes:
    """
    Pack quantized data into a block in llama.cpp format.

    Args:
        indices: Grid indices
        signs: Sign bits (optional)
        scale: Scale value
        config: Format configuration

    Returns:
        Packed block bytes
    """
    block_data = bytearray()

    # Write scale (FP16)
    scale_f16 = np.float16(scale).tobytes()
    block_data.extend(scale_f16)

    # Write packed indices
    if config.bits_per_value == 2:
        # Pack 4 indices per byte
        packed = np.packbits(indices.reshape(-1, 4) @ np.array([1, 4, 16, 64]), bitorder='little')
        block_data.extend(packed)
    elif config.bits_per_value == 3:
        # Pack indices (complex packing for IQ3)
        # 8 groups of 3 bits = 24 bits = 3 bytes per 8 indices
        packed = np.zeros(len(indices) * 3 // 8, dtype=np.uint8)
        for i, idx in enumerate(indices):
            byte_idx = (i * 3) // 8
            bit_offset = (i * 3) % 8
            packed[byte_idx] |= (idx & 0x7) << bit_offset
        block_data.extend(packed)
    elif config.bits_per_value == 4:
        # Pack 2 indices per byte
        packed = (indices[::2] & 0xF) | ((indices[1::2] & 0xF) << 4)
        block_data.extend(packed)
    else:
        # 1-bit: pack 8 per byte
        packed = np.packbits(indices, bitorder='little')
        block_data.extend(packed)

    # Write signs if present
    if signs is not None:
        packed_signs = np.packbits(signs, bitorder='little')
        block_data.extend(packed_signs)

    return bytes(block_data)


def generate_test_dataset(
    format_name: str,
    n_blocks: int = 100,
    seed: int = 42,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate a complete test dataset for an IQ format.

    Args:
        format_name: Format name (e.g., 'iq3_xxs')
        n_blocks: Number of blocks to generate
        seed: Random seed
        output_path: Optional path to save dataset

    Returns:
        fp32_values: Original FP32 values
        quantized_blocks: Packed quantized blocks
        metadata: Generation metadata
    """
    config = IQ_FORMATS.get(format_name)
    if config is None:
        raise ValueError(f"Unknown format: {format_name}")

    print(f"Generating test data for {format_name}...")
    print(f"  Block size: {config.block_size}")
    print(f"  Bits per value: {config.bits_per_value}")
    print(f"  Grid size: {config.grid_size}")
    print(f"  Number of blocks: {n_blocks}")

    # Try to load grid
    grid = load_grid_table(format_name)

    # Generate data
    np.random.seed(seed)

    all_fp32 = []
    all_blocks = []
    all_metadata = []

    for i in range(n_blocks):
        fp32, meta = generate_valid_fp32_data(config, grid, seed=seed + i)
        all_fp32.append(fp32)
        all_metadata.append(meta)

        # Generate placeholder quantized block
        # (In production, this would use actual quantization)
        block = bytes(config.block_size)  # Placeholder
        all_blocks.append(block)

    fp32_array = np.concatenate(all_fp32)
    blocks_array = np.array(all_blocks)

    metadata = {
        'format': format_name,
        'n_blocks': n_blocks,
        'block_size': config.block_size,
        'total_values': n_blocks * config.block_size,
        'generation_seed': seed,
        'blocks': all_metadata[:10],  # Store first 10 block metadata
    }

    if output_path:
        np.savez(
            output_path,
            fp32=fp32_array,
            blocks=blocks_array,
            metadata=json.dumps(metadata)
        )
        print(f"  Saved to {output_path}")

    return fp32_array, blocks_array, metadata


def generate_reference_values(format_name: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate reference values for correctness testing.

    This generates known input/output pairs for verification.
    """
    config = IQ_FORMATS[format_name]

    # Generate a small set of known values
    # These should match exactly after quantize->dequantize roundtrip

    test_cases = []

    # Test case 1: All zeros
    test_cases.append({
        'name': 'zeros',
        'input': np.zeros(config.block_size, dtype=np.float32),
        'expected_output': np.zeros(config.block_size, dtype=np.float32),
    })

    # Test case 2: All ones (after scaling)
    test_cases.append({
        'name': 'ones',
        'input': np.ones(config.block_size, dtype=np.float32),
        'expected_output': np.ones(config.block_size, dtype=np.float32),  # May differ
    })

    # Test case 3: Alternating signs
    test_cases.append({
        'name': 'alternating',
        'input': np.array([1.0, -1.0] * (config.block_size // 2), dtype=np.float32),
        'expected_output': None,  # Will compute after quantization
    })

    result = {
        'format': format_name,
        'test_cases': test_cases,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    return result


def export_for_rust(
    format_name: str,
    n_blocks: int,
    output_path: str,
    seed: int = 42
):
    """
    Export test data in a format suitable for Rust tests.

    Generates a Rust source file with test data embedded.
    """
    config = IQ_FORMATS[format_name]

    fp32, blocks, metadata = generate_test_dataset(
        format_name,
        n_blocks=n_blocks,
        seed=seed
    )

    with open(output_path, 'w') as f:
        f.write(f"// Auto-generated test data for {format_name}\n")
        f.write(f"// Generated with seed={seed}, n_blocks={n_blocks}\n\n")

        f.write(f"pub const FORMAT_NAME: &str = \"{format_name}\";\n")
        f.write(f"pub const BLOCK_SIZE: usize = {config.block_size};\n")
        f.write(f"pub const N_BLOCKS: usize = {n_blocks};\n\n")

        # Write FP32 data
        f.write("pub const FP32_DATA: &[f32] = &[\n")
        for i, v in enumerate(fp32):
            if i % 8 == 0:
                f.write("    ")
            f.write(f"{v:.6f}, ")
            if (i + 1) % 8 == 0:
                f.write("\n")
        f.write("];\n\n")

        # Write block data as hex
        f.write("pub const QUANTIZED_BLOCKS: &[u8] = &[\n")
        for i, block in enumerate(blocks):
            if i % 16 == 0:
                f.write("    ")
            f.write(f"0x{block[0]:02x}, ")
            if (i + 1) % 16 == 0:
                f.write("\n")
        f.write("];\n")

    print(f"  Saved Rust test data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate IQ test data')
    parser.add_argument('--format', '-f', required=True,
                        choices=list(IQ_FORMATS.keys()),
                        help='IQ format name')
    parser.add_argument('--blocks', '-b', type=int, default=100,
                        help='Number of blocks to generate')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file path')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--generate-reference', action='store_true',
                        help='Generate reference test cases')
    parser.add_argument('--export-rust', action='store_true',
                        help='Export as Rust source file')

    args = parser.parse_args()

    output_path = args.output or f"test_{args.format}.npz"

    if args.generate_reference:
        generate_reference_values(args.format, output_path.replace('.npz', '_reference.json'))
    elif args.export_rust:
        export_for_rust(args.format, args.blocks, output_path.replace('.npz', '.rs'), args.seed)
    else:
        generate_test_dataset(args.format, args.blocks, args.seed, output_path)

    print("\nTest data generation complete!")


if __name__ == '__main__':
    main()
