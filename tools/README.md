# IQ Quantization Tools

This directory contains reusable tools for IQ quantization development.

## Tools

### 1. iq_grid_generator.py

Generates grid lookup tables for IQ quantization formats using k-means clustering.

The grid tables are pre-computed optimal values used during quantization and dequantization.
They are NOT derived from fixed mathematical formulas, but from optimization algorithms.

**Usage:**
```bash
# Generate grid for IQ3_XXS
python iq_grid_generator.py --format iq3_xxs --output grid_iq3_xxs.json

# Generate with more samples for better quality
python iq_grid_generator.py --format iq2_xxs --samples 500000 --iterations 200

# Export as C header for Metal shaders
python iq_grid_generator.py --format iq3_s --format-output c_header
```

**Output formats:**
- `json`: JSON file with grid values and metadata
- `c_header`: C header with GGML_TABLE_BEGIN/END macros (for Metal/CUDA)
- `both`: Both formats

### 2. iq_test_data_generator.py

Generates valid test data for IQ format correctness testing.

The key challenge with IQ formats is that test data must quantize correctly to actual
grid entries. This tool ensures generated data will produce exact matches after
quantize->dequantize roundtrip.

**Usage:**
```bash
# Generate test data for IQ3_XXS
python iq_test_data_generator.py --format iq3_xxs --blocks 100 --output test_iq3_xxs.npz

# Generate reference test cases
python iq_test_data_generator.py --format iq2_xxs --generate-reference

# Export as Rust source file
python iq_test_data_generator.py --format iq4_nl --blocks 50 --export-rust
```

**Output formats:**
- `.npz`: NumPy compressed file with fp32 values and quantized blocks
- `_reference.json`: Simple test cases for correctness verification
- `.rs`: Rust source file with embedded test data

### 3. extract_grid_tables.py

Extracts existing grid tables from llama.cpp source code.

Instead of generating new grids, this extracts the canonical tables used by llama.cpp,
ensuring compatibility with existing models.

**Usage:**
```bash
# Extract from C header
python extract_grid_tables.py --input ../shaders/ggml-common.h --output grids.json

# Extract from Python quants.py and export for Metal
python extract_grid_tables.py --input ../llama.cpp-0.5.0/gguf-py/gguf/quants.py --format metal
```

**Output formats:**
- `json`: JSON with decoded grid values
- `metal`: Metal shader header with grid constants
- `rust`: Rust source with grid constants

## IQ Format Reference

| Format | Grid Size | Values/Entry | Bits/Value | Block Size |
|--------|-----------|--------------|------------|------------|
| IQ1_S  | 2048      | 8            | 1          | 256        |
| IQ1_M  | 2048      | 8            | 1          | 256        |
| IQ2_XXS| 256       | 8            | 2          | 256        |
| IQ2_XS | 512       | 8            | 2          | 256        |
| IQ2_S  | 1024      | 8            | 2          | 256        |
| IQ3_XXS| 256       | 4            | 3          | 256        |
| IQ3_S  | 512       | 4            | 3          | 256        |
| IQ4_NL | 16        | 1            | 4          | 32         |
| IQ4_XS | 256       | 1            | 4          | 256        |

## Grid Table Structure

Grid tables map indices to actual quantized values:

```python
# IQ3_XXS example
grid_shape = (256, 4)  # 256 entries, 4 values each
grid_map = (0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e)

# Each grid entry encodes 4 values using indices 0-7
# Index i maps to value grid_map[i]
```

## Workflow

1. **For development**: Use `extract_grid_tables.py` to get llama.cpp's canonical grids
2. **For testing**: Use `iq_test_data_generator.py` to generate valid test data
3. **For research**: Use `iq_grid_generator.py` to experiment with new grids

## Notes

- Grid generation is computationally expensive (k-means with 100K+ samples)
- llama.cpp's grids were generated once and are now fixed
- Use canonical grids for compatibility with existing quantized models
- New grids would require re-quantization of model weights
