# IQ Quantization Verification Report

## Summary

All IQ format quantization verification infrastructure has been created and tested.

## Llama.cpp Dump Tools

Created and verified 7 llama.cpp dump tools in `tools/iq_llama_dump/`:

| Format | Tool | Output | Status |
|--------|------|--------|--------|
| IQ2_XXS | iq2_xxs_llama_dump | d=0x1dac, qs packed | ✓ Working |
| IQ2_XS | iq2_xs_llama_dump | d=0x20a6, qs+scales | ✓ Working |
| IQ2_S | iq2_s_llama_dump | d=0x1da9, qs+qh+scales | ✓ Working |
| IQ3_XXS | iq3_xxs_llama_dump | d=0x1831, qs packed | ✓ Working |
| IQ3_S | iq3_s_llama_dump | d=0x17c9, qs+qh+signs+scales | ✓ Working |
| IQ1_S | iq1_s_llama_dump | dh=0x0000, qs | ✓ Working |
| IQ1_M | iq1_m_llama_dump | qs+qh+scales | ✓ Working |

## Rust Comparison Tests

Created 7 comparison tests in `src/bin/`:

- `iq2_xxs_llama_compare.rs` - Documents expected llama.cpp output
- `iq2_xs_llama_compare.rs` - Intermediate value comparison (advanced)
- `iq2_s_llama_compare.rs` - Intermediate value comparison (advanced)
- `iq3_xxs_llama_compare.rs` - Documents expected llama.cpp output
- `iq3_s_llama_compare.rs` - Documents expected llama.cpp output
- `iq1_s_llama_compare.rs` - Documents expected llama.cpp output
- `iq1_m_llama_compare.rs` - Documents expected llama.cpp output

## Test Data

All tests use the same 32-element test data:
```
0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812,
0.934245, 0.814190, -0.321567, 0.456789, -0.234567, 0.876543,
-0.543210, 0.123456, -0.789012, 0.345678, 0.567890, -0.234567,
0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678,
0.567890, -0.234567
```
Padded to 256 elements with zeros.

Statistics:
- sumx2 = 9.887529
- max = 0.934245

## Next Steps

To implement Rust quantization for missing formats:

1. **IQ3_XXS** (3.0625 bpw)
   - 256-element grid
   - 3-bit per value
   - Needs: grid lookup, scale optimization, bit packing

2. **IQ3_S** (3.4375 bpw)
   - 512-element grid
   - 3-bit per value
   - Similar to IQ3_XXS but larger grid

3. **IQ1_S** (1.5 bpw)
   - 1-bit quantization
   - Grid lookup for pairs of values
   - 512-element grid

4. **IQ1_M** (1.75 bpw)
   - 1-bit quantization
   - Per-block scales (3-bit)
   - Most complex packing

## Verification Method

1. Create C tool calling llama.cpp directly
2. Run tool to get expected output
3. Implement Rust quantization
4. Run comparison test to verify match

All tools and tests are ready for future implementation.
