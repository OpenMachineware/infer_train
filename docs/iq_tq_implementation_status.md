# IQ/TQ Series Implementation Status

## Completed

### Q4_K Optimization ✅
- **File**: `src/quant/vec_dot/arm.rs`
- **Performance**: 84.8 GFLOPS (exceeds llama.cpp by 3-10%)
- **Key optimization**: `vpaddq_s16` for bsums, correct scale decode
- **Commits**: b24aee5, 7f817c7, aaae792

### IQ4_NL ✅
- **Block size**: 32 (pairs with Q8_0)
- **Lookup table**: `KVALUES_IQ4NL` (16 items)
- **Instruction**: `vqtbl1q_s8` (TBL)
- **Commit**: 7f817c7

### IQ4_XS ✅
- **Block size**: 256 (pairs with Q8_K)
- **Scale decode**: `scales_l` + `scales_h`, offset -32
- **Commit**: aaae792

### Lookup Tables Extracted ✅
- **File**: `src/quant/vec_dot/tables.rs`
- `KEVEN_SIGNS_Q2XS` (1024 × i8) - Sign lookup for IQ2/IQ3 series
- `IQ3XXS_GRID` (256 × u32) - Grid for IQ3_XXS
- `IQ3S_GRID` (512 × u32) - Grid for IQ3_S

## Remaining Work

### Priority Order
1. **IQ1_S/IQ1_M** - Most important (lowest bpw: 1.5625)
2. **IQ2_XXS/IQ2_XS/IQ2_S** - Medium bpw
3. **IQ3_XXS/IQ3_S** - Grids extracted, need kernel implementation
4. **TQ1_0/TQ2_0** - New formats (no lookup tables)

### Lookup Tables Still Needed

| Format | Table Name | Size | Location in llama.cpp |
|--------|-----------|------|----------------------|
| IQ1_S | `iq1s_grid` | 2048 × u64 | `ggml-common.h:1135` |
| IQ1_M | Uses iq1s_grid + delta | - | Same as IQ1_S |
| IQ2_XXS | `iq2xxs_grid` | 256 × u64 | `ggml-common.h:560` |
| IQ2_XS | `iq2xs_grid` | 512 × u64 | `ggml-common.h:627` |
| IQ2_S | `iq2s_grid` | 1024 × u64 | `ggml-common.h:758` |

### Implementation Reference

#### IQ1_S Pattern (from llama.cpp:4036-4100)
```rust
// Grid index calculation
let grid_idx = qs[ib] | ((qh[ib] << 8) & 0x700) | ((qh[ib] << 5) & 0x700) | ...;

// Scale: 2*((qh >> 12) & 7) + 1
let scale = 2 * ((qh >> 12) & 7) + 1;

// Delta correction with bsums
const IQ1S_DELTA: f32 = 0.125;
sum += delta * bsum * scale * sign;
```

#### IQ2_XXS Pattern (from llama.cpp:3631-3690)
```rust
// Grid index from qs (8-bit)
let grid_idx = aux8[0]; // 256 entries

// Sign from aux32 high bits (7-bit index into keven_signs)
let sign_idx = (aux32[1] >> 0) & 127;
let signs = KEVEN_SIGNS_Q2XS[sign_idx];

// Scale from high bits
let scale = 0.5 + (aux32[1] >> 28);

// Final multiplier: 0.25
```

#### IQ2_XS Pattern (from llama.cpp:3693-3765)
```rust
// Grid index: q2[i] & 511 (9-bit)
let grid_idx = q2[i] & 511;

// Sign index: q2[i] >> 9 (7-bit)
let sign_idx = q2[i] >> 9;

// Scale: decode from scales array
let scale = 1 + 2*(scales[ib] & 0xf);

// Final multiplier: 0.125
```

#### IQ2_S Pattern (from llama.cpp:3767-3862)
```rust
// Grid index: qs[i] | ((qh << 8) & 0x300) | ...
let grid_idx = qs[i] | ((qh << 8) & 0x300);

// Signs from signs array using mask pattern
let signs = decode_signs(signs_array);

// Scale: 1 + 2*(scales[ib] & 0xf)
let scale = 1 + 2*(scales[ib] & 0xf);

// Final multiplier: 0.125
```

#### IQ3_XXS Pattern (from llama.cpp:3864-3923)
```rust
// Grid index from q3 (8-bit)
let grid_idx = q3[i]; // 256 entries

// Sign from gas (packed 7-bit indices)
let sign_idx = (aux32 >> 0) & 127;
let signs = KEVEN_SIGNS_Q2XS[sign_idx];

// Scale from high bits
let scale = 0.5 + (aux32 >> 28);

// Final multiplier: 0.5
```

#### IQ3_S Pattern (from llama.cpp:3926-4005)
```rust
// Grid index: qs[i] | ((qh << shift) & 256)
let grid_idx = qs[i] | ((qh << shift) & 256);

// Signs from signs array using mask pattern
let signs = decode_signs_mask(signs_array);

// Scale from scales array
let scale = ((scales32 & 0x0f0f0f0f) << 1) | 0x01010101;

// Final multiplier: 1.0 (no delta)
```

## Key Files

- **Rust kernels**: `src/quant/vec_dot/arm.rs`
- **Lookup tables**: `src/quant/vec_dot/tables.rs`
- **Type definitions**: `src/quant/types.rs`
- **llama.cpp reference**: `llama.cpp-0.4.1/ggml/src/ggml-cpu/arch/arm/quants.c`
- **llama.cpp tables**: `llama.cpp-0.4.1/ggml/src/ggml-common.h`

## Benchmark Setup

- **File**: `benches/vec_dot_neon_bench.rs`
- **Pattern**: Add generator function + benchmark function
- **Test size**: nb=256 (n=65536 for block=256, n=131072 for block=32)

## Implementation Steps

For each format:

1. **Extract lookup table**: Run Python script to extract from ggml-common.h
2. **Add to tables.rs**: Copy the generated Rust static array
3. **Implement kernel** in `arm.rs`:
   - Load grid values using `vld1_s8` (8 bytes per grid entry)
   - Load signs from `KEVEN_SIGNS_Q2XS` if needed
   - Apply scale and multiply with Q8_K values
   - Use `vdotq_s32_manual` for SDOT
4. **Add benchmark**: Create generator and benchmark in `vec_dot_neon_bench.rs`
5. **Verify correctness**: Compare output with llama.cpp
6. **Optimize**: Profile and optimize to exceed llama.cpp

## Performance Target

All formats should **exceed llama.cpp** (following Q4_K pattern which achieved 103-121%).

## Python Extraction Script

```python
# Extract grid from ggml-common.h
import re
with open('llama.cpp-0.4.1/ggml/src/ggml-common.h', 'r') as f:
    content = f.read()

match = re.search(r'GGML_TABLE_BEGIN\(uint64_t, iq1s_grid, NGRID_IQ1S\)(.*?)GGML_TABLE_END', content, re.DOTALL)
if match:
    data = match.group(1).strip()
    lines = [l.strip() for l in data.split('\n') if l.strip()]
    values = []
    for line in lines:
        for val in line.split(','):
            val = val.strip()
            if val.startswith('0x'):
                values.append(val)
    print(f"static IQ1S_GRID: [u64; {len(values)}] = [")
    for i in range(0, len(values), 4):
        chunk = values[i:i+4]
        print(f"    {', '.join(chunk)},")
    print("];")
```

---
**Last updated**: 2026-09-23
**Status**: In progress - tables extracted, kernels pending
**Next step**: Implement IQ1_S kernel
