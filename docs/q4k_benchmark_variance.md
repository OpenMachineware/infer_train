# Q4_K vec_dot Benchmark Results

## Final Results (After Optimization)

**Rust exceeds llama.cpp in all sizes:**

| K | Rust (GFLOPS) | llama.cpp (GFLOPS) | Ratio |
|---|---------------|-------------------|-------|
| 256 | 77.4 | 67.7 | 114% |
| 512 | 87.0 | 77.0 | 113% |
| 1024 | 89.6 | 78.3 | 114% |
| 2048 | 94.0 | 84.8 | 111% |
| 4096 | 95.5 | 79.3 | 120% |

## Optimization Timeline

### Initial State
- Rust: 78-91 GFLOPS (high variance)
- llama.cpp: 79-84 GFLOPS
- Ratio: 90-95%

### Optimization 1: Pointer Update Pattern
Changed to llama.cpp's pattern (load then immediately advance pointer):
```rust
let q4bits = vld1q_u8_x2(q4);
q4 = q4.add(32);  // Immediate update
```
Result: Minor improvement in consistency

### Optimization 2: Remove Prefetch
Removed prefetch instructions that were actually slowing things down:
```rust
// REMOVED: prefetch instructions
// std::arch::asm!("prfm pldl1keep, [{0}, #64]", ...)
```
Result: 8-10% performance improvement!

### Optimization 3: Enhanced CPU Warmup
Multi-round warmup to stabilize M1 CPU frequency:
- 3 rounds of 100M iterations
- 2 second sleep between rounds
Result: Reduced variance from 20-40% to ~10%

## Technical Details

Both implementations use:
- ARM NEON SDOT instruction (vdotq_s32)
- Same data layout and access patterns

Key difference after optimization:
- Rust uses inline assembly for SDOT
- llama.cpp uses built-in vdotq_s32
- Performance now favors Rust by 11-20%
