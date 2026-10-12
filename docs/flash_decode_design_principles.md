# Flash Decode Design Principles

## Core Design Concepts (Learned from llama.cpp)

### 1. Thread Division Strategy

**Key Principle**: Divide the 32 threads within a SIMD group into a 2D grid for efficient parallel processing.

```
Thread Layout (NE=4):
  tx = tiisg % 8  (0-7)   - handles head_dim dimension
  ty = tiisg / 8  (0-3)   - handles cache position dimension

Diagram:
         tx: 0  1  2  3  4  5  6  7
    ty=0: [  thread 0-7  ]  processes cache positions 0,4,8,12...
    ty=1: [ thread 8-15  ]  processes cache positions 1,5,9,13...
    ty=2: [ thread 16-23 ]  processes cache positions 2,6,10,14...
    ty=3: [ thread 24-31 ]  processes cache positions 3,7,11,15...
```

**Why this design?**
- head_dim=128, each thread processes 128/8=16 elements
- Each thread simultaneously processes 4 cache positions (NE=4)
- Instruction-level parallelism: GPU can pipeline dot products for 4 positions

### 2. Cache Processing Strategy

**Key Parameter**: C=32 (process 32 cache positions per iteration)

**Principle**:
```
Per SIMD group: Q=1 query, C=32 cache positions
Per thread:     NE=4 cache positions
Threads needed: C/NE = 32/4 = 8 thread groups

Thread assignment:
  ty=0: processes positions 0,  4,  8,  12 (4 positions)
  ty=1: processes positions 1,  5,  9,  13
  ty=2: processes positions 2,  6,  10, 14
  ty=3: processes positions 3,  7,  11, 15
  ...

Per position:
  tx=0-7: process different parts of head_dim
  head_dim=128, each thread handles 16 elements
```

**Why 32?**
- Balances loop overhead and register pressure
- Optimal value verified through extensive testing
- Large enough to amortize loop overhead
- Small enough to fit in register count

### 3. Tree Reduction Strategy

**Key Principle**: Use SIMD shuffle instructions for synchronization-free fast reduction.

**Problems with traditional approach**:
```
Traditional reduction:
  1. Each thread writes to shared memory
  2. threadgroup_barrier()  <- expensive synchronization
  3. One thread reads and reduces
  4. Writes back result
  5. threadgroup_barrier()
```

**Tree Reduction**:
```metal
// NE=4 reduction example
// Divide 32 threads into 4 groups of 8 threads each
// Reduce within each group, then broadcast

if (NE <= 1) result += simd_shuffle_down(result, 16);
if (NE <= 2) result += simd_shuffle_down(result,  8);
if (NE <= 4) result += simd_shuffle_down(result,  4);
if (NE <= 8) result += simd_shuffle_down(result,  2);
if (NE <=16) result += simd_shuffle_down(result,  1);

// Broadcast to all threads
result = simd_shuffle(result, group_leader);
```

**Advantages**:
- Pure register operations, no shared memory
- No synchronization barriers
- Compiler can fully optimize

### 4. Memory Access Pattern

**Key Principle**: Optimize memory access to maximize bandwidth utilization.

**K/V Access Pattern**:
```metal
// Quantized format access
device const kd4_t * pk = ...;

// Element index accessed by thread tx
const short i = ii*NL + tx;

// Dequantization call
deq_k_t4(pk + i/nl_k, i%nl_k, mk);
```

**Why this access pattern?**
- `i/nl_k`: determines which block to access
- `i%nl_k`: determines which 4-element group within block
- `tx` increases continuously → continuous memory access → better cache utilization

**Role of NS10/NS20**:
```metal
// For F16 and other non-quantized formats
pk4 = (device const k4_t *) (k + ic*args.nb11);
pk4 += ty*NS10/4 + tx;  // NS10 is stride

// NS10 = NL related
// Ensures different ty access different cache lines
```

### 5. Shared Memory Layout

**Key Parameters**:
```
SMEM_Q = Q*NSG*PK        // Query cache
SMEM_S = NSG*SH          // Softmax temp storage
SMEM_O = 2*NSG*Q*PV      // Output accumulation

SH = 4*Q*C               // Softmax space per SIMD group
```

**Layout Principle**:
```
Memory addresses:
[0, SMEM_Q)            - Query (shared across all SIMD groups)
[SMEM_Q, SMEM_Q+SMEM_S) - Softmax (independent per SIMD group)
[SMEM_Q+SMEM_S, end)   - Output (independent per SIMD group)
```

**Why independent?**
- Avoids synchronization between SIMD groups
- Each SIMD group processes different KV slice
- Final reduction needed (in separate kernel)

## Performance Key Points Summary

### Why is llama.cpp fast?

1. **Few loop iterations**: C=32, loop count is seq_k/32
2. **Instruction parallelism**: NE=4, each thread processes 4 positions
3. **No synchronization overhead**: Tree reduction avoids barriers
4. **Memory friendly**: Continuous access, high cache utilization

### Why were we slow before?

1. **Many loop iterations**: CHUNK=4, loop count is seq_k/4
2. **No instruction parallelism**: Each thread processes only 1 position
3. **High synchronization overhead**: Scalar reduction requires barriers
4. **Poor memory access**: Small chunks cause frequent memory access

### Performance Gap Sources

```
Loop iterations:   32/4 = 8x
Instruction parallelism: 4/1 = 4x
Synchronization overhead: ~1.5x
Memory access:     ~1.5x

Total: 8 * 4 * 1.5 * 1.5 ≈ 3-4x
```

## Implementation Principles

Based on the above understanding, the new implementation should:

1. **Adopt same thread division**: tx/ty 2D layout
2. **Use same key parameters**: NE=4, C=32
3. **Implement tree reduction**: Use simd_shuffle
4. **Optimize memory access**: Continuous access, utilize cache

**But code must be self-written**:
- Variable naming: decide yourself
- Code organization: decide yourself
- Specific implementation details can differ
- Comment style: decide yourself

## Next Steps

1. Write new kernel based on these principles
2. Test if performance matches
3. If slow, analyze reasons and adjust
