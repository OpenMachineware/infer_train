# Extending Operator Fusion (M5)

This guide explains how the M5 fusion passes rewrite the optimizer IR, how to find fusion
opportunities, and how to add a new pattern with verification. Read these files first:

- `src/core/optimizer/dag_ir.mojo` — the compact DAG IR (`Dag`, `DagNode`, op codes).
- `src/core/optimizer/fusion.mojo` — the pattern matcher (`dag_fusion`).
- `src/core/optimizer/dag_optimizer.mojo` — the pipeline driver (`optimize_dag`).
- `src/core/optimizer/verify.mojo` — the correctness harness (`verify_dags`).

Conventions: Mojo 1.0 (only `def`, `mut self`, `^` moves), no dynamic dispatch — the IR is a
flat node list with integer op codes and index-based edges, deliberately small so passes and
the verify executor share one structure.

---

## 1. How the M5 fusion passes work

### 1.1 The DAG IR (`dag_ir.mojo`)

A `Dag` is a flat `List[DagNode]` plus `outputs: List[Int]` and `n_inputs`. Each node has an
integer `op` code, `inputs: List[Int]` (indices into `nodes`), a dtype code, a runtime
`shape: List[Int]`, and an optional inline constant `const_data: Optional[AnyTensor]`.

```mojo
comptime OP_CONST = 0
comptime OP_INPUT = 1
comptime OP_MATMUL = 2
comptime OP_LM_HEAD = 3
comptime OP_ADD = 4
comptime OP_ADD_BIAS = 5
comptime OP_RMS_NORM = 6
comptime OP_SOFTMAX = 7
comptime OP_SWIGLU = 8
comptime OP_EMBEDDING = 9
comptime OP_FUSED_MATMUL_ADD_BIAS = 10
comptime OP_FUSED_MATMUL_ADD = 11
comptime OP_FUSED_MATMUL_RMS_NORM = 12
comptime OP_FUSED_SWIGLU_MATMUL = 13
comptime OP_COUNT = 14
```

Two helpers matter for fusion: `op_name(op) -> String` maps an op code back to the registry
name (so the verify executor can dispatch through `OpRegistry`), and `max_inputs_of(op)`
bounds arity. `op_is_fused(op)` is `op >= OP_FUSED_MATMUL_ADD_BIAS and op < OP_COUNT`.

### 1.2 The pipeline (`dag_optimizer.mojo`)

`optimize_dag(mut dag, flags)` runs, in order:

```
shape_inference -> constant_fold -> simplify -> fusion -> cse
  -> shape_inference -> dce -> memory_plan
```

Fusion is a single pass (`flags & 8` skips it). The contract fusion relies on: a fused node
must carry a fused **op code**, and that op code's `op_name` must be registered in
`OpRegistry.register_default_ops()` so the runtime and the verify executor can both run it.

### 1.3 The matcher (`fusion.mojo`)

`dag_fusion(mut dag) -> Int` scans every node and rewrites **consumer → producer** chains
in place. The three existing patterns:

- `add_bias(lm_head(x, w), bias)` → `fused_matmul_add_bias(x, w, bias)`
- `rms_norm(lm_head(x, w))` → `fused_matmul_rms_norm(x, w)`
- `lm_head(swiglu(gate, up), w)` → `fused_swiglu_matmul(gate, up, w)`

The single-consumer guard is the key correctness invariant — fusing must never duplicate
work, so the producer is rewritten only when exactly one node consumes it:

```mojo
def _use_count(dag: Dag, node_id: Int) -> Int:
    var count = 0
    for node in dag.nodes:
        for input_id in node.inputs:
            if input_id == node_id:
                count += 1
    return count
```

Here is the real `lm_head + add_bias` pattern, verbatim structure:

```mojo
if dag.nodes[i].op == OP_ADD_BIAS and len(dag.nodes[i].inputs) == 2:
    var producer = dag.nodes[i].inputs[0]
    var bias = dag.nodes[i].inputs[1]
    if (
        dag.nodes[producer].op == OP_LM_HEAD
        and len(dag.nodes[producer].inputs) == 2
        and _use_count(dag, producer) == 1
    ):
        dag.nodes[i].op = OP_FUSED_MATMUL_ADD_BIAS
        var inputs = List[Int]()
        inputs.append(dag.nodes[producer].inputs[0])
        inputs.append(dag.nodes[producer].inputs[1])
        inputs.append(bias)
        dag.nodes[i].inputs = inputs^
        dag.nodes[producer].op = OP_CONST      # dead; dce removes it
        dag.nodes[producer].inputs = List[Int]()
        fused += 1
```

The rewrite consumes the fused consumer in place (`dag.nodes[i].op = ...`), then marks the
producer `OP_CONST` with no inputs so `dce` (run later in the same pipeline) reclaims it.

---

## 2. Identifying fusion opportunities

1. **Profile the interpreter.** The interpreter (`src/runtime/interpreter.mojo`) dispatches
   every node through `OpRegistry.get(...).forward(...)`. Instrument `Interpreter.run` to
   count calls per `op_type`, or print `summarize_dag(dag)` from `dag_optimizer.mojo`, which
   emits a node histogram:

   ```mojo
   print(summarize_dag(dag))  # "DAG: 42 nodes (lm_head x4, add_bias x4, ...)"
   ```

2. **Look for producer→consumer chains** whose intermediate tensor is large (e.g. an
   `lm_head` output fed straight into an `add_bias`, `add`, or `rms_norm`). Fusing saves a
   full tensor allocation + write + read per chain — the memory-planning pass
   (`memory_plan.mojo`) reports `baseline_bytes` vs `planned_bytes`, and fusions that remove
   a materialized intermediate shrink the peak.

3. **Check the fused kernel already exists.** All three fused kernels live in
   `src/core/ops/fused/` (`matmul_add.mojo`, `matmul_rms_norm.mojo`, `swiglu_matmul.mojo`)
   and are already registered under `fused_*` names. A fourth op code,
   `OP_FUSED_MATMUL_ADD`, and its kernel `fused_matmul_add` exist **but no pass emits it yet**
   — that is the natural first extension (see §3).

4. **Only fuse weight-major matmuls.** The fused kernels fold `lm_head` (which uses
   `matmul_weight_cpu`, GGUF `[out, in]` layout), not the plain `matmul` op. Match
   `OP_LM_HEAD`, never `OP_MATMUL`, when the fused kernel assumes weight-major `w`.

---

## 3. Implementing a new fusion pass

### 3.1 Add the pattern matcher — `src/core/optimizer/fusion.mojo`

The missing `lm_head + add → fused_matmul_add` pattern. First extend the import:

```mojo
from .dag_ir import (
    Dag, OP_CONST, OP_LM_HEAD, OP_ADD, OP_ADD_BIAS, OP_RMS_NORM, OP_SWIGLU,
    OP_FUSED_MATMUL_ADD_BIAS, OP_FUSED_MATMUL_ADD, OP_FUSED_MATMUL_RMS_NORM,
    OP_FUSED_SWIGLU_MATMUL,
)
```

Then add an `elif` branch to `dag_fusion` (placed with the other `lm_head` consumers):

```mojo
elif dag.nodes[i].op == OP_ADD and len(dag.nodes[i].inputs) == 2:
    var producer = dag.nodes[i].inputs[0]
    var b = dag.nodes[i].inputs[1]
    if (
        dag.nodes[producer].op == OP_LM_HEAD
        and len(dag.nodes[producer].inputs) == 2
        and _use_count(dag, producer) == 1
    ):
        dag.nodes[i].op = OP_FUSED_MATMUL_ADD
        var inputs = List[Int]()
        inputs.append(dag.nodes[producer].inputs[0])
        inputs.append(dag.nodes[producer].inputs[1])
        inputs.append(b)
        dag.nodes[i].inputs = inputs^
        dag.nodes[producer].op = OP_CONST
        dag.nodes[producer].inputs = List[Int]()
        fused += 1
```

`OP_FUSED_MATMUL_ADD` already has `max_inputs_of == 3` and `op_name == "fused_matmul_add"`,
and `fused_matmul_add` is already registered in `register_default_ops()`, so no other wiring
is needed for this particular pattern. For a **brand-new** fused op you would additionally:

- add a new `OP_FUSED_*` code (and bump `OP_COUNT`), a `max_inputs_of` case, and an
  `op_name` branch in `dag_ir.mojo`;
- write the kernel in `src/core/ops/fused/<name>.mojo`;
- add `_dispatch_cpu/_dispatch_gpu` + two `self.register(...)` entries in
  `op_registry.mojo`.

### 3.2 Verify with `src/core/optimizer/verify.mojo`

The verify harness runs the original and optimized DAGs through the **same** interpretive
executor (`execute_dag`, which dispatches via `OpRegistry`) on identical small inputs and
compares every output element:

```mojo
def verify_dags(
    original: Dag,
    optimized: Dag,
    inputs: List[AnyTensor],
    tolerance: Float32 = Float32(1e-4),
) -> Bool:
    var before = execute_dag(original, inputs)
    var after = execute_dag(optimized, inputs)
    ...
    for i in range(len(before)):
        var diff = _max_abs_diff(before[i], after[i])
        if diff > tolerance:
            print("[verify] FAIL: output ", i, " max diff ", diff)
            return False
    return True
```

`verify_optimization(original, mut optimized, inputs, pass_name, tolerance)` wraps this with
an abort-style failure message naming the pass. Use it after a rewrite to prove semantics are
preserved:

```mojo
var ok = verify_optimization(original, optimized, inputs, "fusion", Float32(1e-4))
if not ok:
    return  # stop the pipeline on a broken pass
```

### 3.3 Testing

The optimizer suite is `tests/test_optimizer.mojo` (built by `make test-m5-mojo`). Add a
case that:

1. builds a `Dag` with the pre-fusion chain (`OP_LM_HEAD` → `OP_ADD`),
2. clones it with `clone_dag` (from `verify.mojo`),
3. runs `dag_fusion` on the clone,
4. asserts the fused node has `op == OP_FUSED_MATMUL_ADD` and the producer is dead,
5. calls `verify_dags(original, fused, inputs)` and checks it returns `True`.

```mojo
var original = build_dag()          # lm_head -> add
var fused = clone_dag(original)
var n = dag_fusion(fused)
assert n == 1, "expected one fusion"
assert verify_dags(original, fused, test_inputs), "fusion changed semantics"
```

Also assert the fused histogram: `summarize_dag(fused)` should show `fused_matmul_add x1`
and one fewer live node after `dag_dce`.

Build and run:

```bash
pixi run mojo build -I . tests/test_optimizer.mojo \
  -Xlinker python/infer_train/_lib/libinfer_train_tp.dylib -o tests/test_optimizer
./tests/test_optimizer
```

---

## 4. Pitfalls

- **Single-consumer rule.** Never fuse a producer with more than one consumer — that would
  recompute the producer's output (or corrupt other edges). Always check `_use_count == 1`.
- **Op-code registration.** A fused op code with no `op_name` → registry entry will hit
  `OpRegistry.get`'s `unimplemented("operator not registered: ...")` at verify time.
- **Weight-major layout.** `fused_matmul_add_bias` / `fused_matmul_add` /
  `fused_matmul_rms_norm` / `fused_swiglu_matmul` all assume `w` is `[out, in]` (GGUF
  layout). Fuse only `OP_LM_HEAD` producers, not `OP_MATMUL`.
- **Shape inference order.** `dag_shape_inference` runs before fusion; a new fused node's
  shape comes from the consumer's original shape, so keep the consumer's `shape` field intact
  (do not clear it) during the rewrite.
