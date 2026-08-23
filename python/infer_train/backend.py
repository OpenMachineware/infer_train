"""torch.compile backend: capture FX graphs and translate them onto the engine.

Pipeline (M5):

    torch.nn.Module
        |  torch.compile(model, backend="infer_train")
        v
    torch.fx.GraphModule   (captured by torch._dynamo)
        |  infer_train_backend(gm, example_inputs)
        v
    translated execution plan
        |  per FX node: native engine op | torch fallback | control flow
        v
    callable(*args) -> outputs

M5 control flow (CFG): ``torch.cond`` / ``torch.while_loop`` (and their
``torch.ops.higher_order.*`` spellings) are translated recursively - each
branch / body GraphModule becomes its own compiled sub-plan executed at
runtime with the loop-carried / operand tensors.  Conditions that are
compile-time constants are resolved *statically*: only the taken ``cond``
branch is compiled, and a ``while_loop`` whose condition/body touch no
runtime inputs is unrolled at compile time into plain constants.  Other
higher-order ops (``map``/``scan``/``trampoline``/``wrap``) still raise
:class:`ControlFlowError` - they land in M6.

Translation rules (``_translate_node``):

* ``placeholder``            -> graph input
* ``get_attr``               -> constant (weights/biases, moved to CPU)
* ``output``                 -> result selection
* ``call_function/method/module`` -> mapped through the aten-name table:

  ======================  =============================================
  FX op                    engine execution
  ======================  =============================================
  mm / matmul / bmm (2D)   ``matmul``
  addmm / linear           ``lm_head`` + ``add_bias`` (composite)
  add (same-shape 2D)      ``add``
  add (2D + 1D bias)       ``add_bias``
  softmax (dim=-1, 2D)     ``softmax``
  rms_norm (no weight)     ``rms_norm``
  mul(silu(g), u)          ``swiglu`` (pattern match)
  embedding (1D indices)   ``embedding``
  cond / while_loop        recursive sub-plan (static or runtime CFG)
  everything else          torch fallback (view/transpose/div/...)
  ======================  =============================================

Anything Dynamo cannot capture (Python control flow on tensors) fails in
Dynamo itself before reaching this backend; the higher-order ops M5 does
not support (``map``, ``scan``, ...) raise :class:`ControlFlowError` with
a friendly message.  Unknown ops fall back to torch execution by default;
``strict=True`` turns that into :class:`UnsupportedOpError`.

M5 scope: inference only (``model.eval()``); training mode raises.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.fx

from .binding import (
    EngineError,
    DTYPE_F32,
    DTYPE_F16,
    DTYPE_I32,
    Tensor as EngineTensor,
    run_op,
)

__all__ = [
    "infer_train_backend",
    "translate_graph",
    "UnsupportedOpError",
    "ControlFlowError",
]

_TORCH_DTYPE_TO_CODE = {
    torch.float32: DTYPE_F32,
    torch.float16: DTYPE_F16,
    torch.int32: DTYPE_I32,
    torch.int64: DTYPE_I32,  # token ids: engine reads i32
}
_CODE_TO_TORCH_DTYPE = {
    DTYPE_F32: torch.float32,
    DTYPE_F16: torch.float16,
    DTYPE_I32: torch.int32,
}

_HIGHER_ORDER = ("cond", "while_loop", "map", "scan", "wrap", "trampoline")
_M5_SUPPORTED_HOP = ("cond", "while_loop")

# The most recently produced CompiledInferTrain (introspection hook for
# tests and tooling; torch.compile's OptimizedModule hides the callable
# behind closures).
last_compiled: Optional["CompiledInferTrain"] = None


class UnsupportedOpError(RuntimeError):
    """Raised when an FX node cannot be translated to the engine.

    The message tells the user which node failed and how to restructure the
    model (e.g. avoid control flow, use static shapes, keep to supported
    dtypes).
    """


class ControlFlowError(UnsupportedOpError):
    """Raised for FX control-flow nodes M5 does not support yet
    (map/scan/trampoline/wrap; cond/while_loop are supported)."""


@dataclass
class _Op:
    """One execution step of the translated graph."""

    name: str  # FX node name the result is stored under
    kind: str  # input | const | native | fallback | control_flow | output
    engine_op: Optional[str] = None
    arg_names: List[str] = field(default_factory=list)
    fn: Optional[Callable[..., Any]] = None
    # composite = list of (engine_op, arg_names) executed in order
    composite: Optional[List[Tuple[str, List[str]]]] = None
    fx_node: Any = None
    # control-flow spec (kind == "control_flow"): a _CondSpec or _WhileSpec
    ctrl: Any = None


# ---------------------------------------------------------------------------
# M5 control flow: CFG capture and recursive sub-plan translation
# ---------------------------------------------------------------------------


@dataclass
class _CondSpec:
    """A translated `torch.cond` node.

    ``pred`` is either an FX node name (runtime predicate) or a Python
    constant (static condition - only the taken branch is ever compiled).
    ``operand_names`` lists the env names of the branch inputs.
    """

    pred: Any  # str (env name) | constant
    true_gm: Any
    false_gm: Any
    operand_names: List[str]
    strict: bool
    static: bool = False  # resolved at compile time
    _plans: Optional[Tuple[Any, Any]] = None  # (true_plan, false_plan) cache


@dataclass
class _WhileSpec:
    """A translated `torch.while_loop` node.

    ``carried_names`` are the loop-carried env names, ``additional_names``
    the loop-invariant inputs.  ``static_result`` is set when the loop body
    and condition touch no runtime inputs: it is then unrolled at compile
    time and the loop node becomes a constant.
    """

    cond_gm: Any
    body_gm: Any
    carried_names: List[str]
    additional_names: List[str]
    strict: bool
    static_result: Any = None
    _plans: Optional[Tuple[Any, Any]] = None  # (cond_plan, body_plan) cache


def _resolve_branch_gm(gm: torch.fx.GraphModule, spec: Any) -> Any:
    """Resolve a cond/while branch spec to its GraphModule.

    Accepts an fx.Node (get_attr), a plain GraphModule, or a dict with a
    "graphmodule" entry (the kwargs["submodules"] shape some torch
    versions use).
    """
    if isinstance(spec, torch.fx.Node):
        return _resolve_attr(gm, str(spec.target))
    if isinstance(spec, torch.fx.GraphModule):
        return spec
    if isinstance(spec, dict):
        for v in spec.values():
            if isinstance(v, torch.fx.GraphModule):
                return v
    raise ControlFlowError(
        "[infer_train] control-flow branch could not be resolved: "
        f"{spec!r}"
    )


def _node_or_const_names(args: Any) -> List[str]:
    """Map a tuple of FX args to env names (non-nodes are dropped)."""
    names: List[str] = []
    for a in args:
        if isinstance(a, torch.fx.Node):
            names.append(a.name)
    return names


def _to_tuple(value: Any) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list) or type(value).__name__ == "immutable_list":
        return tuple(value)
    return (value,)


def _build_subplan(
    gm: torch.fx.GraphModule,
    example_inputs: List[Any],
    strict: bool,
    shape_mode: bool,
    stats: Optional[Dict[str, int]] = None,
) -> "CompiledInferTrain":
    """Translate a branch/body GraphModule into a compiled sub-plan."""
    plan, _s = translate_graph(gm, example_inputs, strict=strict)
    if stats is not None:
        for k, v in _s.items():
            stats[k] = stats.get(k, 0) + v
    return CompiledInferTrain(plan, gm, _s, shape_mode=shape_mode)


def _cond_spec_from_node(
    node: torch.fx.Node, gm: torch.fx.GraphModule, strict: bool
) -> Tuple[_CondSpec, Any]:
    """Build a _CondSpec from a cond higher-order-op FX node.

    Returns (spec, static_result_or_None).  ``static_result`` is set when
    the predicate is a compile-time constant: only that branch is kept.
    """
    args = list(node.args)
    # torch.cond layout: (pred, true_fn, false_fn, operands) or
    # (pred, true_fn, false_fn) with operands in kwargs["operands"].
    if len(args) >= 4:
        pred, true_spec, false_spec, operands = args[0], args[1], args[2], args[3]
    elif len(args) == 3 and node.kwargs.get("operands") is not None:
        pred, true_spec, false_spec = args
        operands = node.kwargs["operands"]
    else:
        raise ControlFlowError(
            f"[infer_train] node '{node.name}' (cond) has an unsupported "
            f"argument layout {node.args!r}; expected "
            "(pred, true_fn, false_fn, operands)."
        )
    true_gm = _resolve_branch_gm(gm, true_spec)
    false_gm = _resolve_branch_gm(gm, false_spec)
    operand_names = _node_or_const_names(operands)
    spec = _CondSpec(
        pred=pred.name if isinstance(pred, torch.fx.Node) else pred,
        true_gm=true_gm,
        false_gm=false_gm,
        operand_names=operand_names,
        strict=strict,
    )
    static = None
    if not isinstance(pred, torch.fx.Node):
        # compile-time-known predicate: resolve statically (M5 static CFG)
        truth = bool(pred.item() if isinstance(pred, torch.Tensor) else pred)
        spec.static = True
        static = "true" if truth else "false"
    return spec, static


def _while_spec_from_node(
    node: torch.fx.Node, gm: torch.fx.GraphModule, strict: bool
) -> _WhileSpec:
    """Build a _WhileSpec from a while_loop higher-order-op FX node."""
    args = list(node.args)
    if len(args) >= 4:
        cond_spec, body_spec, carried, additional = (
            args[0], args[1], args[2], args[3],
        )
    elif len(args) == 3 and node.kwargs.get("carried_inputs") is not None:
        cond_spec, body_spec, carried = args
        additional = node.kwargs["carried_inputs"]
    else:
        raise ControlFlowError(
            f"[infer_train] node '{node.name}' (while_loop) has an "
            f"unsupported argument layout {node.args!r}; expected "
            "(cond_fn, body_fn, carried, additional)."
        )
    cond_gm = _resolve_branch_gm(gm, cond_spec)
    body_gm = _resolve_branch_gm(gm, body_spec)
    return _WhileSpec(
        cond_gm=cond_gm,
        body_gm=body_gm,
        carried_names=_node_or_const_names(carried),
        additional_names=_node_or_const_names(additional),
        strict=strict,
    )


def _placeholder_count(gm: torch.fx.GraphModule) -> int:
    return len(list(gm.graph.find_nodes(op="placeholder")))


def _cond_runner(
    spec: _CondSpec, env: Dict[str, Any], shape_mode: bool = False
) -> Any:
    """Execute a cond node: pick the branch and run its sub-plan."""
    if spec._plans is None:
        spec._plans = (None, None)
    if spec.static:
        pred = spec.pred
        truth = bool(pred.item() if isinstance(pred, torch.Tensor) else pred)
    else:
        pred_value = env[spec.pred] if isinstance(spec.pred, str) else spec.pred
        if isinstance(pred_value, torch.Tensor):
            pred_value = bool(pred_value.item() if pred_value.numel() == 1 else pred_value)
        truth = bool(pred_value)
    operands = [env[n] for n in spec.operand_names]
    if truth:
        plan = spec._plans[0]
        if plan is None:
            plan = _build_subplan(spec.true_gm, operands, spec.strict, shape_mode)
            spec._plans = (plan, spec._plans[1])
        return plan(*operands)
    plan = spec._plans[1]
    if plan is None:
        plan = _build_subplan(spec.false_gm, operands, spec.strict, shape_mode)
        spec._plans = (spec._plans[0], plan)
    return plan(*operands)


def _while_runner(
    spec: _WhileSpec, env: Dict[str, Any], shape_mode: bool = False
) -> Any:
    """Execute a while_loop node: iterate cond/body sub-plans on carried
    tensors until the condition is false."""
    if spec.static_result is not None:
        return spec.static_result
    if spec._plans is None:
        spec._plans = (None, None)
    carried = tuple(env[n] for n in spec.carried_names)
    additional = tuple(env[n] for n in spec.additional_names)
    cond_plan = spec._plans[0]
    if cond_plan is None:
        examples = list(carried) + list(additional)
        cond_plan = _build_subplan(spec.cond_gm, examples, spec.strict, shape_mode)
        spec._plans = (cond_plan, spec._plans[1])
    body_plan = spec._plans[1]
    if body_plan is None:
        examples = list(carried) + list(additional)
        body_plan = _build_subplan(spec.body_gm, examples, spec.strict, shape_mode)
        spec._plans = (spec._plans[0], body_plan)
    if shape_mode:
        # shape inference only: zero iterations, return carried shapes
        return carried
    iterations = 0
    while True:
        cond_value = cond_plan(*carried, *additional)
        cond_tensor = _to_tuple(cond_value)[0]
        if isinstance(cond_tensor, torch.Tensor):
            keep = bool(cond_tensor.item() if cond_tensor.numel() == 1 else cond_tensor)
        else:
            keep = bool(cond_tensor)
        if not keep:
            break
        body_out = body_plan(*carried, *additional)
        carried = _to_tuple(body_out)
        iterations += 1
        if iterations > 1_000_000:
            raise ControlFlowError(
                "[infer_train] while_loop exceeded 1e6 iterations - "
                "likely a non-terminating loop."
            )
    return carried


def _unroll_static_while(
    spec: _WhileSpec, const_values: Dict[str, Any]
) -> bool:
    """Try to unroll a while_loop at compile time.

    Returns True when the loop was fully unrolled (spec.static_result set):
    every carried/additional input must be a compile-time constant (checked
    by the caller via ``const_names``), so the loop executes entirely on
    constants at translation time.  Guards against runaway unrolling.
    """
    carried = []
    for n in spec.carried_names:
        v = const_values.get(n)
        if v is None:
            return False
        carried.append(v)
    additional = [const_values.get(n) for n in spec.additional_names]
    if any(v is None for v in additional):
        return False
    cond_plan = _build_subplan(spec.cond_gm, list(carried) + list(additional),
                               spec.strict, shape_mode=False)
    body_plan = _build_subplan(spec.body_gm, list(carried) + list(additional),
                               spec.strict, shape_mode=False)
    limit = 100_000
    for _ in range(limit):
        cond_value = cond_plan(*carried, *additional)
        if not bool(_to_tuple(cond_value)[0].item()):
            spec.static_result = tuple(carried)
            return True
        carried = _to_tuple(body_plan(*carried, *additional))
    raise ControlFlowError(
        "[infer_train] static while_loop did not terminate within "
        f"{limit} iterations; refusing to unroll."
    )


def _run_subplan_shapes(
    gm: torch.fx.GraphModule,
    example_inputs: List[Any],
    strict: bool,
) -> Any:
    """Dry-run a sub-plan in shape mode to learn its output shapes."""
    plan, _ = translate_graph(gm, example_inputs, strict=strict)
    return CompiledInferTrain(plan, gm, None, shape_mode=True)(*example_inputs)


def _target_name(target: Any) -> str:
    """Normalize an FX node target to its base op name ('mm', 'add', ...)."""
    if isinstance(target, str):
        return target.split("::")[-1].split(".")[-1]
    name = getattr(target, "name", None)
    if callable(name):
        try:
            name = name()
        except Exception:
            name = None
    if not name:
        name = getattr(target, "__name__", None)
    if not name:
        name = str(target)
    name = str(name).replace("::", ".")
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] in ("aten", "torch", "prims"):
        return parts[1]
    return parts[-1]


def _arg_values(node: Any, env: Dict[str, Any]) -> Tuple[list, dict]:
    """Materialize a node's arguments from the env."""
    args = []
    for a in node.args:
        if isinstance(a, torch.fx.Node):
            args.append(env[a.name])
        else:
            args.append(a)
    kwargs = {}
    for k, v in node.kwargs.items():
        if isinstance(v, torch.fx.Node):
            kwargs[k] = env[v.name]
        else:
            kwargs[k] = v
    return args, kwargs


def _resolve_attr(gm: torch.fx.GraphModule, target: str) -> Any:
    obj: Any = gm
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def _is_float_tensor(t: Any) -> bool:
    return isinstance(t, torch.Tensor) and t.dtype in (
        torch.float32,
        torch.float16,
    )


def _shape_ok_2d(t: Any) -> bool:
    return isinstance(t, torch.Tensor) and t.dim() == 2


def _translate_node(
    node: torch.fx.Node,
    gm: torch.fx.GraphModule,
    env_shapes: Dict[str, Any],
    strict: bool,
    stats: Dict[str, int],
    prev_kind: Dict[str, str],
    const_names: Optional[set] = None,
    const_values: Optional[Dict[str, Any]] = None,
    prev_ops: Optional[Dict[str, _Op]] = None,
) -> _Op:
    """Translate one FX node into an _Op plan step."""
    name = node.name
    if const_names is None:
        const_names = set()
    if const_values is None:
        const_values = {}
    if prev_ops is None:
        prev_ops = {}

    if node.op == "placeholder":
        stats["inputs"] += 1
        return _Op(name=name, kind="input", fx_node=node)

    if node.op == "get_attr":
        value = _resolve_attr(gm, str(node.target))
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.device.type == "cuda":
                warnings.warn(
                    f"[infer_train] parameter '{node.target}' is on CUDA; "
                    "the M4 backend runs on CPU, moving it to CPU."
                )
                value = value.cpu()
        stats["consts"] += 1
        return _Op(name=name, kind="const", fx_node=node, fn=lambda v=value: v)

    if node.op == "output":
        stats["outputs"] += 1
        return _Op(name=name, kind="output", fx_node=node)

    if node.op not in ("call_function", "call_method", "call_module"):
        raise UnsupportedOpError(
            f"[infer_train] node '{name}' has unsupported op kind "
            f"'{node.op}' - this usually means Python-level control flow "
            "(if/while on tensors) Dynamo could not capture. Use "
            "torch.cond/torch.while_loop (M5 supports them) or rewrite "
            "the model with plain tensor ops."
        )

    target = node.target
    opname = _target_name(target)

    is_hop = any(
        opname == h or opname.startswith(h + "_") for h in _HIGHER_ORDER
    )
    if is_hop:
        if opname in _M5_SUPPORTED_HOP:
            stats["control_flow"] = stats.get("control_flow", 0) + 1
            if opname == "cond":
                spec, static = _cond_spec_from_node(node, gm, strict)
                op = _Op(
                    name=name,
                    kind="control_flow",
                    ctrl=spec,
                    fx_node=node,
                )
                if static is not None:
                    # static condition: drop the dead branch and mark the
                    # plan so the runner inlines only the taken branch.
                    stats["static_cf"] = stats.get("static_cf", 0) + 1
                    return op
                return op
            if opname == "while_loop":
                spec = _while_spec_from_node(node, gm, strict)
                # static loop: every carried/additional input is a
                # compile-time constant -> unroll at translation time
                if all(
                    n in const_names
                    for n in spec.carried_names + spec.additional_names
                ) and _unroll_static_while(spec, const_values):
                    stats["static_cf"] = stats.get("static_cf", 0) + 1
                    stats["consts"] += 1
                    return _Op(
                        name=name,
                        kind="const",
                        fn=lambda s=spec: s.static_result,
                        fx_node=node,
                    )
                return _Op(
                    name=name,
                    kind="control_flow",
                    ctrl=spec,
                    fx_node=node,
                )
        raise ControlFlowError(
            f"[infer_train] node '{name}' uses control flow ({opname}), "
            "which is not supported yet in M5 (cond/while_loop are "
            "supported; map/scan/trampoline land in M6)."
        )

    # shapes of the node's tensor arguments (from the traced env)
    def arg_tensor(i: int) -> Optional[torch.Tensor]:
        if i < len(node.args) and isinstance(node.args[i], torch.fx.Node):
            v = env_shapes.get(node.args[i].name)
            if isinstance(v, torch.Tensor):
                return v
        return None

    kwargs = node.kwargs

    # -- native translations ------------------------------------------------

    # matmul family: aten.mm / aten.matmul / aten.bmm on rank-2 tensors
    if opname in ("mm", "matmul", "bmm"):
        a, b = arg_tensor(0), arg_tensor(1)
        if (
            _is_float_tensor(a)
            and _is_float_tensor(b)
            and _shape_ok_2d(a)
            and _shape_ok_2d(b)
            and a.dtype == b.dtype
            and a.shape[1] == b.shape[0]
        ):
            stats["native"] += 1
            prev_kind[name] = "native"
            return _Op(
                name=name,
                kind="native",
                engine_op="matmul",
                arg_names=[node.args[0].name, node.args[1].name],
                fx_node=node,
            )

    # linear family: aten.addmm(bias, x, w) / aten.linear(x, w, b)
    if opname in ("addmm", "linear"):
        if opname == "addmm":
            bias, x, w = arg_tensor(0), arg_tensor(1), arg_tensor(2)
        else:
            x = arg_tensor(0)
            w = arg_tensor(1)
            bias = arg_tensor(2)
            if bias is None and len(node.args) > 2 and node.args[2] is not None:
                bias = env_shapes.get(node.args[2].name)
        if (
            _is_float_tensor(x)
            and _is_float_tensor(w)
            and _shape_ok_2d(x)
            and _shape_ok_2d(w)
            and x.dtype == w.dtype
            and (x.shape[1] == w.shape[0] or x.shape[1] == w.shape[1])
        ):
            xname = node.args[1].name if opname == "addmm" else node.args[0].name
            wname = node.args[2].name if opname == "addmm" else node.args[1].name
            bias_name = None
            bias_shape_ok = (
                bias is not None
                and isinstance(bias, torch.Tensor)
                and bias.dim() == 1
                and _is_float_tensor(bias)
                and bias.dtype == x.dtype
                and bias.shape[0] == w.shape[0]
            )
            if bias_shape_ok:
                bias_name = (
                    node.args[0].name
                    if opname == "addmm"
                    else node.args[2].name
                )
            elif bias is not None and not (
                isinstance(bias, torch.Tensor) and bias.dim() == 0
            ):
                bias_name = None
            # M5 fusion patterns (weight-major layout only):
            #   linear(swiglu(g, u), w) -> fused_swiglu_matmul(g, u, w)
            #   linear(x, w, bias)     -> fused_matmul_add_bias(x, w, bias)
            src_op = prev_ops.get(xname)
            if (
                x.shape[1] == w.shape[1]
                and src_op is not None
                and src_op.engine_op == "swiglu"
            ):
                steps: List[Tuple[str, List[str]]] = [
                    (
                        "fused_swiglu_matmul",
                        [src_op.arg_names[0], src_op.arg_names[1], wname],
                    )
                ]
                if bias_name is not None:
                    steps.append(("add_bias", ["__out0__", bias_name]))
                stats["native"] += 1
                stats["fused"] = stats.get("fused", 0) + 1
                prev_kind[name] = "native"
                return _Op(
                    name=name,
                    kind="native",
                    composite=steps,
                    fx_node=node,
                )
            if x.shape[1] == w.shape[1] and bias_name is not None:
                stats["native"] += 1
                stats["fused"] = stats.get("fused", 0) + 1
                prev_kind[name] = "native"
                return _Op(
                    name=name,
                    kind="native",
                    composite=[("fused_matmul_add_bias", [xname, wname, bias_name])],
                    fx_node=node,
                )
            # torch semantics: y = x @ w^T (+ bias).  With w stored
            # [out, in] the engine's lm_head computes exactly x @ w^T;
            # with w already transposed ([in, out]) a plain matmul does.
            steps = []
            if x.shape[1] == w.shape[1]:
                steps.append(("lm_head", [xname, wname]))
            else:
                steps.append(("matmul", [xname, wname]))
            if bias_name is not None:
                steps.append(("add_bias", ["__out0__", bias_name]))
            elif bias is not None and isinstance(bias, torch.Tensor) and bias.dim() == 0:
                # scalar bias: fall back to torch for the whole node
                steps = []
            if steps:
                stats["native"] += 1
                prev_kind[name] = "native"
                return _Op(
                    name=name,
                    kind="native",
                    composite=steps,
                    fx_node=node,
                )

    # elementwise add
    if opname == "add":
        a, b = arg_tensor(0), arg_tensor(1)
        if (
            _is_float_tensor(a)
            and _is_float_tensor(b)
            and _shape_ok_2d(a)
            and _shape_ok_2d(b)
            and a.dtype == b.dtype
            and a.shape == b.shape
        ):
            stats["native"] += 1
            prev_kind[name] = "native"
            return _Op(
                name=name,
                kind="native",
                engine_op="add",
                arg_names=[node.args[0].name, node.args[1].name],
                fx_node=node,
            )
        if (
            _is_float_tensor(a)
            and _is_float_tensor(b)
            and _shape_ok_2d(a)
            and b.dim() == 1
            and a.dtype == b.dtype
            and a.shape[1] == b.shape[0]
        ):
            stats["native"] += 1
            prev_kind[name] = "native"
            return _Op(
                name=name,
                kind="native",
                engine_op="add_bias",
                arg_names=[node.args[0].name, node.args[1].name],
                fx_node=node,
            )

    # softmax along the last dim of a 2D tensor
    if opname == "softmax":
        x = arg_tensor(0)
        dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim", -1)
        if (
            _is_float_tensor(x)
            and _shape_ok_2d(x)
            and dim in (-1, x.dim() - 1)
        ):
            stats["native"] += 1
            prev_kind[name] = "native"
            return _Op(
                name=name,
                kind="native",
                engine_op="softmax",
                arg_names=[node.args[0].name],
                fx_node=node,
            )

    # torch.nn.functional.rms_norm without a weight, eps 1e-5 (the engine
    # kernel's fixed epsilon)
    if opname == "rms_norm":
        x = arg_tensor(0)
        weight = kwargs.get("weight", node.args[2] if len(node.args) > 2 else None)
        eps = kwargs.get("eps", node.args[3] if len(node.args) > 3 else None)
        if isinstance(weight, torch.fx.Node):
            weight = env_shapes.get(weight.name)
        # M5 fusion: rms_norm(linear(x, w)) -> fused_matmul_rms_norm(x, w)
        # (the linear's lm_head output is weight-major; no bias allowed so
        # the fused kernel's contract holds)
        src_name = node.args[0].name
        src_op = prev_ops.get(src_name)
        if (
            isinstance(src_op, _Op)
            and src_op.engine_op is None
            and src_op.composite
            and src_op.composite[0][0] == "lm_head"
            and len(src_op.composite) == 1
            and (weight is None)
            and (eps is None or eps == 1e-5)
        ):
            stats["native"] += 1
            stats["fused"] = stats.get("fused", 0) + 1
            prev_kind[name] = "native"
            return _Op(
                name=name,
                kind="native",
                engine_op="fused_matmul_rms_norm",
                arg_names=[src_op.composite[0][1][0], src_op.composite[0][1][1]],
                fx_node=node,
            )
        if (
            _is_float_tensor(x)
            and _shape_ok_2d(x)
            and (weight is None)
            and (eps is None or eps == 1e-5)
        ):
            stats["native"] += 1
            prev_kind[name] = "native"
            return _Op(
                name=name,
                kind="native",
                engine_op="rms_norm",
                arg_names=[node.args[0].name],
                fx_node=node,
            )

    # SwiGLU pattern: mul(silu(g), u) -> swiglu(g, u)
    if opname == "mul":
        a, b = arg_tensor(0), arg_tensor(1)
        silu_operand = None
        other = None
        for i in (0, 1):
            src = node.args[i]
            if (
                isinstance(src, torch.fx.Node)
                and prev_kind.get(src.name) == "silu"
            ):
                silu_operand = src
                other = node.args[1 - i]
        if (
            silu_operand is not None
            and _is_float_tensor(a)
            and _is_float_tensor(b)
            and _shape_ok_2d(a)
            and _shape_ok_2d(b)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and isinstance(other, torch.fx.Node)
        ):
            # swiglu takes the silu *input* and the other operand
            silu_input = silu_operand.args[0]
            if isinstance(silu_input, torch.fx.Node):
                stats["native"] += 1
                prev_kind[name] = "native"
                return _Op(
                    name=name,
                    kind="native",
                    engine_op="swiglu",
                    arg_names=[silu_input.name, other.name],
                    fx_node=node,
                )

    # embedding lookup
    if opname == "embedding":
        w = arg_tensor(0)
        idx = arg_tensor(1)
        if (
            _is_float_tensor(w)
            and _shape_ok_2d(w)
            and isinstance(idx, torch.Tensor)
            and idx.dim() == 1
            and idx.dtype in (torch.int32, torch.int64)
        ):
            stats["native"] += 1
            prev_kind[name] = "native"
            return _Op(
                name=name,
                kind="native",
                engine_op="embedding",
                arg_names=[node.args[1].name, node.args[0].name],
                fx_node=node,
            )

    # -- fallback ------------------------------------------------------------

    if strict:
        raise UnsupportedOpError(
            f"[infer_train] node '{name}' ({opname}) has no engine "
            f"implementation. M5 supports: mm/matmul/bmm, addmm/linear, "
            f"add, softmax(-1), rms_norm(no weight), mul(silu(g),u), "
            f"embedding, cond/while_loop; everything else runs as a torch "
            f"fallback unless strict=True."
        )

    stats["fallback"] += 1
    if opname == "silu":
        prev_kind[name] = "silu"  # enables the swiglu pattern match

    if node.op == "call_module":
        module = _resolve_attr(gm, str(target))
        return _Op(
            name=name,
            kind="fallback",
            fn=lambda *a, _m=module, **k: _m(*a, **k),
            fx_node=node,
        )

    if node.op == "call_method":
        # target is a method name string ("view", "transpose", ...);
        # CompiledInferTrain dispatches it on the first argument.
        return _Op(name=name, kind="fallback", fn=None, fx_node=node)

    return _Op(
        name=name,
        kind="fallback",
        fn=lambda *a, _t=target, **k: _t(*a, **k),
        fx_node=node,
    )


def translate_graph(
    gm: torch.fx.GraphModule,
    example_inputs: List[Any],
    strict: bool = False,
) -> Tuple[List[_Op], Dict[str, int]]:
    """Walk the FX graph and produce the execution plan.

    Returns (plan, stats).  Raises ControlFlowError/UnsupportedOpError for
    graphs M4 cannot handle, and a clear error for training-mode graphs.
    """
    # M4 is inference-only: parameters/inputs may still carry requires_grad
    # (normal even after model.eval()), so the compiled callable detaches
    # everything instead of rejecting the graph.  Autograd-aware execution
    # lands in M6+.
    if any(getattr(p, "requires_grad", False) for p in gm.parameters()) or any(
        isinstance(e, torch.Tensor) and e.requires_grad
        for e in example_inputs
    ):
        warnings.warn(
            "[infer_train] M4 executes inference only: inputs/parameters "
            "with requires_grad=True are detached (autograd support lands "
            "in M6+)."
        )

    # shape env: placeholder tensors + traced tensor metadata
    env_shapes: Dict[str, Any] = {}
    for node, example in zip(gm.graph.find_nodes(op="placeholder"), example_inputs):
        env_shapes[node.name] = example

    stats = {
        "nodes": 0,
        "inputs": 0,
        "consts": 0,
        "native": 0,
        "fallback": 0,
        "outputs": 0,
        "control_flow": 0,
        "static_cf": 0,
        "fused": 0,
    }
    prev_kind: Dict[str, str] = {}
    prev_ops: Dict[str, _Op] = {}
    plan: List[_Op] = []
    # nodes whose values are compile-time constants (for static CFG)
    const_names: set = set()
    const_values: Dict[str, Any] = {}
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            const_names.add(node.name)
            try:
                value = _resolve_attr(gm, str(node.target))
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu()
                const_values[node.name] = value
            except Exception:
                pass
        elif node.op == "call_function":
            if not any(isinstance(a, torch.fx.Node) for a in node.args) and not any(
                isinstance(v, torch.fx.Node) for v in node.kwargs.values()
            ):
                const_names.add(node.name)
                try:
                    const_values[node.name] = node.target(
                        *node.args, **node.kwargs
                    )
                except Exception:
                    pass

    for node in gm.graph.nodes:
        stats["nodes"] += 1
        op = _translate_node(
            node, gm, env_shapes, strict, stats, prev_kind,
            const_names, const_values, prev_ops,
        )
        plan.append(op)
        if op.kind == "native":
            prev_ops[node.name] = op
        # track the node's output shape for later translation decisions
        if op.kind in ("input", "const"):
            if op.kind == "const":
                env_shapes[node.name] = op.fn() if op.fn else None
        elif op.kind == "control_flow":
            # M5 CFG: dry-run the sub-plans in shape mode to learn the
            # node's output shape without executing engine kernels.
            try:
                spec = op.ctrl
                if isinstance(spec, _CondSpec):
                    operands = [env_shapes[n] for n in spec.operand_names]
                    chosen = spec.true_gm
                    if spec.static:
                        pred = spec.pred
                        truth = bool(
                            pred.item() if isinstance(pred, torch.Tensor) else pred
                        )
                        if not truth:
                            chosen = spec.false_gm
                    env_shapes[node.name] = _run_subplan_shapes(
                        chosen, operands, strict
                    )
                elif isinstance(spec, _WhileSpec):
                    # while_loop output shape == carried shape
                    env_shapes[node.name] = tuple(
                        env_shapes[n] for n in spec.carried_names
                    )
            except Exception:
                pass
        elif op.kind == "native":
            # infer the output shape statically where possible
            try:
                if op.engine_op == "matmul":
                    a = env_shapes[op.arg_names[0]]
                    b = env_shapes[op.arg_names[1]]
                    env_shapes[node.name] = torch.empty(
                        (a.shape[0], b.shape[1]), dtype=a.dtype
                    )
                elif op.engine_op in (
                    "lm_head",
                    "fused_matmul_rms_norm",
                ):
                    a = env_shapes[op.arg_names[0]]
                    w = env_shapes[op.arg_names[1]]
                    env_shapes[node.name] = torch.empty(
                        (a.shape[0], w.shape[0]), dtype=a.dtype
                    )
                elif op.engine_op in ("add", "rms_norm", "softmax", "swiglu"):
                    a = env_shapes[op.arg_names[0]]
                    env_shapes[node.name] = torch.empty(a.shape, dtype=a.dtype)
                elif op.engine_op == "add_bias":
                    a = env_shapes[op.arg_names[0]]
                    env_shapes[node.name] = torch.empty(a.shape, dtype=a.dtype)
                elif op.engine_op == "embedding":
                    idx = env_shapes[op.arg_names[0]]
                    w = env_shapes[op.arg_names[1]]
                    env_shapes[node.name] = torch.empty(
                        (idx.shape[0], w.shape[1]), dtype=w.dtype
                    )
                elif op.composite:
                    # run the composite against empty tensors to learn shape
                    x = env_shapes[op.composite[0][1][0]]
                    w = env_shapes[op.composite[0][1][1]]
                    if op.composite[0][0] in (
                        "lm_head",
                        "fused_matmul_add_bias",
                        "fused_swiglu_matmul",
                        "fused_matmul_add",
                    ):
                        out_shape = (x.shape[0], w.shape[0])
                    else:
                        out_shape = (x.shape[0], w.shape[1])
                    env_shapes[node.name] = torch.empty(out_shape, dtype=x.dtype)
            except Exception:
                pass
        elif op.kind == "fallback":
            # run the torch fallback on the example tensors to learn its
            # output shape (cheap: structural/elementwise ops only)
            try:
                fnode = op.fx_node
                fargs, fkwargs = _arg_values(fnode, env_shapes)
                if op.fn is not None:
                    env_shapes[node.name] = op.fn(*fargs, **fkwargs)
                elif fnode.op == "call_method":
                    method = str(fnode.target)
                    env_shapes[node.name] = getattr(fargs[0], method)(
                        *fargs[1:], **fkwargs
                    )
                else:
                    env_shapes[node.name] = fnode.target(*fargs, **fkwargs)
            except Exception:
                pass
    return plan, stats


def _run_native(
    op: _Op, env: Dict[str, Any], shape_mode: bool = False
) -> torch.Tensor:
    """Execute one native engine op (or composite) on torch tensors.

    In ``shape_mode`` the engine kernel is skipped and only the output
    shape is computed from the input shapes (used for compile-time shape
    inference of control-flow sub-plans).
    """
    def run_one(engine_op: str, names: List[str]) -> torch.Tensor:
        tensors = [env[n] for n in names]
        if shape_mode:
            # shape-only execution: reuse the static shape rules
            if engine_op in ("matmul",):
                return torch.empty(
                    (tensors[0].shape[0], tensors[1].shape[1]),
                    dtype=tensors[0].dtype,
                )
            if engine_op in (
                "lm_head",
                "fused_matmul_add_bias",
                "fused_matmul_add",
                "fused_matmul_rms_norm",
                "fused_swiglu_matmul",
            ):
                return torch.empty(
                    (tensors[0].shape[0], tensors[1].shape[0]),
                    dtype=tensors[0].dtype,
                )
            if engine_op in ("add", "rms_norm", "softmax", "swiglu",
                             "add_bias"):
                return torch.empty(tensors[0].shape, dtype=tensors[0].dtype)
            if engine_op == "embedding":
                return torch.empty(
                    (tensors[0].shape[0], tensors[1].shape[1]),
                    dtype=tensors[1].dtype,
                )
            return torch.empty(tensors[0].shape, dtype=tensors[0].dtype)
        handles: List[EngineTensor] = []
        try:
            for t in tensors:
                t = t.detach().contiguous()
                if t.dtype not in _TORCH_DTYPE_TO_CODE:
                    raise UnsupportedOpError(
                        f"[infer_train] dtype {t.dtype} not supported by "
                        f"the engine (use float32/float16/int32)."
                    )
                code = _TORCH_DTYPE_TO_CODE[t.dtype]
                if t.dtype == torch.int64:
                    t = t.to(torch.int32)
                handles.append(
                    EngineTensor.from_buffer(
                        code, tuple(t.shape), t.data_ptr()
                    )
                )
            snap = run_op(engine_op, handles)[0]
        except EngineError as e:
            raise UnsupportedOpError(
                f"[infer_train] native op '{engine_op}' failed for node "
                f"'{op.name}': {e}"
            ) from e
        finally:
            for h in handles:
                h.free()
        out_dtype = _CODE_TO_TORCH_DTYPE[snap.dtype]
        return torch.frombuffer(bytearray(snap.data), dtype=out_dtype).reshape(
            snap.shape
        ).clone()

    if op.composite:
        result = run_one(op.composite[0][0], op.composite[0][1])
        env["__out0__"] = result
        for engine_op, names in op.composite[1:]:
            mapped = ["__out0__" if n == "__out0__" else n for n in names]
            result = run_one(engine_op, mapped)
            env["__out0__"] = result
        return result
    return run_one(op.engine_op, op.arg_names)


def _materialize_output(a: Any, env: Dict[str, Any]) -> Any:
    """Resolve an FX output spec to concrete values (unwraps containers)."""
    if isinstance(a, torch.fx.Node):
        return env[a.name]
    if isinstance(a, (tuple, list)) or type(a).__name__ == "immutable_list":
        items = [_materialize_output(x, env) for x in a]
        if len(items) == 1:
            return items[0]
        return tuple(items)
    return a


class CompiledInferTrain:
    """The callable returned by :func:`infer_train_backend`."""

    def __init__(
        self,
        plan: List[_Op],
        gm: torch.fx.GraphModule,
        stats: Dict[str, int],
        original_device: str = "cpu",
        shape_mode: bool = False,
    ):
        self._plan = plan
        self._gm = gm
        self.stats = stats
        self._original_device = original_device
        self._shape_mode = shape_mode
        self.native_ops: List[str] = [
            op.name for op in plan if op.kind == "native"
        ]
        self.fallback_ops: List[str] = [
            op.name for op in plan if op.kind == "fallback"
        ]
        self.control_flow_ops: List[str] = [
            op.name for op in plan if op.kind == "control_flow"
        ]

    def __call__(self, *args: Any) -> Any:
        env: Dict[str, Any] = {}
        inputs = list(args)

        # task 5: the M4 backend runs on CPU; accept CUDA inputs by moving
        # them here (and back on output) with a warning.
        orig_devices = [a.device if isinstance(a, torch.Tensor) else None for a in inputs]
        moved = False
        for i, a in enumerate(inputs):
            if isinstance(a, torch.Tensor) and a.device.type == "cuda":
                if not moved:
                    warnings.warn(
                        "[infer_train] inputs are on CUDA but the M4 "
                        "backend runs on CPU; moving inputs to CPU and the "
                        "outputs back afterwards."
                    )
                    moved = True
                inputs[i] = a.cpu()
            if isinstance(a, torch.Tensor) and a.requires_grad:
                inputs[i] = a.detach()

        for op, value in zip(self._plan, inputs):
            if op.kind == "input":
                env[op.name] = value

        outputs: Any = None
        for op in self._plan:
            if op.kind == "input":
                continue
            if op.kind == "const":
                env[op.name] = op.fn()
            elif op.kind == "native":
                env[op.name] = _run_native(op, env, shape_mode=self._shape_mode)
            elif op.kind == "control_flow":
                if isinstance(op.ctrl, _CondSpec):
                    env[op.name] = _cond_runner(
                        op.ctrl, env, shape_mode=self._shape_mode
                    )
                elif isinstance(op.ctrl, _WhileSpec):
                    env[op.name] = _while_runner(
                        op.ctrl, env, shape_mode=self._shape_mode
                    )
                else:
                    raise ControlFlowError(
                        f"[infer_train] unknown control-flow spec for "
                        f"node '{op.name}'"
                    )
            elif op.kind == "fallback":
                node = op.fx_node
                args, kwargs = _arg_values(node, env)
                if op.fn is not None:
                    env[op.name] = op.fn(*args, **kwargs)
                elif node.op == "call_method":
                    method = str(node.target)
                    env[op.name] = getattr(args[0], method)(
                        *args[1:], **kwargs
                    )
                else:
                    env[op.name] = node.target(*args, **kwargs)
            elif op.kind == "output":
                node = op.fx_node
                # The backend must return what the GraphModule would.
                # torch.compile graphs put a *tuple* of traced return values
                # in the output node (gm returns that tuple; torch.compile
                # unwraps a 1-tuple on the way out), while dynamo.export
                # graphs put a *list* and gm returns the bare element.
                spec = node.args[0]
                if isinstance(spec, tuple):
                    outputs = tuple(_materialize_output(a, env) for a in spec)
                elif isinstance(spec, list) or type(spec).__name__ == "immutable_list":
                    items = [_materialize_output(a, env) for a in spec]
                    outputs = items[0] if len(items) == 1 else tuple(items)
                else:
                    outputs = _materialize_output(spec, env)

        # move outputs back to the original device (task 5)
        if moved and outputs is not None:
            target = orig_devices[0] if orig_devices else None
            if target is not None:
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.to(target)
                else:
                    outputs = tuple(
                        o.to(target) if isinstance(o, torch.Tensor) else o
                        for o in outputs
                    )
        return outputs

    def summary(self) -> str:
        stats = self.stats or {}
        cf = stats.get("control_flow", 0)
        scf = stats.get("static_cf", 0)
        lines = [
            f"[infer_train] compiled graph: {stats.get('nodes', 0)} nodes "
            f"({stats.get('native', 0)} native engine ops, "
            f"{stats.get('fallback', 0)} torch fallback, "
            f"{cf} control flow ({scf} static), "
            f"{stats.get('inputs', 0)} inputs, {stats.get('consts', 0)} "
            f"constants)",
        ]
        for n in self.native_ops:
            lines.append(f"  native: {n}")
        if self.control_flow_ops:
            lines.append(
                f"  control flow ({len(self.control_flow_ops)}): "
                + ", ".join(self.control_flow_ops)
            )
        if self.fallback_ops:
            lines.append(
                f"  fallback ({len(self.fallback_ops)}): "
                + ", ".join(self.fallback_ops[:8])
                + ("..." if len(self.fallback_ops) > 8 else "")
            )
        return "\n".join(lines)


def infer_train_backend(
    gm: torch.fx.GraphModule,
    example_inputs: List[Any],
    *,
    strict: bool = False,
    verbose: bool = False,
) -> Callable[..., Any]:
    """torch.compile backend entry point.

    Captured FX graph + example inputs -> callable executing the model on
    the infer_train engine (with a torch fallback for structural ops).
    """
    plan, stats = translate_graph(gm, example_inputs, strict=strict)
    compiled = CompiledInferTrain(plan, gm, stats)
    global last_compiled
    last_compiled = compiled
    if verbose:
        print(compiled.summary())
    return compiled
