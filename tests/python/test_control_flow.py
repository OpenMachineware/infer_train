"""M5 control-flow (CFG) tests.

Covers:
  1. data-dependent ``torch.cond`` (both branches) - outputs match eager;
  2. compile-time-constant ``torch.cond`` - the dead branch is dropped;
  3. ``torch.while_loop`` with a runtime trip count - outputs match eager;
  4. fully-static ``torch.while_loop`` - unrolled at compile time;
  5. ``torch.compile(model, backend="infer_train")`` with cond + while_loop
     in one model, with native engine ops inside the branches;
  6. unsupported higher-order ops (map/scan) still raise ControlFlowError.
"""

from __future__ import annotations

import pytest
import torch

import torch._dynamo as dynamo

from infer_train.backend import (
    ControlFlowError,
    infer_train_backend,
    translate_graph,
)


def _export(model, *args):
    gm, _guards = dynamo.export(model)(*args)
    return gm


# ---------------------------------------------------------------------------
# 1. data-dependent cond
# ---------------------------------------------------------------------------


def test_torch_cond_data_dependent_branches_match_eager():
    def model(x):
        return torch.cond(
            x.sum() > 0,
            lambda: x * 2 + 1,
            lambda: x - 1,
            operands=(),
        )

    x = torch.randn(4, 6)
    gm = _export(model, x)
    compiled = infer_train_backend(gm, [x])
    with torch.no_grad():
        assert torch.allclose(compiled(x), model(x))
        assert torch.allclose(compiled(-x), model(-x))

    stats = compiled.stats
    assert stats["control_flow"] == 1
    assert stats["static_cf"] == 0


def test_torch_cond_with_native_ops_in_branches():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(8, 8) * 0.1)
            self.b = torch.nn.Parameter(torch.randn(8) * 0.1)

        def forward(self, x):
            return torch.cond(
                x.sum() > 0,
                lambda: torch.nn.functional.linear(x, self.w, self.b),
                lambda: torch.nn.functional.linear(-x, self.w, self.b),
                operands=(),
            )

    torch.manual_seed(0)
    m = M().eval()
    x = torch.randn(3, 8)
    gm = _export(m, x)
    compiled = infer_train_backend(gm, [x])
    with torch.no_grad():
        assert torch.allclose(compiled(x), m(x), atol=1e-4)
        assert torch.allclose(compiled(-x), m(-x), atol=1e-4)

    # the taken branch must have gone through the engine
    op = [o for o in compiled._plan if o.kind == "control_flow"][0]
    true_plan = op.ctrl._plans[0]
    assert "linear" in true_plan.native_ops


# ---------------------------------------------------------------------------
# 2. static (compile-time) cond
# ---------------------------------------------------------------------------


def test_torch_cond_constant_predicate_resolves_statically():
    def model(x):
        return torch.cond(True, lambda: x * 2, lambda: x + 1, operands=())

    x = torch.randn(4, 6)
    gm = _export(model, x)
    compiled = infer_train_backend(gm, [x])
    with torch.no_grad():
        assert torch.allclose(compiled(x), x * 2)


# ---------------------------------------------------------------------------
# 3. while_loop with runtime trip count
# ---------------------------------------------------------------------------


def test_torch_while_loop_runtime_count_matches_eager():
    def model(x, n):
        def cond(i, acc):
            return i < n

        def body(i, acc):
            return i + 1, acc + x

        return torch.while_loop(cond, body, (torch.tensor(0), x))[1]

    x = torch.randn(2, 3)
    n = torch.tensor(7)
    gm = _export(model, x, n)
    compiled = infer_train_backend(gm, [x, n])
    with torch.no_grad():
        got = compiled(x, n)
    want = model(x, n)
    assert torch.allclose(got, want)
    assert got.shape == (2, 3)

    stats = compiled.stats
    assert stats["control_flow"] == 1
    assert stats["static_cf"] == 0


# ---------------------------------------------------------------------------
# 4. static while_loop: unrolled at compile time
# ---------------------------------------------------------------------------


def test_torch_while_loop_all_const_unrolls_at_compile_time():
    def model():
        def cond(i, acc):
            return i < 5

        def body(i, acc):
            return i + 1, acc + i

        return torch.while_loop(
            cond, body, (torch.tensor(0), torch.tensor(0))
        )[1]

    gm = _export(model)
    compiled = infer_train_backend(gm, [])
    out = compiled()
    assert out.item() == sum(range(5))
    # the loop node became a compile-time constant
    assert compiled.stats["static_cf"] == 1


# ---------------------------------------------------------------------------
# 5. torch.compile end-to-end: cond + while_loop in one model
# ---------------------------------------------------------------------------


def test_torch_compile_control_flow_matches_eager():
    import infer_train

    class CfgModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(8, 8) * 0.1)
            self.b = torch.nn.Parameter(torch.randn(8) * 0.1)

        def forward(self, x, n):
            y = torch.cond(
                x.sum() > 0,
                lambda: torch.nn.functional.linear(x, self.w, self.b),
                lambda: torch.nn.functional.linear(-x, self.w, self.b),
                operands=(),
            )

            def cond(i, acc):
                return i < n

            def body(i, acc):
                return i + 1, acc + y

            return torch.while_loop(cond, body, (torch.tensor(0), y))[1]

    torch.manual_seed(0)
    model = CfgModel().eval()
    x = torch.randn(3, 8)
    n = torch.tensor(3)
    compiled = torch.compile(model, backend="infer_train")
    with torch.no_grad():
        want = model(x, n)
        got = compiled(x, n)
        want_neg = model(-x, n)
        got_neg = compiled(-x, n)
    assert torch.allclose(got, want, atol=1e-4)
    assert torch.allclose(got_neg, want_neg, atol=1e-4)


def test_torch_compile_while_loop_fixed_steps():
    import infer_train

    def model(x):
        def cond(i, acc):
            return i < 4

        def body(i, acc):
            return i + 1, acc * x

        return torch.while_loop(
            cond, body, (torch.tensor(0), torch.ones_like(x))
        )[1]

    x = torch.randn(2, 4)
    compiled = torch.compile(model, backend="infer_train")
    with torch.no_grad():
        got = compiled(x)
    assert torch.allclose(got, x**4, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. unsupported higher-order ops still raise
# ---------------------------------------------------------------------------


def test_map_and_scan_raise_control_flow_error():
    import torch.fx as fx

    for opname in ("map_impl", "scan"):
        graph = fx.Graph()
        x = graph.placeholder("x")
        with graph.inserting_after(x):
            call = graph.call_function(
                torch.ops.higher_order.__getattr__(opname),
                (x,),
                kwargs={},
            )
            graph.output(call)
        gm = fx.GraphModule(torch.nn.Module(), graph)
        with pytest.raises(ControlFlowError, match="not supported yet in M5"):
            infer_train_backend(gm, [torch.randn(2, 2)])


def test_malformed_cond_node_raises_control_flow_error():
    import torch.fx as fx

    graph = fx.Graph()
    x = graph.placeholder("x")
    with graph.inserting_after(x):
        cond = graph.call_function(
            torch.ops.higher_order.cond,
            (torch.tensor(True),),
            kwargs={},
        )
        graph.output(cond)
    gm = fx.GraphModule(torch.nn.Module(), graph)
    with pytest.raises(ControlFlowError, match="unsupported argument layout"):
        infer_train_backend(gm, [torch.randn(2, 2)])
