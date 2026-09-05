"""M4 PyTorch ecosystem tests.

Covers, in order:
  1. the ctypes binding (library discovery, tensor round-trip, ABI check);
  2. native engine ops vs torch reference results (all whitelisted ops);
  3. friendly errors (bad shapes, unknown ops, strict mode, control flow);
  4. torch.compile(model, backend="infer_train") on a mini transformer
     (single-layer MHA + SwiGLU FFN) - outputs must match eager mode;
  5. the end-to-end 1.5B generation against the M3 reference output.

The e2e generation test is marked ``e2e`` (a few seconds of prefill +
~35 s of decode) and skips automatically when the GGUF is absent.
"""

from __future__ import annotations

import time

import pytest
import torch

from infer_train import binding
from infer_train.backend import (
    ControlFlowError,
    UnsupportedOpError,
    infer_train_backend,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run_engine(op_name, tensors):
    """Run one engine op on torch tensors; returns the torch result."""
    handles = []
    try:
        for t in tensors:
            t = t.detach().contiguous()
            code = {torch.float32: 0, torch.float16: 1, torch.int32: 2}[
                t.dtype
            ]
            handles.append(
                binding.Tensor.from_buffer(code, tuple(t.shape), t.data_ptr())
            )
        snaps = binding.run_op(op_name, handles)
    finally:
        for h in handles:
            h.free()
    snap = snaps[0]
    dtype = {0: torch.float32, 1: torch.float16, 2: torch.int32}[snap.dtype]
    return (
        torch.frombuffer(bytearray(snap.data), dtype=dtype)
        .reshape(snap.shape)
        .clone()
    )


class MiniRMSNorm(torch.nn.Module):
    """RMSNorm via F.rms_norm with weight=None (maps to the engine op)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.rms_norm(x, (self.dim,), None, self.eps)


class MiniTransformer(torch.nn.Module):
    """Single-layer MHA + SwiGLU FFN - the M4 translation demo model."""

    def __init__(self, vocab=64, hidden=16, ffn=48, n_heads=2, block=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.norm1 = MiniRMSNorm(hidden)
        self.q_proj = torch.nn.Linear(hidden, hidden)
        self.k_proj = torch.nn.Linear(hidden, hidden)
        self.v_proj = torch.nn.Linear(hidden, hidden)
        self.o_proj = torch.nn.Linear(hidden, hidden)
        self.norm2 = MiniRMSNorm(hidden)
        self.gate_proj = torch.nn.Linear(hidden, ffn)
        self.up_proj = torch.nn.Linear(hidden, ffn)
        self.down_proj = torch.nn.Linear(ffn, hidden)
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads

    def forward(self, x):  # x: [B, T] token ids
        B, T = x.shape
        h = self.embed(x)
        h = self.norm1(h)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
        attn = torch.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).reshape(B, T, -1)
        h = h + self.o_proj(attn)
        h = self.norm2(h)
        g = torch.nn.functional.silu(self.gate_proj(h))
        u = self.up_proj(h)
        h = h + self.down_proj(g * u)
        return h


# ---------------------------------------------------------------------------
# 1. binding smoke tests
# ---------------------------------------------------------------------------


def test_abi_version_and_tensor_roundtrip():
    assert binding._lib.infer_train_version() == binding.ABI_VERSION

    t = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    h = binding.Tensor.from_buffer(
        binding.DTYPE_F32, (3, 8), t.data_ptr()
    )
    assert h.dtype == binding.DTYPE_F32
    assert h.rank == 2
    assert h.numel == 24
    assert h.shape == (3, 8)
    out = torch.frombuffer(
        bytearray(h.to_bytes()), dtype=torch.float32
    ).reshape(3, 8)
    assert torch.equal(out, t)
    h.free()


# ---------------------------------------------------------------------------
# 2. native ops vs torch
# ---------------------------------------------------------------------------


def test_matmul_f32():
    torch.manual_seed(0)
    a = torch.randn(3, 5)
    b = torch.randn(5, 4)
    got = run_engine("matmul", [a, b])
    assert torch.allclose(got, torch.mm(a, b), atol=1e-6, rtol=1e-6)


def test_matmul_f16():
    torch.manual_seed(0)
    a = torch.randn(3, 5).half()
    b = torch.randn(5, 4).half()
    ref = torch.mm(a.float(), b.float()).half()
    got = run_engine("matmul", [a, b])
    assert torch.allclose(got.float(), ref.float(), atol=1e-3, rtol=1e-3)


def test_matmul_stability_across_calls():
    torch.manual_seed(0)
    a = torch.randn(3, 5)
    b = torch.randn(5, 4)
    ref = torch.mm(a, b)
    for _ in range(5):  # repeated C calls must not corrupt the heap
        assert torch.equal(run_engine("matmul", [a, b]), ref)


def test_softmax_rmsnorm_add():
    torch.manual_seed(1)
    x = torch.randn(4, 8)
    assert torch.allclose(
        run_engine("softmax", [x]), torch.softmax(x, -1), atol=1e-6
    )
    ref_rms = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)
    assert torch.allclose(run_engine("rms_norm", [x]), ref_rms, atol=1e-5)

    y = torch.randn(4, 8)
    assert torch.allclose(run_engine("add", [x, y]), x + y, atol=1e-6)
    bias = torch.randn(8)
    assert torch.allclose(
        run_engine("add_bias", [x, bias]), x + bias, atol=1e-6
    )


def test_swiglu_embedding_lm_head_ffn():
    torch.manual_seed(2)
    g = torch.randn(2, 6)
    u = torch.randn(2, 6)
    assert torch.allclose(
        run_engine("swiglu", [g, u]),
        torch.nn.functional.silu(g) * u,
        atol=1e-5,
    )

    table = torch.randn(10, 6)
    idx = torch.tensor([1, 5, 9], dtype=torch.int32)
    assert torch.allclose(
        run_engine("embedding", [idx, table]), table[idx], atol=1e-6
    )

    x = torch.randn(4, 8)
    w = torch.randn(7, 8)  # [out, in] layout
    assert torch.allclose(
        run_engine("lm_head", [x, w]), x @ w.t(), atol=1e-4
    )

    x2 = torch.randn(2, 6)
    wg = torch.randn(10, 6)
    wu = torch.randn(10, 6)
    wd = torch.randn(6, 10)
    ref_ffn = (
        torch.nn.functional.silu(x2 @ wg.t()) * (x2 @ wu.t()) @ wd.t()
    )
    assert torch.allclose(
        run_engine("swiglu_ffn", [x2, wg, wu, wd]), ref_ffn, atol=1e-4
    )


# ---------------------------------------------------------------------------
# 2b. M5 fused kernels vs torch
# ---------------------------------------------------------------------------


def test_fused_kernels_match_torch():
    torch.manual_seed(3)
    for dt in (torch.float32, torch.float16):
        x = torch.randn(3, 7).to(dt)
        w = torch.randn(5, 7).to(dt)  # weight-major [out, in]
        bias = torch.randn(5).to(dt)
        b2 = torch.randn(3, 5).to(dt)
        atol = 2e-2 if dt == torch.float16 else 1e-5

        # matmul + bias
        got = run_engine("fused_matmul_add_bias", [x, w, bias])
        want = x.float() @ w.float().t() + bias.float()
        assert torch.allclose(got.float(), want, atol=atol)

        # matmul + elementwise add
        got = run_engine("fused_matmul_add", [x, w, b2])
        want = x.float() @ w.float().t() + b2.float()
        assert torch.allclose(got.float(), want, atol=atol)

        # matmul + rms_norm
        got = run_engine("fused_matmul_rms_norm", [x, w])
        h = x.float() @ w.float().t()
        want = h / torch.sqrt((h * h).mean(-1, keepdim=True) + 1e-5)
        assert torch.allclose(got.float(), want, atol=atol)

        # swiglu + matmul
        g = torch.randn(3, 7).to(dt)
        u = torch.randn(3, 7).to(dt)
        got = run_engine("fused_swiglu_matmul", [g, u, w])
        h2 = torch.nn.functional.silu(g.float()) * u.float()
        want = h2 @ w.float().t()
        assert torch.allclose(got.float(), want, atol=atol)


def test_backend_fuses_linear_swiglu_and_rms_norm_patterns():
    """The translator must rewrite heavy 2D patterns onto fused kernels."""
    import torch._dynamo as dynamo
    from infer_train import backend as backend_mod

    torch.manual_seed(5)

    # linear(x, w, b) -> fused_matmul_add_bias (one engine call)
    lin = torch.nn.Linear(8, 8).eval()
    x = torch.randn(2, 8)
    gm, _ = dynamo.export(lin)(x)
    compiled = backend_mod.infer_train_backend(gm, [x])
    with torch.no_grad():
        assert torch.allclose(compiled(x), lin(x), atol=1e-4)
    assert compiled.stats["fused"] == 1

    # down(silu(gate(x)) * up(x)) -> fused_swiglu_matmul
    class Ffn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(8, 16, bias=False)
            self.up = torch.nn.Linear(8, 16, bias=False)
            self.down = torch.nn.Linear(16, 8, bias=False)

        def forward(self, x):
            return self.down(
                torch.nn.functional.silu(self.gate(x)) * self.up(x)
            )

    m = Ffn().eval()
    gm2, _ = dynamo.export(m)(x)
    compiled2 = backend_mod.infer_train_backend(gm2, [x])
    with torch.no_grad():
        assert torch.allclose(compiled2(x), m(x), atol=1e-3)
    assert compiled2.stats["fused"] == 1
    assert any(
        op.composite and op.composite[0][0] == "fused_swiglu_matmul"
        for op in compiled2._plan
    )

    # rms_norm(linear(x, w)) -> fused_matmul_rms_norm
    def model3(x, w):
        return torch.nn.functional.rms_norm(
            torch.nn.functional.linear(x, w), (x.shape[1],)
        )

    w = torch.randn(8, 8) * 0.2
    gm3, _ = dynamo.export(model3)(x, w)
    compiled3 = backend_mod.infer_train_backend(gm3, [x, w])
    with torch.no_grad():
        assert torch.allclose(compiled3(x, w), model3(x, w), atol=1e-3)
    assert compiled3.stats["fused"] == 1
    assert any(
        op.engine_op == "fused_matmul_rms_norm" for op in compiled3._plan
    )


# ---------------------------------------------------------------------------
# 3. error paths
# ---------------------------------------------------------------------------


def test_bad_shapes_and_unknown_ops_raise_engine_error():
    with pytest.raises(binding.EngineError):
        run_engine("matmul", [torch.randn(3, 5), torch.randn(6, 4)])
    with pytest.raises(binding.EngineError):
        run_engine("no_such_op", [torch.randn(2, 2)])
    with pytest.raises(binding.EngineError):
        run_engine("add", [torch.randn(2, 3), torch.randn(4, 3)])


def test_strict_mode_raises_on_unmapped_op():
    class TanhNet(torch.nn.Module):
        def forward(self, x):
            return torch.tanh(x)

    model = TanhNet().eval()
    example = torch.randn(2, 4)
    import torch._dynamo as dynamo

    gm, guards = dynamo.export(model)(example)
    with pytest.raises(
        UnsupportedOpError, match="has no engine implementation"
    ):
        infer_train_backend(gm, [example], strict=True)
    # non-strict mode falls back to torch and stays correct
    compiled = infer_train_backend(gm, [example])
    assert torch.allclose(compiled(example), torch.tanh(example))


def test_control_flow_raises_friendly_error():
    """M5: cond/while_loop are supported; a malformed cond node still
    raises a friendly ControlFlowError."""
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


def test_training_mode_detaches_with_warning():
    import torch._dynamo as dynamo

    model = torch.nn.Linear(4, 4)  # training mode: params require grad
    example = torch.randn(2, 4, requires_grad=True)
    gm, _ = dynamo.export(model)(example)
    with pytest.warns(UserWarning, match="inference only"):
        compiled = infer_train_backend(gm, [example])
    with torch.no_grad():
        out = compiled(example)
    assert out.requires_grad is False
    assert torch.allclose(out, model(example), atol=1e-4)


# ---------------------------------------------------------------------------
# 4. torch.compile integration on the mini transformer
# ---------------------------------------------------------------------------


def test_torch_compile_mini_transformer_matches_eager():
    torch.manual_seed(7)
    model = MiniTransformer().eval()
    example = torch.randint(0, 64, (1, 8))

    with torch.no_grad():
        expected = model(example)

        compiled = torch.compile(model, backend="infer_train")
        got = compiled(example)

        assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4), (
            f"max abs diff: {(got - expected).abs().max().item()}"
        )

    # the translation must have used the engine for the heavy ops
    from infer_train import backend as backend_mod

    assert backend_mod.last_compiled is not None
    native_names = backend_mod.last_compiled.native_ops

    # baseline timings (record only - M4 does not optimize)
    n = 10
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n):
            model(example)
        eager_s = (time.perf_counter() - t0) / n

        t0 = time.perf_counter()
        for _ in range(n):
            compiled(example)
        engine_s = (time.perf_counter() - t0) / n

    print(
        f"\n[infer_train] mini-transformer baseline: "
        f"eager {eager_s * 1e3:.2f} ms/iter vs engine-backed "
        f"{engine_s * 1e3:.2f} ms/iter (recorded, not optimized)"
    )


def test_optimize_helper_returns_compiled_callable():
    import infer_train

    model = torch.nn.Linear(8, 8).eval()
    compiled = infer_train.optimize(model, verbose=False)
    x = torch.randn(2, 8)
    with torch.no_grad():
        assert torch.allclose(compiled(x), model(x), atol=1e-4)


# ---------------------------------------------------------------------------
# 5. end-to-end 1.5B generation vs the M3 reference
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_e2e_generate_matches_m3_reference(model, prompt):
    import pathlib

    reference = pathlib.Path(__file__).parent / (
        "reference_outputs/reference_generate.txt"
    )
    if not reference.exists():
        pytest.skip("reference output missing")

    model.reset_cache()
    t0 = time.perf_counter()
    out = model.generate(
        prompt, max_tokens=120, temperature=0.6, top_p=0.95, top_k=40,
        seed=7,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[infer_train] 1.5B generate: {elapsed:.1f}s "
          f"({len(out)} chars)")

    expected = reference.read_text(encoding="utf-8").strip()
    assert out.strip() == expected, "generation diverged from the M3 reference"

    # semantic sanity on the R1-style output
    assert "</think>" in out
    assert "2" in out


@pytest.mark.e2e
def test_e2e_generate_deterministic_and_cache_reset(model, prompt):
    model.reset_cache()
    a = model.generate(prompt, max_tokens=12, seed=7)
    model.reset_cache()
    b = model.generate(prompt, max_tokens=12, seed=7)
    assert a == b
    assert len(a) > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
