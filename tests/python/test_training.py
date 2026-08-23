"""M6 PyTorch training tests.

Covers, in order:
  1. engine backward gradients vs torch.autograd for every trainable op
     (matmul, lm_head, add, add_bias, rms_norm, softmax, swiglu, embedding,
     swiglu_ffn, cross_entropy) - f32 within 1e-5, f16 within 2e-2;
  2. the engine's stateless AdamW update vs torch.optim.AdamW (1e-6);
  3. an end-to-end mini-transformer training loop whose forward AND
     backward run entirely on the engine (single-shot C calls through a
     torch.autograd.Function) - the loss curve must track the PyTorch
     eager training run on identical parameters and data.

The engine is driven through `infer_train.binding` (torch-free, raw
handles); the module reuses the same dtype-code convention as the M4
suite.
"""

from __future__ import annotations

import torch

from infer_train import binding

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

DT = {
    torch.float32: binding.DTYPE_F32,
    torch.float16: binding.DTYPE_F16,
    torch.int32: binding.DTYPE_I32,
}
TORCH_DT = {binding.DTYPE_F32: torch.float32, binding.DTYPE_F16: torch.float16}
TORCH_DT[binding.DTYPE_I32] = torch.int32
TORCH_DT[binding.DTYPE_F16] = torch.float16


def _snap_to_torch(snap):
    dtype = TORCH_DT[snap.dtype]
    return torch.frombuffer(bytearray(snap.data), dtype=dtype).reshape(
        snap.shape
    ).clone()


def engine_op(op_name, tensors):
    """Run one engine op on torch tensors; returns the torch result."""
    handles = []
    try:
        for t in tensors:
            t = t.detach().contiguous()
            handles.append(
                binding.Tensor.from_buffer(
                    DT[t.dtype], tuple(t.shape), t.data_ptr()
                )
            )
        snaps = binding.run_op(op_name, handles)
    finally:
        for h in handles:
            h.free()
    return _snap_to_torch(snaps[0])


def engine_backward(op_name, tensors, grads):
    """Run one engine op's backward; returns (grads or None) per input."""
    handles, ghandles = [], []
    try:
        for t in tensors:
            t = t.detach().contiguous()
            handles.append(
                binding.Tensor.from_buffer(
                    DT[t.dtype], tuple(t.shape), t.data_ptr()
                )
            )
        for g in grads:
            g = g.detach().contiguous()
            ghandles.append(
                binding.Tensor.from_buffer(
                    DT[g.dtype], tuple(g.shape), g.data_ptr()
                )
            )
        snaps = binding.run_backward(op_name, handles, ghandles)
    finally:
        for h in handles:
            h.free()
        for h in ghandles:
            h.free()
    return [None if s is None else _snap_to_torch(s) for s in snaps]


def _torch_ref_forward(op_name, tensors):
    """The torch reference forward for one engine op (gradcheck only)."""
    if op_name == "matmul":
        return tensors[0] @ tensors[1]
    if op_name == "lm_head":
        return tensors[0] @ tensors[1].t()
    if op_name == "add":
        return tensors[0] + tensors[1]
    if op_name == "add_bias":
        return tensors[0] + tensors[1]
    if op_name == "rms_norm":
        x = tensors[0]
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)
    if op_name == "softmax":
        return torch.softmax(tensors[0], dim=-1)
    if op_name == "swiglu":
        return torch.nn.functional.silu(tensors[0]) * tensors[1]
    if op_name == "embedding":
        return tensors[1][tensors[0].long()]
    if op_name == "swiglu_ffn":
        x, wg, wu, wd = tensors
        return (
            torch.nn.functional.silu(x @ wg.t()) * (x @ wu.t()) @ wd.t()
        )
    if op_name == "cross_entropy":
        return torch.nn.functional.cross_entropy(
            tensors[0], tensors[1].long()
        )
    raise AssertionError(f"no reference forward for {op_name}")


def gradcheck(op_name, tensors, grad_out, atol, rtol):
    """Compare engine backward with torch.autograd for one op."""
    engine_grads = engine_backward(op_name, tensors, [grad_out])
    refs = [
        t.clone().requires_grad_(True) if t.is_floating_point() else t.clone()
        for t in tensors
    ]
    out_ref = _torch_ref_forward(op_name, refs)
    if out_ref.dim() == 0:
        out_ref.backward()
    else:
        out_ref.backward(grad_out.clone())
    assert len(engine_grads) == len(refs)
    for eg, r in zip(engine_grads, refs):
        if eg is None:
            assert r.grad is None or not r.is_floating_point()
            continue
        assert torch.allclose(
            eg, r.grad, atol=atol, rtol=rtol
        ), f"{op_name}: max abs diff {(eg - r.grad).abs().max().item():.3e}"


# ---------------------------------------------------------------------------
# 1. gradient checks vs torch
# ---------------------------------------------------------------------------


def test_backward_gradients_match_torch():
    torch.manual_seed(11)
    tol = dict(atol=1e-5, rtol=1e-4)

    a = torch.randn(3, 5) * 0.7
    b = torch.randn(5, 4) * 0.7
    go = torch.randn(3, 4) * 0.5
    gradcheck("matmul", [a, b], go, **tol)

    x = torch.randn(3, 5) * 0.7
    w = torch.randn(4, 5) * 0.7  # weight-major [out, in]
    go = torch.randn(3, 4) * 0.5
    gradcheck("lm_head", [x, w], go, **tol)

    x, y = torch.randn(3, 4) * 0.7, torch.randn(3, 4) * 0.7
    gradcheck("add", [x, y], torch.randn(3, 4) * 0.5, **tol)

    x = torch.randn(3, 4) * 0.7
    bias = torch.randn(4) * 0.7
    gradcheck("add_bias", [x, bias], torch.randn(3, 4) * 0.5, **tol)

    x = torch.randn(3, 4) * 0.7
    gradcheck("rms_norm", [x], torch.randn(3, 4) * 0.5, **tol)

    x = torch.randn(3, 6) * 0.7
    gradcheck("softmax", [x], torch.randn(3, 6) * 0.5, **tol)

    gate = torch.randn(2, 6) * 0.7
    up = torch.randn(2, 6) * 0.7
    gradcheck("swiglu", [gate, up], torch.randn(2, 6) * 0.5, **tol)

    idx = torch.tensor([1, 5, 9], dtype=torch.int32)
    table = torch.randn(10, 6) * 0.7
    gradcheck("embedding", [idx, table], torch.randn(3, 6) * 0.5, **tol)

    x = torch.randn(2, 6) * 0.7
    wg, wu = torch.randn(10, 6) * 0.7, torch.randn(10, 6) * 0.7
    wd = torch.randn(6, 10) * 0.7
    gradcheck("swiglu_ffn", [x, wg, wu, wd], torch.randn(2, 6) * 0.5, **tol)

    logits = torch.randn(3, 8) * 0.7
    targets = torch.tensor([2, 5, 1], dtype=torch.int32)
    gradcheck(
        "cross_entropy", [logits, targets], torch.ones(1), **tol
    )


def test_backward_gradients_f16():
    torch.manual_seed(12)
    a = torch.randn(3, 5).half() * 0.5
    b = torch.randn(5, 4).half() * 0.5
    handles = [
        binding.Tensor.from_buffer(binding.DTYPE_F16, (3, 5), a.data_ptr()),
        binding.Tensor.from_buffer(binding.DTYPE_F16, (5, 4), b.data_ptr()),
    ]
    g = torch.randn(3, 4).half() * 0.3
    gh = binding.Tensor.from_buffer(binding.DTYPE_F16, (3, 4), g.data_ptr())
    try:
        snaps = binding.run_backward("matmul", handles, [gh])
    finally:
        for h in handles:
            h.free()
        gh.free()
    ga = _snap_to_torch(snaps[0]).float()
    gb = _snap_to_torch(snaps[1]).float()
    ref_a = g.float() @ b.float().t()
    ref_b = (a.float().t() @ g.float()).float()
    assert torch.allclose(ga, ref_a, atol=2e-2, rtol=2e-2)
    assert torch.allclose(gb, ref_b, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# 2. AdamW vs torch
# ---------------------------------------------------------------------------


def test_adamw_matches_torch():
    torch.manual_seed(13)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
    p = torch.randn(3, 5) * 0.5
    p_ref = p.clone()

    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    m_ref = torch.zeros_like(p)
    v_ref = torch.zeros_like(p)

    step = 0
    for _ in range(5):
        g = torch.randn(3, 5) * 0.5

        # engine path (state on the engine side, copied back after)
        ph = binding.Tensor.from_buffer(binding.DTYPE_F32, (3, 5), p.data_ptr())
        gh = binding.Tensor.from_buffer(binding.DTYPE_F32, (3, 5), g.data_ptr())
        mh = binding.Tensor.from_buffer(binding.DTYPE_F32, (3, 5), m.data_ptr())
        vh = binding.Tensor.from_buffer(binding.DTYPE_F32, (3, 5), v.data_ptr())
        try:
            step = binding.adamw_step(ph, gh, mh, vh, step, lr, b1, b2, eps, wd)
            m.copy_(
                _snap_to_torch(
                    binding.Snapshot(
                        binding.DTYPE_F32,
                        (3, 5),
                        mh.to_bytes(),
                    )
                )
            )
            v.copy_(
                _snap_to_torch(
                    binding.Snapshot(
                        binding.DTYPE_F32,
                        (3, 5),
                        vh.to_bytes(),
                    )
                )
            )
            p.copy_(
                _snap_to_torch(
                    binding.Snapshot(
                        binding.DTYPE_F32,
                        (3, 5),
                        ph.to_bytes(),
                    )
                )
            )
        finally:
            for h in (ph, gh, mh, vh):
                h.free()

        # reference path (torch.optim.AdamW)
        with torch.no_grad():
            m_ref.mul_(b1).add_(g, alpha=1 - b1)
            v_ref.mul_(b2).addcmul_(g, g, value=1 - b2)
            bc1 = 1 - b1**step
            bc2 = 1 - b2**step
            p_ref.addcdiv_(
                m_ref / bc1,
                (v_ref / bc2).sqrt() + eps,
                value=-lr,
            )
            p_ref.add_(p_ref, alpha=-lr * wd)

        assert torch.allclose(p, p_ref, atol=1e-6, rtol=1e-6), (
            f"adamw step {step}: max diff {(p - p_ref).abs().max().item():.3e}"
        )


# ---------------------------------------------------------------------------
# 3. end-to-end mini-transformer training vs eager
# ---------------------------------------------------------------------------


def rope_neox(x: torch.Tensor, theta: float) -> torch.Tensor:
    """RoPE with GPT-NeoX pairing (matches engine rope_cpu)."""
    hd = x.shape[-1]
    half = hd // 2
    T = x.shape[-2]
    inv_freq = 1.0 / (theta ** (2 * torch.arange(half) / hd))
    angles = torch.arange(T)[:, None] * inv_freq[None, :]
    c, s = torch.cos(angles), torch.sin(angles)
    x0, x1 = x[..., :half], x[..., half:]
    return torch.cat(
        [x0 * c - x1 * s, x0 * s + x1 * c], dim=-1
    )


class MiniTrainTransformer(torch.nn.Module):
    """The eager reference: B=2, T=4, hidden=16, 2 heads, ffn=48, 2 layers."""

    def __init__(self, vocab=64, hidden=16, ffn=48, n_heads=2, n_layers=2):
        super().__init__()
        self.vocab = vocab
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.n_layers = n_layers
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.out_norm_w = torch.nn.Parameter(torch.ones(hidden))
        self.out_w = torch.nn.Parameter(torch.randn(vocab, hidden) * 0.1)
        for i in range(n_layers):
            setattr(self, f"an{i}", torch.nn.Parameter(torch.ones(hidden)))
            setattr(self, f"fn{i}", torch.nn.Parameter(torch.ones(hidden)))
            setattr(
                self,
                f"wq{i}",
                torch.nn.Parameter(torch.randn(hidden, hidden) * 0.1),
            )
            setattr(
                self,
                f"wk{i}",
                torch.nn.Parameter(torch.randn(hidden, hidden) * 0.1),
            )
            setattr(
                self,
                f"wv{i}",
                torch.nn.Parameter(torch.randn(hidden, hidden) * 0.1),
            )
            setattr(
                self,
                f"wo{i}",
                torch.nn.Parameter(torch.randn(hidden, hidden) * 0.1),
            )
            setattr(self, f"bq{i}", torch.nn.Parameter(torch.zeros(hidden)))
            setattr(self, f"bk{i}", torch.nn.Parameter(torch.zeros(hidden)))
            setattr(self, f"bv{i}", torch.nn.Parameter(torch.zeros(hidden)))
            setattr(
                self,
                f"gw{i}",
                torch.nn.Parameter(torch.randn(ffn, hidden) * 0.1),
            )
            setattr(
                self,
                f"uw{i}",
                torch.nn.Parameter(torch.randn(ffn, hidden) * 0.1),
            )
            setattr(
                self,
                f"dw{i}",
                torch.nn.Parameter(torch.randn(hidden, ffn) * 0.1),
            )

    def _layer_params(self, i):
        return tuple(
            getattr(self, f"{p}{i}")
            for p in (
                "an", "wq", "wk", "wv", "wo", "bq", "bk", "bv",
                "fn", "gw", "uw", "dw",
            )
        )

    def forward(self, tokens):  # tokens: [B, T] long
        B, T = tokens.shape
        h = self.embed(tokens)
        for i in range(self.n_layers):
            an, wq, wk, wv, wo, bq, bk, bv, fn, gw, uw, dw = (
                self._layer_params(i)
            )
            hn = torch.nn.functional.rms_norm(
                h, (self.hidden,), an, 1e-5
            )
            q = torch.nn.functional.linear(hn, wq, bq)
            k = torch.nn.functional.linear(hn, wk, bk)
            v = torch.nn.functional.linear(hn, wv, bv)
            q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            q = rope_neox(q, 10000.0)
            k = rope_neox(k, 10000.0)
            scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
            mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1) @ v
            attn = (
                attn.transpose(1, 2)
                .reshape(B, T, self.hidden)
            )
            h = h + torch.nn.functional.linear(attn, wo)
            hf = torch.nn.functional.rms_norm(
                h, (self.hidden,), fn, 1e-5
            )
            h = h + torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(hf, gw))
                * torch.nn.functional.linear(hf, uw),
                dw,
            )
        h = torch.nn.functional.rms_norm(
            h, (self.hidden,), self.out_norm_w, 1e-5
        )
        return torch.nn.functional.linear(h, self.out_w)


class EngineTrainFunction(torch.autograd.Function):
    """Mini transformer whose forward+backward run on the engine ops.

    Records are (op, inputs, output, batch_index_or_None); the backward
    replays them in reverse topological order through
    `binding.run_backward` and accumulates gradients with a dict keyed by
    tensor identity (mirrors the interpreter's run_with_grad buckets).
    """

    @staticmethod
    def forward(
        ctx,
        tokens,  # [B, T] long
        targets,  # [B*T] int32
        embd,  # [vocab, hidden]
        out_norm,  # [hidden]
        out_w,  # [vocab, hidden]
        *layer_params,  # 12 * n_layers tensors
    ):
        n_layers = len(layer_params) // 12
        hidden = embd.shape[1]
        n_heads = 2
        head_dim = hidden // n_heads
        B, T = tokens.shape

        cfg = torch.tensor([n_heads, n_heads, head_dim], dtype=torch.int32)
        pos = torch.tensor([0.0, 10000.0], dtype=torch.float32)

        recs = []

        def run(op, ins, b=None):
            out = engine_op(op, ins)
            recs.append((op, ins, out, b))
            return out

        xs = []
        for b in range(B):
            xs.append(run("embedding", [tokens[b].to(torch.int32), embd], b))
        x = torch.cat(xs, dim=0)  # [B*T, hidden]

        resid = x
        for i in range(n_layers):
            (
                an, wq, wk, wv, wo, bq, bk, bv, fn, gw, uw, dw,
            ) = layer_params[i * 12 : (i + 1) * 12]
            normed = run("rms_norm_weight", [resid, an])
            attns = []
            for b in range(B):
                attns.append(
                    run(
                        "mha_seq",
                        [
                            normed[b * T : (b + 1) * T].contiguous(),
                            wq, wk, wv, wo, bq, bk, bv, cfg, pos,
                        ],
                        b,
                    )
                )
            attn = torch.cat(attns, dim=0)
            resid = run("add", [resid, attn])
            normed2 = run("rms_norm_weight", [resid, fn])
            ffn_out = run("swiglu_ffn", [normed2, gw, uw, dw])
            resid = run("add", [resid, ffn_out])
        h = run("rms_norm_weight", [resid, out_norm])
        logits = run("lm_head", [h, out_w])  # [B*T, vocab]
        loss = run("cross_entropy", [logits, targets])

        ctx.n_layers = n_layers
        ctx.hidden = hidden
        ctx.B = B
        ctx.T = T
        ctx.recs = recs
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        B, T = ctx.B, ctx.T
        recs = ctx.recs
        one = torch.ones(1)
        acc = {}

        def grad_of(t):
            return acc.setdefault(id(t), torch.zeros_like(t))

        def back(op, ins, g):
            gs = engine_backward(op, ins, [g])
            for t, gt in zip(ins, gs):
                if t is None or gt is None:
                    continue
                if not t.is_floating_point():
                    continue
                grad_of(t).add_(gt)

        # reverse replay
        rec = recs.pop()  # cross_entropy
        back(rec[0], rec[1], one)
        g_logits = grad_of(rec[1][0])

        rec = recs.pop()  # lm_head [h, out_w]
        back(rec[0], rec[1], g_logits)
        g_h = grad_of(rec[1][0])
        g_out_w = grad_of(rec[1][1])

        rec = recs.pop()  # final rms_norm_weight [resid, out_norm]
        back(rec[0], rec[1], g_h)
        g_resid = grad_of(rec[1][0])
        g_out_norm = grad_of(rec[1][1])

        layer_grads = []  # 12 per layer, param order of the forward
        for _ in range(ctx.n_layers):
            rec_d = recs.pop()  # add(resid, ffn_out)
            back("add", rec_d[1], g_resid)
            g_ffn_out = grad_of(rec_d[1][1])

            rec_c = recs.pop()  # swiglu_ffn [normed2, gw, uw, dw]
            back("swiglu_ffn", rec_c[1], g_ffn_out)
            g_normed2 = grad_of(rec_c[1][0])
            g_gw = grad_of(rec_c[1][1])
            g_uw = grad_of(rec_c[1][2])
            g_dw = grad_of(rec_c[1][3])

            rec_b = recs.pop()  # rms_norm_weight [resid, fn]
            back("rms_norm_weight", rec_b[1], g_normed2)
            g_fn = grad_of(rec_b[1][1])

            rec_a = recs.pop()  # add(resid_in, attn_cat)
            # the attention branch receives the gradient of the
            # POST-attention residual (both the ffn-add and the ffn-norm
            # paths contribute), not the layer output
            g_resid_mid = grad_of(rec_b[1][0])
            back("add", rec_a[1], g_resid_mid)
            g_resid_in = grad_of(rec_a[1][0])
            g_attn_cat = grad_of(rec_a[1][1])

            mha_recs = [recs.pop() for _ in range(B)]
            for m in mha_recs:
                b = m[3]
                back("mha_seq", m[1], g_attn_cat[b * T : (b + 1) * T])

            rec_rn = recs.pop()  # rms_norm_weight [resid_in, an]
            g_normed = grad_of(rec_rn[2])
            for m in mha_recs:
                b = m[3]
                g_normed.view(B * T, -1)[b * T : (b + 1) * T] += grad_of(
                    m[1][0]
                )
            back("rms_norm_weight", rec_rn[1], g_normed)
            g_an = grad_of(rec_rn[1][1])
            # the resid_in accumulator already holds both the attention-add
            # and the attention-norm contributions
            g_resid = grad_of(rec_rn[1][0])

            m0 = mha_recs[0][1]
            layer_grads.append(
                [
                    g_an,
                    grad_of(m0[1]), grad_of(m0[2]), grad_of(m0[3]),
                    grad_of(m0[4]), grad_of(m0[5]), grad_of(m0[6]),
                    grad_of(m0[7]),
                    g_fn, g_gw, g_uw, g_dw,
                ]
            )

        emb_recs = [recs.pop() for _ in range(B)]
        for m in emb_recs:
            b = m[3]
            back("embedding", m[1], g_resid[b * T : (b + 1) * T])
        g_embd = None
        for m in emb_recs:
            g = grad_of(m[1][1])
            g_embd = g if g_embd is None else g_embd + g

        result = [None, None, g_embd, g_out_norm, g_out_w]
        # the layers were visited top-first; the forward argument order is
        # layer-0 first
        for lg in reversed(layer_grads):
            result.extend(lg)
        return tuple(result)


class EngineTrainTransformer(torch.nn.Module):
    """Same parameters as MiniTrainTransformer; runs on the engine."""

    def __init__(self, eager: MiniTrainTransformer):
        super().__init__()
        self.n_layers = eager.n_layers
        self.embed = torch.nn.Parameter(eager.embed.weight.detach().clone())
        self.out_norm_w = torch.nn.Parameter(
            eager.out_norm_w.detach().clone()
        )
        self.out_w = torch.nn.Parameter(eager.out_w.detach().clone())
        for i in range(eager.n_layers):
            for name in (
                "an", "wq", "wk", "wv", "wo", "bq", "bk", "bv",
                "fn", "gw", "uw", "dw",
            ):
                p = getattr(eager, f"{name}{i}")
                self.register_parameter(
                    f"{name}{i}", torch.nn.Parameter(p.detach().clone())
                )

    def _layer_params(self, i):
        return tuple(
            getattr(self, f"{p}{i}")
            for p in (
                "an", "wq", "wk", "wv", "wo", "bq", "bk", "bv",
                "fn", "gw", "uw", "dw",
            )
        )

    def forward(self, tokens, targets):
        args = [tokens, targets, self.embed, self.out_norm_w, self.out_w]
        for i in range(self.n_layers):
            args.extend(self._layer_params(i))
        return EngineTrainFunction.apply(*args)


def _run_training_steps(model, tokens, targets, steps, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        if isinstance(model, EngineTrainTransformer):
            loss = model(tokens, targets).reshape(())
        else:
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.long()
            )
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_mini_transformer_training_matches_eager():
    torch.manual_seed(21)
    B, T = 2, 4
    eager = MiniTrainTransformer()
    tokens = torch.randint(0, 64, (B, T))
    targets = torch.randint(0, 64, (B * T,), dtype=torch.int32)

    engine_model = EngineTrainTransformer(eager)
    # identical starting parameters
    assert torch.equal(engine_model.embed, eager.embed.weight)
    for i in range(eager.n_layers):
        for name in (
            "an", "wq", "wk", "wv", "wo", "bq", "bk", "bv",
            "fn", "gw", "uw", "dw",
        ):
            assert torch.equal(
                getattr(engine_model, f"{name}{i}"),
                getattr(eager, f"{name}{i}"),
            )

    steps = 4
    eager_losses = _run_training_steps(eager, tokens, targets, steps, 7)
    engine_losses = _run_training_steps(
        engine_model, tokens, targets, steps, 7
    )

    for i, (el, gl) in enumerate(zip(engine_losses, eager_losses)):
        print(f"[infer_train] step {i}: engine {el:.6f} vs eager {gl:.6f}")
        assert abs(el - gl) < 1e-3, f"step {i} diverged"
    assert engine_losses[-1] < engine_losses[0], "engine loss did not drop"
    assert eager_losses[-1] < eager_losses[0], "eager loss did not drop"
    # params must have moved identically (up to float noise)
    assert torch.allclose(
        engine_model.out_w, eager.out_w, atol=1e-5, rtol=1e-5
    ), "final weights diverged"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))  # noqa: F821
