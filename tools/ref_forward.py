#!/usr/bin/env python3
"""Independent f32 reference forward for bisecting the Mojo implementation.

Dequantizes all weights from the GGUF (vectorized numpy) and runs the same
Qwen2 algorithm as the Mojo transformer.  Prints the per-step top-5 and
compares against reference_logits_5.npy (llama-cpp-python ground truth).
"""
import struct
import numpy as np

GGUF = "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"

# GGML metadata type -> element size in bytes.
TYPE_SIZE = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8
}

f = open(GGUF, "rb")
magic = f.read(4)
ver, n_t, n_m = struct.unpack("<IQQ", f.read(20))

def rs():
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n)

for _ in range(n_m):
    key = rs().decode("utf-8", "replace")
    t = struct.unpack("<I", f.read(4))[0]
    if t == 8:
        rs()
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        ln = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            for _ in range(ln):
                rs()
        else:
            sz = TYPE_SIZE[et]
            f.read(sz * ln)
    else:
        sz = TYPE_SIZE[t]
        f.read(sz)

tensors = {}
for _ in range(n_t):
    name = rs().decode("utf-8", "replace")
    nd = struct.unpack("<I", f.read(4))[0]
    dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
    gtype = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    tensors[name] = (dims, gtype, off)

data_offset = (f.tell() + 31) // 32 * 32
file_size = f.seek(0, 2)


def get_scale_min_k4_vec(j, scales):
    if j < 4:
        return scales[:, j] & 63, scales[:, j + 4] & 63
    d = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
    m = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return d, m


def deq_q5k(raw, numel):
    nb = numel // 256
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 176)
    dm = blocks[:, 0:4].reshape(-1).view(np.float16).reshape(nb, 2)
    dm = dm.astype(np.float32)
    d = dm[:, 0]
    dmin = dm[:, 1]
    scales = blocks[:, 4:16]
    qh = blocks[:, 16:48]
    qs = blocks[:, 48:176]
    out = np.empty((nb, 256), dtype=np.float32)
    for sb in range(8):
        sc, m = get_scale_min_k4_vec(sb, scales)
        d_sb = d * sc.astype(np.float32)
        m_sb = dmin * m.astype(np.float32)
        ql = qs[:, (sb // 2) * 32:(sb // 2) * 32 + 32]
        nib = (ql & 0xF) if sb % 2 == 0 else (ql >> 4)
        bit = 1 << sb
        q = nib + np.where((qh & bit) != 0, 16, 0).astype(np.int16)
        out[:, sb * 32:(sb + 1) * 32] = d_sb[:, None] * q - m_sb[:, None]
    return out.reshape(-1)


def deq_q6k(raw, numel):
    nb = numel // 256
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 210)
    d = blocks[:, 208:210].reshape(-1).view(np.float16).astype(np.float32)
    ql = blocks[:, 0:128].astype(np.int16)
    qh = blocks[:, 128:192].astype(np.int16)
    sc = blocks[:, 192:208].view(np.int8).astype(np.float32)
    out = np.empty((nb, 256), dtype=np.float32)
    idx = (np.arange(32) // 16)[None, :]
    for n in range(2):
        qlo = ql[:, n * 64:n * 64 + 32]
        qhi = ql[:, n * 64 + 32:n * 64 + 64]
        qhb = qh[:, n * 32:n * 32 + 32]
        q1 = ((qlo & 0xF) | ((qhb & 3) << 4)) - 32
        q2 = ((qhi & 0xF) | (((qhb >> 2) & 3) << 4)) - 32
        q3 = ((qlo >> 4) | (((qhb >> 4) & 3) << 4)) - 32
        q4 = ((qhi >> 4) | (((qhb >> 6) & 3) << 4)) - 32
        s0 = np.take_along_axis(sc[:, n * 8:n * 8 + 8], idx, axis=1)
        s1 = np.take_along_axis(sc[:, n * 8 + 2:n * 8 + 10], idx, axis=1)
        s2 = np.take_along_axis(sc[:, n * 8 + 4:n * 8 + 12], idx, axis=1)
        s3 = np.take_along_axis(sc[:, n * 8 + 6:n * 8 + 14], idx, axis=1)
        out[:, n * 128:n * 128 + 32] = d[:, None] * s0 * q1
        out[:, n * 128 + 32:n * 128 + 64] = d[:, None] * s1 * q2
        out[:, n * 128 + 64:n * 128 + 96] = d[:, None] * s2 * q3
        out[:, n * 128 + 96:n * 128 + 128] = d[:, None] * s3 * q4
    return out.reshape(-1)


def load_tensor(name):
    dims, gtype, off = tensors[name]
    numel = int(np.prod(dims))
    f.seek(data_offset + off)
    if gtype == 13:
        raw = f.read(numel // 256 * 176)
        vals = deq_q5k(raw, numel)
    elif gtype == 14:
        raw = f.read(numel // 256 * 210)
        vals = deq_q6k(raw, numel)
    elif gtype == 1:
        raw = f.read(numel * 2)
        vals = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif gtype == 0:
        raw = f.read(numel * 4)
        vals = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"unsupported type {gtype} for {name}")
    # GGUF dims are (outer, inner); our matmuls want W[in, out] row-major
    # = exactly the file layout, so keep dims as-is (n_out, n_in) reversed:
    return vals.reshape(tuple(dims))


print("dequantizing...")
emb = load_tensor("token_embd.weight")          # (1536, 151936)
out_norm = load_tensor("output_norm.weight")    # (1536,)
wout = load_tensor("output.weight")             # (1536, 151936)

N_LAYERS = 28
HIDDEN = 1536
N_HEADS = 12
N_KV = 2
HEAD_DIM = 128
EPS = 1e-6
THETA = 10000.0

wq = []; wk = []; wv = []; wo = []; bq = []; bk = []; bv = []
an = []; fn = []; wg = []; wu = []; wd = []
for i in range(N_LAYERS):
    wq.append(load_tensor(f"blk.{i}.attn_q.weight"))
    wk.append(load_tensor(f"blk.{i}.attn_k.weight"))
    wv.append(load_tensor(f"blk.{i}.attn_v.weight"))
    wo.append(load_tensor(f"blk.{i}.attn_output.weight"))
    bq.append(load_tensor(f"blk.{i}.attn_q.bias"))
    bk.append(load_tensor(f"blk.{i}.attn_k.bias"))
    bv.append(load_tensor(f"blk.{i}.attn_v.bias"))
    an.append(load_tensor(f"blk.{i}.attn_norm.weight"))
    fn.append(load_tensor(f"blk.{i}.ffn_norm.weight"))
    wg.append(load_tensor(f"blk.{i}.ffn_gate.weight"))
    wu.append(load_tensor(f"blk.{i}.ffn_up.weight"))
    wd.append(load_tensor(f"blk.{i}.ffn_down.weight"))
print("weights loaded")



import numpy as _np

def rmsnorm(x, w):
    rms = _np.sqrt(
        _np.mean(x.astype(_np.float64) ** 2, axis=-1, keepdims=True) + EPS
    )
    return (x / rms).astype(_np.float32) * w

def rope(x, pos):
    half = x.shape[-1] // 2
    inv = THETA ** (-2 * _np.arange(half) / x.shape[-1])
    ang = pos * inv
    c = _np.cos(ang); s = _np.sin(ang)
    out = _np.empty_like(x)
    out[..., :half] = x[..., :half] * c - x[..., half:] * s
    out[..., half:] = x[..., :half] * s + x[..., half:] * c
    return out

def forward(token, pos, K, V):
    x = emb.reshape(151936, 1536)[token].astype(_np.float32)[None, :]
    for l in range(N_LAYERS):
        xn = rmsnorm(x, an[l])
        q = (wq[l].reshape(1536, 1536) @ xn[0]) + bq[l]
        k = (wk[l].reshape(256, 1536) @ xn[0]) + bk[l]
        v = (wv[l].reshape(256, 1536) @ xn[0]) + bv[l]
        qr = rope(q.reshape(N_HEADS, HEAD_DIM), pos)
        kr = rope(k.reshape(N_KV, HEAD_DIM), pos)
        K[l][:, pos] = kr
        V[l][:, pos] = v.reshape(N_KV, HEAD_DIM)
        seq = pos + 1
        attn = _np.zeros((N_HEADS, HEAD_DIM), dtype=_np.float32)
        for h in range(N_HEADS):
            kv = h * N_KV // N_HEADS
            s = (qr[h] @ K[l][kv, :seq].T) / _np.sqrt(HEAD_DIM)
            s = s - s.max()
            p = _np.exp(s); p = p / p.sum()
            attn[h] = p @ V[l][kv, :seq]
        x = x + (wo[l].reshape(1536, 1536) @ attn.reshape(1536))[None, :]
        xn2 = rmsnorm(x, fn[l])
        g = wg[l].reshape(8960, 1536) @ xn2[0]
        u = wu[l].reshape(8960, 1536) @ xn2[0]
        h = g / (1 + _np.exp(-g)) * u
        x = x + (wd[l].reshape(1536, 8960) @ h)[None, :]
    xf = rmsnorm(x, out_norm)
    return (wout.reshape(151936, 1536) @ xf[0])[None, :]

ref = _np.load("reference_logits_5.npy")
toks = [151646, 16, 10, 16, 28]
K = _np.zeros((N_LAYERS, N_KV, 64, HEAD_DIM), dtype=_np.float32)
V = _np.zeros((N_LAYERS, N_KV, 64, HEAD_DIM), dtype=_np.float32)
for i, tok in enumerate(toks):
    logits = forward(tok, i, K, V)[0]
    rrow = ref[i+1]
    top5 = _np.argsort(logits)[::-1][:5].tolist()
    rtop5 = _np.argsort(rrow)[::-1][:5].tolist()
    corr = _np.corrcoef(logits, rrow)[0, 1]
    print(f"step {i}: top5 {top5} ref {rtop5} corr {corr:.5f}")
print("DONE")
