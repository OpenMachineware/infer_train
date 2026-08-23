#!/usr/bin/env python
"""Generate gguf-py-validated dequant reference bins for tests/test_dequant_m7.mojo.

Uses gguf-py when importable (authoritative); falls back to a bundled pure
numpy implementation of the same llama.cpp formulas otherwise.
"""

import sys

import numpy as np

# -- pure-numpy fallback reference (llama.cpp ggml-quants.c formulas) ---------

KV = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def scale_min_k4(j, scales):
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            ((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)))


def deq_fallback(gt, raw, numel):
    a = np.frombuffer(raw, dtype=np.uint8).copy()
    out = np.zeros(numel, dtype=np.float32)
    if gt == 0:
        return np.frombuffer(raw, dtype=np.float32).astype(np.float32)[:numel]
    if gt == 1:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)[:numel]
    if gt == 8:  # Q8_0
        for b in range(numel // 32):
            blk = a[b * 34:(b + 1) * 34]
            d = np.frombuffer(blk[:2], dtype=np.float16)[0].astype(np.float32)
            out[b * 32:(b + 1) * 32] = d * blk[2:34].astype(np.int8)
        return out
    if gt == 12:  # Q4_K
        for b in range(numel // 256):
            blk = a[b * 144:(b + 1) * 144]
            d = np.frombuffer(blk[:2], dtype=np.float16)[0].astype(np.float32)
            dmin = np.frombuffer(blk[2:4], dtype=np.float16)[0].astype(np.float32)
            scales = blk[4:16]
            qs = blk[16:144]
            for sb in range(8):
                sc, m = scale_min_k4(sb, scales)
                ql_base = (sb // 2) * 32
                nib = qs[ql_base:ql_base + 32] & 0xF if sb % 2 == 0 else qs[ql_base:ql_base + 32] >> 4
                out[b * 256 + sb * 32:b * 256 + (sb + 1) * 32] = d * sc * nib - dmin * m
        return out
    if gt == 13:  # Q5_K
        for b in range(numel // 256):
            blk = a[b * 176:(b + 1) * 176]
            d = np.frombuffer(blk[:2], dtype=np.float16)[0].astype(np.float32)
            dmin = np.frombuffer(blk[2:4], dtype=np.float16)[0].astype(np.float32)
            scales = blk[4:16]
            qh = blk[16:48]
            qs = blk[48:176]
            for sb in range(8):
                sc, m = scale_min_k4(sb, scales)
                ql_base = (sb // 2) * 32
                nib = (qs[ql_base:ql_base + 32] & 0xF) if sb % 2 == 0 else (qs[ql_base:ql_base + 32] >> 4)
                add = np.where((qh & (1 << sb)) != 0, 16, 0)
                out[b * 256 + sb * 32:b * 256 + (sb + 1) * 32] = d * sc * (nib + add) - dmin * m
        return out
    if gt == 14:  # Q6_K
        for b in range(numel // 256):
            blk = a[b * 210:(b + 1) * 210]
            d = np.frombuffer(blk[208:210], dtype=np.float16)[0].astype(np.float32)
            ql = blk[0:128].astype(np.int32)
            qh = blk[128:192].astype(np.int32)
            sc = blk[192:208].astype(np.int8).astype(np.float32)
            for n in range(2):
                for l in range(32):
                    si = l // 16
                    q1 = ((ql[64 * n + l] & 0xF) | ((qh[32 * n + l] & 3) << 4)) - 32
                    q2 = ((ql[64 * n + l + 32] & 0xF) | (((qh[32 * n + l] >> 2) & 3) << 4)) - 32
                    q3 = ((ql[64 * n + l] >> 4) | (((qh[32 * n + l] >> 4) & 3) << 4)) - 32
                    q4 = ((ql[64 * n + l + 32] >> 4) | (((qh[32 * n + l] >> 6) & 3) << 4)) - 32
                    base = b * 256 + n * 128 + l
                    out[base] = d * sc[8 * n + si] * q1
                    out[base + 32] = d * sc[8 * n + si + 2] * q2
                    out[base + 64] = d * sc[8 * n + si + 4] * q3
                    out[base + 96] = d * sc[8 * n + si + 6] * q4
        return out
    if gt == 20:  # IQ4_NL
        for b in range(numel // 32):
            blk = a[b * 18:(b + 1) * 18]
            d = np.frombuffer(blk[:2], dtype=np.float16)[0].astype(np.float32)
            qs = blk[2:18]
            out[b * 32:b * 32 + 16] = d * KV[qs & 0xF]
            out[b * 32 + 16:b * 32 + 32] = d * KV[qs >> 4]
        return out
    if gt == 23:  # IQ4_XS
        for b in range(numel // 256):
            blk = a[b * 136:(b + 1) * 136]
            d = np.frombuffer(blk[:2], dtype=np.float16)[0].astype(np.float32)
            sh = int(np.frombuffer(blk[2:4], dtype=np.uint16)[0])
            scales_l = blk[4:8]
            qs = blk[8:136]
            for ib in range(8):
                low = (scales_l[ib // 2] >> (4 * (ib % 2))) & 0xF
                high = (sh >> (2 * ib)) & 3
                ls = low | (high << 4)
                dl = d * (ls - 32)
                q = qs[ib * 16:(ib + 1) * 16]
                out[b * 256 + ib * 32:b * 256 + ib * 32 + 16] = dl * KV[q & 0xF]
                out[b * 256 + ib * 32 + 16:b * 256 + ib * 32 + 32] = dl * KV[q >> 4]
        return out
    raise ValueError(f"unsupported type {gt}")


def parse_gguf(path):
    with open(path, "rb") as f:
        data = f.read()
    nt = int(np.frombuffer(data[8:16], dtype=np.uint64)[0])
    nkv = int(np.frombuffer(data[16:24], dtype=np.uint64)[0])
    off = 24

    def rd_str(o):
        n = int(np.frombuffer(data[o:o + 8], dtype=np.uint64)[0])
        return data[o + 8:o + 8 + n], o + 8 + n

    for _ in range(nkv):
        key, off = rd_str(off)
        t = int(np.frombuffer(data[off:off + 4], dtype=np.uint32)[0])
        off += 4
        if t == 8:
            _, off = rd_str(off)
        elif t in (0, 1):
            off += 1
        elif t in (2, 3):
            off += 2
        elif t in (4, 5, 6, 7):
            off += 4
        elif t == 9:
            et = int(np.frombuffer(data[off:off + 4], dtype=np.uint32)[0])
            n = int(np.frombuffer(data[off + 4:off + 12], dtype=np.uint64)[0])
            off += 12
            if et == 8:
                for _ in range(n):
                    _, off = rd_str(off)
            elif et in (0, 1):
                off += n
            else:
                off += n * (2 if et in (2, 3) else 4 if et in (4, 5, 6, 7) else 8)
        elif t == 10:
            off += 8
        elif t == 12:
            off += 8
    tensors = {}
    for _ in range(nt):
        name, off = rd_str(off)
        name = name.decode("utf8")
        nd = int(np.frombuffer(data[off:off + 4], dtype=np.uint32)[0])
        off += 4
        dims = tuple(int(x) for x in np.frombuffer(data[off:off + 8 * nd], dtype=np.uint64))
        off += 8 * nd
        gt = int(np.frombuffer(data[off:off + 4], dtype=np.uint32)[0])
        off += 4
        to = int(np.frombuffer(data[off:off + 8], dtype=np.uint64)[0])
        off += 8
        tensors[name] = (gt, dims, to)
    return data, tensors, (off + 31) & ~31


SIZES = {0: 4, 1: 2, 8: 34, 12: 144, 13: 176, 14: 210, 20: 18, 23: 136}
NBLK = {0: 1, 1: 1, 8: 32, 12: 256, 13: 256, 14: 256, 20: 32, 23: 256}
N = 1024

PLAN = [
    ("Hy-MT2-7B-Q4_K_M.gguf", "blk.0.attn_k.weight", "dequant_ref_hy.npz_0.bin"),
    ("Hy-MT2-7B-Q4_K_M.gguf", "blk.0.attn_output.weight", "dequant_ref_hy.npz_1.bin"),
    ("Hy-MT2-7B-Q4_K_M.gguf", "token_embd.weight", "dequant_ref_hy.npz_2.bin"),
    ("Hy-MT2-7B-Q4_K_M.gguf", "output_norm.weight", "dequant_ref_hy.npz_3.bin"),
    ("Qwen3.8-27B-UD-Q5_K_M.gguf", "blk.0.attn_norm.weight", "dequant_ref_q35.npz_0.bin"),
    ("Qwen3.8-27B-UD-Q5_K_M.gguf", "blk.0.ssm_alpha.weight", "dequant_ref_q35.npz_1.bin"),
    ("Qwen3.8-27B-UD-Q5_K_M.gguf", "blk.2.ffn_up.weight", "dequant_ref_q35.npz_2.bin"),
    ("Qwen3.8-27B-UD-Q5_K_M.gguf", "blk.0.ffn_gate.weight", "dequant_ref_q35.npz_3.bin"),
    ("Qwen3.8-27B-UD-Q5_K_M.gguf", "blk.0.attn_qkv.weight", "dequant_ref_q35.npz_4.bin"),
]


def gen_with_gguf_py(dest):
    sys.path.insert(0, "/tmp/llama_ref/llama.cpp/gguf-py")
    from gguf import GGUFReader

    import gguf.quants as q

    QT = {'Q4_K': q.Q4_K, 'Q5_K': q.Q5_K, 'Q6_K': q.Q6_K, 'Q8_0': q.Q8_0,
          'IQ4_NL': q.IQ4_NL, 'IQ4_XS': q.IQ4_XS}
    SZ = {'Q4_K': 144, 'Q5_K': 176, 'Q6_K': 210, 'Q8_0': 34, 'IQ4_NL': 18, 'IQ4_XS': 136}
    NB = {'Q4_K': 256, 'Q5_K': 256, 'Q6_K': 256, 'Q8_0': 32, 'IQ4_NL': 32, 'IQ4_XS': 256}
    readers = {}
    for model, name, out in PLAN:
        r = readers.get(model) or GGUFReader(model)
        readers[model] = r
        t = [x for x in r.tensors if x.name == name][0]
        tn = t.tensor_type.name
        numel = int(np.prod(t.shape))
        if tn == "F32":
            ref = np.frombuffer(t.data, dtype=np.float32)[:N].astype(np.float32)
        elif tn == "F16":
            ref = np.frombuffer(t.data, dtype=np.float16)[:N].astype(np.float32)
        else:
            nbytes = (N // NB[tn]) * SZ[tn]
            arr = np.frombuffer(t.data[:nbytes], dtype=np.uint8).reshape((-1, SZ[tn]))
            ref = QT[tn].dequantize_blocks(arr).reshape(-1)[:N].astype(np.float32)
        ref.tofile(dest + out)


def gen_fallback(dest):
    for model, name, out in PLAN:
        data, tensors, data_off = parse_gguf(model)
        gt, dims, to = tensors[name]
        numel = int(np.prod(dims))
        nbytes = (N // NBLK[gt]) * SIZES[gt] if gt not in (0, 1) else N * SIZES[gt]
        raw = data[data_off + to:data_off + to + nbytes]
        if gt in (0, 1):
            ref = deq_fallback(gt, raw, N)
        else:
            ref = deq_fallback(gt, raw, N)
        ref.tofile(dest + out)


SIZES_T = {k: v for k, v in SIZES.items()}
NBLK_T = {k: v for k, v in NBLK.items()}


def main():
    dest = "/tmp/"
    try:
        gen_with_gguf_py(dest)
        print("refs generated with gguf-py")
    except Exception as exc:  # noqa: BLE001
        print(f"gguf-py unavailable ({exc}); using the bundled numpy reference")
        gen_fallback(dest)
    print("done")


if __name__ == "__main__":
    main()
