#!/usr/bin/env bash
# M8: multi-process RPC test (llama.cpp-style -sm layer --rpc).
#
# Starts two it-rpc-server workers on localhost, runs the same greedy
# generation (--temp 0 --top-k 1) once locally and once across the two
# workers (master: it-cli), and requires the outputs to match exactly.
# The fp16 hidden state crosses the wire losslessly and every layer runs
# the same kernel code, so a distributed run is numerically identical to
# the local one.
#
# Usage: make test-rpc   (or: bash tools/test_rpc.sh)
# Env:   RPC_TEST_MODEL (default: the 1.5B GGUF at the repo root),
#        RPC_PORT1 / RPC_PORT2 (default: 50052 / 50053).

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${RPC_TEST_MODEL:-DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf}"
if [ ! -f "$MODEL" ]; then
    echo "SKIP: $MODEL not found (set RPC_TEST_MODEL to override)"
    exit 0
fi

P1="${RPC_PORT1:-50052}"
P2="${RPC_PORT2:-50053}"
L1="$(mktemp /tmp/infer_train_rpc1.XXXXXX.log)"
L2="$(mktemp /tmp/infer_train_rpc2.XXXXXX.log)"

./it-rpc-server -m "$MODEL" --port "$P1" >"$L1" 2>&1 &
S1=$!
./it-rpc-server -m "$MODEL" --port "$P2" >"$L2" 2>&1 &
S2=$!
trap 'kill $S1 $S2 2>/dev/null || true; rm -f "$L1" "$L2"' EXIT

# Wait for both listeners (the "listening" line is printed after bind,
# before accept - probing the port would consume the single accept slot).
for log in "$L1" "$L2"; do
    for _ in $(seq 1 100); do
        grep -q "listening" "$log" 2>/dev/null && break
        sleep 0.1
    done
    grep -q "listening" "$log" || { echo "FAIL: worker did not start"; cat "$log"; exit 1; }
done

PROMPT="What is 1+1? Answer with just the number."
N=16

# gen <extra args...> -> the generated text (everything after the
# "tokenizer:" header line, which differs between local and rpc runs).
gen() {
    ./it-cli -m "$MODEL" -p "$PROMPT" -n "$N" --seed 7 --temp 0 --top-k 1 \
        "$@" 2>/dev/null | sed -n '/tokenizer: /,$p' | tail -n +2
}

echo "--- local (single process) ---"
OUT_LOCAL="$(gen)"
echo "$OUT_LOCAL"
echo "--- rpc (-sm layer, 2 workers) ---"
OUT_RPC="$(gen -sm layer --rpc "127.0.0.1:$P1" --rpc "127.0.0.1:$P2")"
echo "$OUT_RPC"

# The workers must have received the layer assignment (28 layers / 2).
# (Mojo print separates arguments with spaces: "layers 0 .. 13".)
grep -q "shard ready: layers 0 .. 13" "$L1" || { echo "FAIL: worker 1 shard"; cat "$L1"; exit 1; }
grep -q "shard ready: layers 14 .. 27" "$L2" || { echo "FAIL: worker 2 shard"; cat "$L2"; exit 1; }

if [ -z "$OUT_LOCAL" ] || [ -z "$OUT_RPC" ]; then
    echo "FAIL: empty output"
    exit 1
fi
if [ "$OUT_LOCAL" != "$OUT_RPC" ]; then
    echo "FAIL: rpc output differs from local"
    exit 1
fi
echo "test-rpc OK"
