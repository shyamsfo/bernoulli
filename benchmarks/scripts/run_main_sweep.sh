#!/usr/bin/env bash
# Run Bernoulli + Generative + BGE-m3+LR at N=500 across the 6 academic
# benchmarks. Expected to run on the dev box (bernoulli, g5.xlarge, A10G 24GB).
#
# Writes `benchmarks/academic/<spec>/results/<YYYY-MM-DD>.main.md` per
# benchmark. DeBERTa is deliberately excluded — the DeBERTa column lives
# in `<YYYY-MM-DD>.deberta.md` (produced separately on a Lambda A100).
# A later merge step can combine the two sidecar JSONs into a unified
# per-benchmark report.
#
# Prereqs:
# - Model weights pulled (`just model-pull`) to the local HF cache.
# - `uv sync --extra eval` already run (or let the script run it).
#
# Env overrides:
#   N          — examples per benchmark (default 500)
#   TRAIN_N    — training examples cap for BGE-m3+LR (default 2000)
#   DATE_TAG   — output filename stem (default $(date +%Y-%m-%d))
#   PORT       — Bernoulli server port (default 8000)
set -euo pipefail

DATE_TAG=${DATE_TAG:-$(date +%Y-%m-%d)}
N=${N:-500}
TRAIN_N=${TRAIN_N:-2000}
PORT=${PORT:-8000}

BENCHMARKS=(sst2 ag_news banking77 paws tweeteval_emotion arxiv_post_cutoff)

# Non-interactive SSH doesn't source ~/.bashrc, so uv isn't on PATH by default.
export PATH="$HOME/.local/bin:$PATH"

echo "== Main sweep: N=$N, train_N=$TRAIN_N, date=$DATE_TAG =="
echo "== host: $(hostname) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# --- setup -----------------------------------------------------------------
# NVMe cache (survives nothing but is 229 GB and fast; EBS root is 80 GB).
# Non-interactive SSH doesn't source ~/.bashrc on this box so set HF_HOME here.
export HF_HOME=${HF_HOME:-/opt/dlami/nvme/hf-cache}
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

uv sync --extra eval

MODEL_ID=${BERNOULLI_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}
MODEL_REV=${BERNOULLI_MODEL_REVISION:-cc594898137f460bfe9f0759e9844b3ce807cfb5}
echo "[setup] pre-pulling $MODEL_ID @ $MODEL_REV ..."
uv run hf download "$MODEL_ID" --revision "$MODEL_REV"

# --- start server in background --------------------------------------------
if curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
  echo "[serve] already up on :${PORT}"
  SERVE_PID=""
else
  echo "[serve] starting uvicorn ..."
  nohup uv run uvicorn bernoulli.server:app --host 127.0.0.1 --port "$PORT" \
    > ~/serve.log 2>&1 &
  SERVE_PID=$!
  echo "[serve] PID=$SERVE_PID; waiting for /healthz ..."
  for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
      echo "[serve] ready after ${i} checks"
      break
    fi
    sleep 3
  done
  if ! curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    echo "[serve] FAILED to start in 3 minutes; log tail:"
    tail -40 ~/serve.log
    exit 1
  fi
fi

cleanup() {
  if [ -n "$SERVE_PID" ]; then
    echo "[cleanup] stopping server PID=$SERVE_PID"
    kill "$SERVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# --- sweep -----------------------------------------------------------------
for b in "${BENCHMARKS[@]}"; do
  out="benchmarks/academic/$b/results/${DATE_TAG}.main.md"
  echo
  echo "== academic/$b → $out =="
  mkdir -p "$(dirname "$out")"
  uv run python -m benchmarks.run "academic/$b" \
    --baselines bernoulli,generative,bge-m3-lr \
    --limit "$N" \
    --train-limit "$TRAIN_N" \
    --bernoulli-url "http://127.0.0.1:${PORT}" \
    --device cuda \
    --out "$out"
done

echo
echo "== done. main-sweep results =="
ls -la benchmarks/academic/*/results/${DATE_TAG}.main.* 2>/dev/null || true
