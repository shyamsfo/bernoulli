#!/usr/bin/env bash
# Run Bernoulli + Generative + (BGE-m3+LR where applicable) at N=500 across
# the 4 M9 use-case benchmarks. Expected to run on the dev box
# (bernoulli, g5.xlarge, A10G 24GB).
#
# Writes `benchmarks/<spec>/results/<YYYY-MM-DD>.main.md` per benchmark.
#
# WildGuardTest is deliberately skipped — the HF dataset `allenai/wildguardmix`
# is license-gated and needs a one-time acceptance + HF_TOKEN. Can be added
# later as a one-off once the auth story is sorted.
#
# XSTest runs Bernoulli + Generative only — no labeled train split, so
# BGE-m3+LR doesn't apply (same convention as the just sweep-usecase recipe).
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

# spec:baselines (XSTest has no train split → no BGE+LR)
declare -a SPECS=(
  "guardrails/toxicchat|bernoulli,generative,bge-m3-lr"
  "guardrails/xstest|bernoulli,generative"
  "triage/clinc150_oos|bernoulli,generative,bge-m3-lr"
  "ratings/yelp_stars|bernoulli,generative,bge-m3-lr"
)

# Non-interactive SSH doesn't source ~/.bashrc, so uv isn't on PATH by default.
export PATH="$HOME/.local/bin:$PATH"

echo "== Use-case sweep: N=$N, train_N=$TRAIN_N, date=$DATE_TAG =="
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
for entry in "${SPECS[@]}"; do
  spec="${entry%%|*}"
  baselines="${entry##*|}"
  out="benchmarks/$spec/results/${DATE_TAG}.main.md"
  echo
  echo "== $spec | $baselines → $out =="
  mkdir -p "$(dirname "$out")"
  uv run python -m benchmarks.run "$spec" \
    --baselines "$baselines" \
    --limit "$N" \
    --train-limit "$TRAIN_N" \
    --bernoulli-url "http://127.0.0.1:${PORT}" \
    --device cuda \
    --out "$out"
done

echo
echo "== done. use-case results =="
ls -la benchmarks/guardrails/*/results/${DATE_TAG}.main.* \
       benchmarks/triage/*/results/${DATE_TAG}.main.* \
       benchmarks/ratings/*/results/${DATE_TAG}.main.* 2>/dev/null || true
