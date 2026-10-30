#!/usr/bin/env bash
# Run DeBERTa-v3-zeroshot baseline at N=500 on the 5 ChoiceQuestion academic
# benchmarks. Idempotent setup (installs uv + syncs the eval extra if needed),
# then fires each benchmark's `benchmarks.run --baselines deberta --device cuda`.
#
# Writes `benchmarks/<spec>/results/<YYYY-MM-DD>.deberta.{md,json}` per benchmark.
#
# PAWS (BinaryQuestion) is intentionally skipped — DeBERTaZeroshot.predict()
# raises NotImplementedError on it by design (see benchmarks/common/baselines.py).
# The missing cell is handled at merge time.
#
# Expected to run on a GPU-attached box (Lambda A100/H100 or similar).
# Expected wall-clock on an A100 at N=500: ~15 min for the full suite.
#
# Env overrides:
#   N          — examples per benchmark (default 500)
#   DATE_TAG   — output filename stem (default $(date +%Y-%m-%d))
set -euo pipefail

DATE_TAG=${DATE_TAG:-$(date +%Y-%m-%d)}
N=${N:-500}

BENCHMARKS=(
  "academic/sst2"
  "academic/ag_news"
  "academic/banking77"
  "academic/tweeteval_emotion"
  "academic/arxiv_post_cutoff"
)

echo "== DeBERTa sweep: N=$N, date=$DATE_TAG =="
echo "== host: $(hostname) =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1

# --- setup: uv + env --------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[setup] uv sync --extra eval ..."
uv sync --extra eval

export HF_HUB_ENABLE_HF_TRANSFER=1

# --- sweep ------------------------------------------------------------------
for spec in "${BENCHMARKS[@]}"; do
  out="benchmarks/$spec/results/${DATE_TAG}.deberta.md"
  echo
  echo "== $spec → $out =="
  mkdir -p "$(dirname "$out")"
  uv run python -m benchmarks.run "$spec" \
    --baselines deberta \
    --limit "$N" \
    --device cuda \
    --out "$out"
done

echo
echo "== done. deberta results =="
ls -la benchmarks/academic/*/results/${DATE_TAG}.deberta.* 2>/dev/null || true
