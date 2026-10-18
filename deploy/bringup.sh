#!/usr/bin/env bash
# deploy/bringup.sh — stepwise, resumable bring-up for the Bernoulli serving box.
#
# Each step prints what it's doing, has an explicit timeout, and fails loudly.
# Idempotent: safe to re-run after any failure; steps that already succeeded
# detect their work is done and skip (rsync is incremental, uv sync detects
# already-synced, hf download detects already-cached, uvicorn start kills
# any existing one first).
#
# The explicit-timeout + loud-progress design is a reaction to the
# 2026-10-17/18 incident where a UEFI-reboot-looped box caused `docker build`
# to appear-to-hang for 1h+ because every SSH probe silently timed out. Here
# every SSH call uses short timeouts and prints wall-clock on completion.
#
# Prereqs:
# - EC2 bernoulli instance running + status-ok (both AWS health checks passed)
# - SSH alias `bernoulli` (or BERNOULLI_HOST=<ip>) resolving to the current IP
# - BERNOULLI_API_KEYS exported from ~/.credentials (comma-separated)
# - Local repo at the parent of this script
#
# Usage:
#   source ~/.credentials && ./deploy/bringup.sh

set -euo pipefail

HOST="${BERNOULLI_HOST:-bernoulli}"
SSH_OPTS=(-o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${BERNOULLI_API_KEYS:?set BERNOULLI_API_KEYS in ~/.credentials (comma-separated 64-char hex tokens)}"

step() {
  local n="$1" title="$2"
  shift 2
  local started=$(date +%s)
  echo
  echo "=== [$n] $title ==="
  if ! "$@"; then
    local elapsed=$(( $(date +%s) - started ))
    echo "    FAIL (${elapsed}s)"
    exit 1
  fi
  local elapsed=$(( $(date +%s) - started ))
  echo "    ok (${elapsed}s)"
}

# ---------------------------------------------------------------------------

check_ssh() {
  # 15s timeout; if the box is unreachable we fail here, not an hour later.
  ssh "${SSH_OPTS[@]}" "$HOST" 'echo ok' > /dev/null
}

check_box_health() {
  ssh "${SSH_OPTS[@]}" "$HOST" '
    set -e
    echo "cloud-init:"
    sudo cloud-init status --wait --long | head -3 || echo "cloud-init not available"
    echo "nvidia-smi:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo "nvme mount:"
    df -h /opt/dlami/nvme 2>/dev/null | tail -1 || echo "NVMe not mounted"
    echo "uv:"
    which uv && uv --version
  '
}

sync_code() {
  cd "$REPO_ROOT"
  ./deploy/sync.sh --mirror | tail -5
}

uv_sync() {
  # 15 min overall timeout; keepalives catch stalls within 30s.
  # scorer-hf (torch + transformers) + serve (uvicorn + fastapi).
  timeout 900 ssh "${SSH_OPTS[@]}" "$HOST" '
    cd ~/bernoulli && uv sync --extra scorer-hf --extra serve 2>&1 | tail -10
  '
}

model_pull() {
  # 5 min overall. hf download detects already-cached and skips.
  timeout 300 ssh "${SSH_OPTS[@]}" "$HOST" '
    cd ~/bernoulli && uv run hf download \
      Qwen/Qwen2.5-VL-7B-Instruct \
      --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 2>&1 | tail -5
  '
}

start_server() {
  # Kill any existing uvicorn first.
  ssh "${SSH_OPTS[@]}" "$HOST" 'pkill -f "uvicorn bernoulli.server" 2>/dev/null; sleep 1' || true
  # Start new one via nohup, detached from the SSH session.
  # Env vars inline so BERNOULLI_API_KEYS never lands on disk on the box.
  ssh "${SSH_OPTS[@]}" "$HOST" "cd ~/bernoulli && \
    BERNOULLI_SCORER=hf \
    BERNOULLI_MAX_MODEL_LEN=8192 \
    BERNOULLI_API_KEYS='$BERNOULLI_API_KEYS' \
    nohup uv run uvicorn bernoulli.server:app --host 0.0.0.0 --port 8000 \
      > ~/serve.log 2>&1 </dev/null &
    disown
    sleep 2
    pid=\$(pgrep -f 'uvicorn bernoulli.server' | head -1)
    echo \"uvicorn pid \$pid\"
  "
}

wait_healthz() {
  # Model load + server bind: ~90s on cold start, less on warm.
  # Fail at 2 min with log tail so the error is visible, not hidden.
  for i in $(seq 1 24); do
    if ssh "${SSH_OPTS[@]}" "$HOST" 'curl -sf http://127.0.0.1:8000/healthz' > /dev/null 2>&1; then
      echo "    healthz responded after ${i} probes (~$((i*5))s)"
      return 0
    fi
    sleep 5
  done
  echo
  echo "    ERROR: /healthz did not respond within 2 min"
  echo "    tail of ~/serve.log:"
  ssh "${SSH_OPTS[@]}" "$HOST" 'tail -30 ~/serve.log' 2>/dev/null || true
  return 1
}

verify_public_endpoint() {
  # ssd2 → :8000 on the box. Needs cloudns to point bernoulli-internal at
  # the right IP + nginx re-resolver window (up to 30s) to pick it up.
  # /healthz is auth-exempt so no API key needed for this probe.
  for i in $(seq 1 12); do
    if curl -sf --max-time 10 https://bernoulli-api.nuwire.dev/healthz > /dev/null 2>&1; then
      echo "    public endpoint ok after ${i} probes"
      return 0
    fi
    sleep 5
  done
  echo "    WARN: https://bernoulli-api.nuwire.dev/healthz unreachable in 1 min"
  echo "    (likely cloudns stale: ~/bin/cloudns_mgr get bernoulli-internal nuwire.dev)"
  return 1
}

# ---------------------------------------------------------------------------

step "1/7" "SSH reachable (15s timeout; fail loud if box is boot-looping)" check_ssh
step "2/7" "Box health (cloud-init, GPU, uv, NVMe mount)"                  check_box_health
step "3/7" "Rsync code"                                                     sync_code
step "4/7" "uv sync (torch, transformers, uvicorn, fastapi; 3-5 min)"       uv_sync
step "5/7" "Model pull (Qwen2.5-VL-7B; ~90s first time, instant if cached)" model_pull
step "6/7" "Start uvicorn via nohup"                                        start_server
step "7/7" "Wait for /healthz on the box"                                   wait_healthz

echo
echo "=== public endpoint check ==="
verify_public_endpoint || true

echo
echo "DONE."
echo "  on-box health : ssh $HOST 'curl http://127.0.0.1:8000/healthz'"
echo "  public URL    : https://bernoulli-api.nuwire.dev"
echo "  logs          : ssh $HOST 'tail -f ~/serve.log'"
echo "  stop server   : just serve-stop"
echo "  stop box      : just serve-down"
