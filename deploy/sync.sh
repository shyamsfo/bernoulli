#!/usr/bin/env bash
# Push the repo to the dev box. Run from the repo root OR deploy/.
# Usage: ./deploy/sync.sh         # one-way push, no delete
#        ./deploy/sync.sh --mirror  # mirror (deletes remote files not in local)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${BERNOULLI_HOST:-bernoulli}"
REMOTE_DIR="${BERNOULLI_REMOTE_DIR:-/home/ubuntu/bernoulli}"

DELETE_FLAG=""
if [ "${1:-}" = "--mirror" ]; then
  DELETE_FLAG="--delete"
fi

cd "$REPO_ROOT"

rsync -av $DELETE_FLAG \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='*.pyc' \
  --exclude='deploy/.terraform/' \
  --exclude='deploy/*.tfstate*' \
  --exclude='deploy/*.tfplan' \
  --exclude='product/' \
  ./ "$HOST:$REMOTE_DIR/"

echo
echo "Synced to $HOST:$REMOTE_DIR"
echo "Next: ssh $HOST 'cd $REMOTE_DIR && uv sync && uv run pytest'"
