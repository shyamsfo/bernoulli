#!/usr/bin/env bash
# Launched by `just serve-start` via `setsid --fork bash deploy/start-serve.sh`.
# Inherits env vars BERNOULLI_SCORER, BERNOULLI_MAX_MODEL_LEN, BERNOULLI_API_KEYS
# from the caller's `env ... bash deploy/start-serve.sh` invocation.
# `exec` replaces this bash with uvicorn so no extra bash hangs around holding
# SSH's stdout/stderr open — the surrounding setsid --fork completes the detach.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
exec uv run uvicorn bernoulli.server:app --host 0.0.0.0 --port 8000 \
  > "$HOME/serve.log" 2>&1 < /dev/null
