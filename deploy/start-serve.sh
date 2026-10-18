#!/usr/bin/env bash
# Launches uvicorn, detached, with env sourced from ~/.bernoulli.env.
# Invoked both by:
#   - Mac-side `just serve-start`  → `setsid --fork bash deploy/start-serve.sh`
#   - On-box  `just start`         → same, from on-box/justfile
#
# `exec` replaces this bash with uvicorn so no bash parent hangs around
# holding SSH's stdout open; `setsid --fork` by the caller completes the
# detach so `ssh ... <this script>` returns right away.
#
# ~/.bernoulli.env holds BERNOULLI_API_KEYS + BERNOULLI_SCORER + BERNOULLI_MAX_MODEL_LEN,
# chmod 600, written by Mac-side `just serve-start` from ~/.credentials on
# the laptop. Never committed to git.
set -e

if [ -f "$HOME/.bernoulli.env" ]; then
  set -a
  . "$HOME/.bernoulli.env"
  set +a
fi

cd "$(dirname "$(readlink -f "$0")")/.."
exec uv run uvicorn bernoulli.server:app --host 0.0.0.0 --port 8000 \
  > "$HOME/serve.log" 2>&1 < /dev/null
