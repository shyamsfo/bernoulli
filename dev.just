# dev.just — development recipes (invoked as `just dev <name>`)
# Imported as a module by the root justfile. Covers code quality (lint / test
# / fmt / ci), local serving (serve-local), Docker image builds, model pulls,
# and clean.

# ---------- code quality ----------

# lint + typecheck + test
ci: lint test
    @echo "ci: ok"

# pytest, full suite including GPU-marked
test:
    uv run pytest -v

# pytest, non-GPU only — fast feedback
test-fast:
    uv run pytest -v -m "not gpu"

# ruff + mypy
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy bernoulli/ evals/

# fix formatting + autofix lints
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# ---------- local serving (dev only; the daily production path is `serve-*` at root) ----------

# Start a local uvicorn server (dev / HFScorer path). Listens on 127.0.0.1:8000
# by default. For the daily production endpoint on bernoulli, use `just serve-up`
# then `just serve-start` from the repo root.
serve-local host="127.0.0.1" port="8000":
    uv run uvicorn bernoulli.server:app --host {{host}} --port {{port}}

# ---------- Docker (vLLM-based production image) ----------

# build the serving docker image
docker-build tag="bernoulli:latest":
    docker build -t {{tag}} .

# run the serving image; mounts the host's HF cache so the container doesn't re-pull the model
docker-run tag="bernoulli:latest" port="8000":
    docker run --rm --gpus all -p {{port}}:8000 \
        -v $HOME/.cache/huggingface:/data/hf-cache \
        -e BERNOULLI_SCORER \
        -e BERNOULLI_MODEL_ID \
        -e BERNOULLI_MODEL_REVISION \
        -e BERNOULLI_MAX_MODEL_LEN \
        {{tag}}

# ---------- model weights ----------

# download the dev-tier backbone to the NVMe cache (~15 GB, ~90 s via hf_transfer)
# `uv run` wrapper so `hf` resolves via the venv — non-interactive SSH doesn't
# source ~/.bashrc, so bare `hf` isn't on PATH there (bit us during serve-provision).
model-pull:
    uv run hf download Qwen/Qwen2.5-VL-7B-Instruct --revision cc594898137f460bfe9f0759e9844b3ce807cfb5

# download the production backbone — AWQ int4 of the 32B VL model (~20 GB). Needs L40S 48GB.
model-pull-production:
    uv run hf download Qwen/Qwen2.5-VL-32B-Instruct-AWQ --revision 66c370b74a18e7b1e871c97918f032ed3578dfef

# ---------- housekeeping ----------

# clear caches + build artifacts
clean:
    rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
