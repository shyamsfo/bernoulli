# justfile — Bernoulli dev workflow

host := env_var_or_default("HOST", "bernoulli")

# Default: show the task list
default:
    @just --list

# ---------- local-only targets ----------

# rsync repo to the dev box (no delete)
sync:
    ./deploy/sync.sh

# rsync --delete — mirrors local state exactly (deletes remote files not in local)
sync-mirror:
    ./deploy/sync.sh --mirror

# ssh into the dev box
shell:
    ssh {{host}}

# terraform apply in deploy/
up:
    terraform -chdir=deploy init
    terraform -chdir=deploy apply

# terraform destroy in deploy/ (confirms)
down:
    terraform -chdir=deploy destroy

# instance state, GPU, HF cache size
status:
    @echo "--- terraform ---"
    @terraform -chdir=deploy output 2>/dev/null || echo "(not applied)"
    @echo
    @echo "--- instance ---"
    @ssh -o ConnectTimeout=5 {{host}} 'nvidia-smi | head -14; echo; df -h /opt/dlami/nvme | tail -1; echo; du -sh /opt/dlami/nvme/hf-cache 2>/dev/null' 2>/dev/null || echo "(unreachable)"

# ---------- remote-safe targets (run via `just shell` or `ssh bernoulli`) ----------

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

# clear caches + build artifacts
clean:
    rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# download the dev-tier backbone to the NVMe cache (~15 GB, ~90 s via hf_transfer)
model-pull:
    hf download Qwen/Qwen2.5-VL-7B-Instruct --revision cc594898137f460bfe9f0759e9844b3ce807cfb5

# ---------- eval shortcuts ----------

# run one dataset end-to-end; usage: just eval sst2 reverse
eval dataset debias="reverse":
    uv run python -m evals.run_eval --dataset {{dataset}} --debias {{debias}} --out evals/reports/{{dataset}}.md

# pull eval reports from the dev box back into the repo
reports-pull:
    scp '{{host}}:~/bernoulli/evals/reports/*.md' '{{host}}:~/bernoulli/evals/reports/*.json' evals/reports/
