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

# start the FastAPI server (uvicorn); listens on 127.0.0.1:8000 by default
serve host="127.0.0.1" port="8000":
    uv run uvicorn bernoulli.server:app --host {{host}} --port {{port}}

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

# clear caches + build artifacts
clean:
    rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# download the dev-tier backbone to the NVMe cache (~15 GB, ~90 s via hf_transfer)
model-pull:
    hf download Qwen/Qwen2.5-VL-7B-Instruct --revision cc594898137f460bfe9f0759e9844b3ce807cfb5

# download the production backbone — AWQ int4 of the 32B VL model (~20 GB). Needs L40S 48GB.
model-pull-production:
    hf download Qwen/Qwen2.5-VL-32B-Instruct-AWQ --revision 66c370b74a18e7b1e871c97918f032ed3578dfef

# ---------- eval shortcuts ----------

# run one dataset end-to-end; usage: just eval sst2 reverse
eval dataset debias="reverse":
    uv run python -m evals.run_eval --dataset {{dataset}} --debias {{debias}} --out evals/reports/{{dataset}}.md

# fit calibration from one or more eval sidecars; usage: just fit-cal evals/reports/sst2.json
fit-cal +sidecars:
    uv run python -m evals.fit_calibration --from {{sidecars}} --out calibration/qwen2.5-vl-7b.json

# run one dataset with calibration applied; usage: just eval-calibrated sst2
eval-calibrated dataset debias="reverse" cal="calibration/qwen2.5-vl-7b.json":
    uv run python -m evals.run_eval --dataset {{dataset}} --debias {{debias}} --calibrate {{cal}} --out evals/reports/{{dataset}}.calibrated.md

# pull eval reports from the dev box back into the repo
reports-pull:
    scp '{{host}}:~/bernoulli/evals/reports/*.md' '{{host}}:~/bernoulli/evals/reports/*.json' evals/reports/

# ---------- benchmark sweep (M8 task 12b) ----------

# Run one benchmark through benchmarks.run with the current eval-only defaults.
# Usage: just benchmark academic/sst2 bernoulli,generative
benchmark spec="academic/sst2" baselines="bernoulli" limit="" train_limit="2000" url="http://127.0.0.1:8000":
    uv run python -m benchmarks.run {{spec}} \
        --baselines {{baselines}} \
        {{ if limit == "" { "" } else { "--limit " + limit } }} \
        --train-limit {{train_limit}} \
        --bernoulli-url {{url}}

# Sweep the academic suite × the applicable baselines. One per-benchmark failure
# doesn't stop the rest; stdout gets a success/fail summary at the end.
# Default --limit 500 keeps a full-matrix run under ~2h on g5.xlarge/7B.
# Override with: just sweep-academic 2000 (or "" for no cap).
# Requires `just serve` or docker-run already up at :8000 for the bernoulli baseline.
# NOTE: `generative` is intentionally excluded from the default matrix on a
# single-GPU box — it loads a second copy of the backbone and OOMs against the
# server's copy. See parking-lot 2026-10-06. Re-add once an HTTP /v1/generate
# endpoint lands.
sweep-academic limit="500" train_limit="2000" url="http://127.0.0.1:8000":
    #!/usr/bin/env bash
    set -u
    declare -A matrix=(
        ["academic/sst2"]="bernoulli,deberta,bge-m3-lr"
        ["academic/ag_news"]="bernoulli,deberta,bge-m3-lr"
        ["academic/banking77"]="bernoulli,deberta,bge-m3-lr"
        ["academic/tweeteval_emotion"]="bernoulli,deberta,bge-m3-lr"
        ["academic/paws"]="bernoulli,bge-m3-lr"
        ["academic/arxiv_post_cutoff"]="bernoulli,deberta"
    )
    succeeded=()
    failed=()
    for spec in "${!matrix[@]}"; do
        baselines="${matrix[$spec]}"
        echo "[sweep] ----- $spec | $baselines -----"
        limit_arg=""
        if [ -n "{{limit}}" ]; then limit_arg="--limit {{limit}}"; fi
        if uv run python -m benchmarks.run "$spec" --baselines "$baselines" $limit_arg \
            --train-limit {{train_limit}} --bernoulli-url {{url}}; then
            succeeded+=("$spec")
        else
            echo "[sweep] FAILED: $spec" >&2
            failed+=("$spec")
        fi
    done
    echo
    echo "[sweep] succeeded (${#succeeded[@]}): ${succeeded[*]:-none}"
    echo "[sweep] failed    (${#failed[@]}): ${failed[*]:-none}"
    [ ${#failed[@]} -eq 0 ]

# Sweep the M9 use-case benchmarks. Same pattern as sweep-academic.
# `generative` excluded for the same single-GPU reason — see sweep-academic note.
sweep-usecase limit="500" train_limit="2000" url="http://127.0.0.1:8000":
    #!/usr/bin/env bash
    set -u
    declare -A matrix=(
        ["guardrails/wildguard_test"]="bernoulli,bge-m3-lr"
        ["guardrails/toxicchat"]="bernoulli,bge-m3-lr"
        ["guardrails/xstest"]="bernoulli"
        ["triage/clinc150_oos"]="bernoulli,bge-m3-lr"
        ["ratings/yelp_stars"]="bernoulli,bge-m3-lr"
    )
    succeeded=()
    failed=()
    for spec in "${!matrix[@]}"; do
        baselines="${matrix[$spec]}"
        echo "[sweep-usecase] ----- $spec | $baselines -----"
        limit_arg=""
        if [ -n "{{limit}}" ]; then limit_arg="--limit {{limit}}"; fi
        if uv run python -m benchmarks.run "$spec" --baselines "$baselines" $limit_arg \
            --train-limit {{train_limit}} --bernoulli-url {{url}}; then
            succeeded+=("$spec")
        else
            echo "[sweep-usecase] FAILED: $spec" >&2
            failed+=("$spec")
        fi
    done
    echo
    echo "[sweep-usecase] succeeded (${#succeeded[@]}): ${succeeded[*]:-none}"
    echo "[sweep-usecase] failed    (${#failed[@]}): ${failed[*]:-none}"
    [ ${#failed[@]} -eq 0 ]

# Pull benchmark results back into the repo.
benchmarks-pull:
    rsync -av '{{host}}:~/bernoulli/benchmarks/' benchmarks/ \
        --include='*/' --include='results/**' --exclude='*'

# load test the serving HTTP API; usage: just loadtest  (needs the server up)
loadtest url="http://127.0.0.1:8000":
    uv run python -m evals.loadtest --url {{url}} --out evals/reports/loadtest.md

# deploy the bernoulli.live landing page (delegates to web/justfile)
deploy-web:
    cd web && just deploy
