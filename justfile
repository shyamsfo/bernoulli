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
# `uv run` wrapper so `hf` resolves via the venv — non-interactive SSH doesn't
# source ~/.bashrc, so bare `hf` isn't on PATH there (bit us during serve-provision).
model-pull:
    uv run hf download Qwen/Qwen2.5-VL-7B-Instruct --revision cc594898137f460bfe9f0759e9844b3ce807cfb5

# download the production backbone — AWQ int4 of the 32B VL model (~20 GB). Needs L40S 48GB.
model-pull-production:
    uv run hf download Qwen/Qwen2.5-VL-32B-Instruct-AWQ --revision 66c370b74a18e7b1e871c97918f032ed3578dfef

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
# Requires `just serve` already up at :8000 — both `bernoulli` and `generative`
# baselines call it (via /v1/decide and /v1/generate respectively).
sweep-academic limit="500" train_limit="2000" url="http://127.0.0.1:8000":
    #!/usr/bin/env bash
    set -u
    declare -A matrix=(
        ["academic/sst2"]="bernoulli,generative,deberta,bge-m3-lr"
        ["academic/ag_news"]="bernoulli,generative,deberta,bge-m3-lr"
        ["academic/banking77"]="bernoulli,generative,deberta,bge-m3-lr"
        ["academic/tweeteval_emotion"]="bernoulli,generative,deberta,bge-m3-lr"
        ["academic/paws"]="bernoulli,generative,bge-m3-lr"
        ["academic/arxiv_post_cutoff"]="bernoulli,generative,deberta"
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
sweep-usecase limit="500" train_limit="2000" url="http://127.0.0.1:8000":
    #!/usr/bin/env bash
    set -u
    declare -A matrix=(
        ["guardrails/wildguard_test"]="bernoulli,generative,bge-m3-lr"
        ["guardrails/toxicchat"]="bernoulli,generative,bge-m3-lr"
        ["guardrails/xstest"]="bernoulli,generative"
        ["triage/clinc150_oos"]="bernoulli,generative,bge-m3-lr"
        ["ratings/yelp_stars"]="bernoulli,generative,bge-m3-lr"
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

# ---------- public-endpoint lifecycle (dev-tier Salesforce-callable) ----------
# These recipes operate the g5.xlarge as a reverse-proxied public endpoint at
# https://bernoulli-api.nuwire.dev — TLS terminated on ssd2, forwarded to the
# EC2 box's ephemeral public IP (tracked via `bernoulli-internal.nuwire.dev`
# in ClouDNS, updated on each `serve-up`).
#
# Flow: `just up` (once) → `just serve-provision` (once per EC2 instance) →
# `just serve-up` (daily) → `just serve-start` (daily, launches uvicorn) ...
# `just serve-down` (end of day; stop also kills the process).
#
# No Docker, no systemd — minimum ceremony. The uvicorn process runs under
# nohup; if it crashes, re-run `serve-start`. For restart-on-crash + boot
# auto-start, add a systemd unit later (see M11 in product/milestones.md).
# The Dockerfile + `just docker-*` recipes stay for future production use.

# One-time per EC2 instance: sync code, uv sync, pull model.
# Fast enough to re-run safely (uv sync + hf download are both idempotent).
# After a `terraform destroy && apply` or a fresh instance, run this again.
serve-provision:
    #!/usr/bin/env bash
    set -euo pipefail
    just sync-mirror
    echo "---uv sync (installs torch + transformers etc.)---"
    ssh {{host}} 'cd ~/bernoulli && uv sync --extra scorer-hf'
    echo "---model-pull (Qwen2.5-VL-7B, ~15 GB, ~90s via hf_transfer)---"
    ssh {{host}} 'cd ~/bernoulli && just model-pull'
    echo "---done. run 'just serve-start' to launch the server.---"

# Launch uvicorn on the box via nohup, with API keys from ~/.credentials.
# Idempotent: kills any existing uvicorn first. Returns when /healthz on the
# box responds locally (which means the model is loaded in VRAM).
# Supports BERNOULLI_HOST=<ip> override for debugging; defaults to the
# `bernoulli` ssh alias (updated by `serve-up`).
serve-start:
    #!/usr/bin/env bash
    set -euo pipefail
    source ~/.credentials
    : "${BERNOULLI_API_KEYS:?set BERNOULLI_API_KEYS in ~/.credentials (comma-separated 64-char hex tokens)}"
    HOST="${BERNOULLI_HOST:-{{host}}}"
    echo "[1/3] stopping any existing uvicorn on $HOST..."
    ssh "$HOST" 'pkill -f "uvicorn bernoulli.server" 2>/dev/null; sleep 1; echo stopped' || true
    echo "[2/3] starting uvicorn via nohup..."
    # Env vars inline so BERNOULLI_API_KEYS never lands on disk on the box.
    # Logs go to ~/serve.log on the box.
    ssh "$HOST" "cd ~/bernoulli && \
        BERNOULLI_SCORER=hf \
        BERNOULLI_MAX_MODEL_LEN=8192 \
        BERNOULLI_API_KEYS='$BERNOULLI_API_KEYS' \
        nohup uv run uvicorn bernoulli.server:app --host 0.0.0.0 --port 8000 \
            > ~/serve.log 2>&1 </dev/null &
        disown
        echo 'uvicorn backgrounded; PID follows:'
        pgrep -f 'uvicorn bernoulli.server' | head -1"
    echo "[3/3] waiting for /healthz on the box (model load takes ~90s)..."
    for i in $(seq 1 60); do
        if ssh "$HOST" 'curl -sf http://127.0.0.1:8000/healthz' >/dev/null 2>&1; then
            echo "ready on $HOST:8000; logs: ssh $HOST 'tail -f ~/serve.log'"
            exit 0
        fi
        sleep 5
    done
    echo "WARN: /healthz didn't come up in 5 min; check 'ssh $HOST tail ~/serve.log'"
    exit 1

# Stop uvicorn only (box stays running). Useful for restarting the server
# after a code change without a full stop/start cycle.
# `just serve-down` additionally stops the EC2 box.
serve-stop:
    ssh {{host}} 'pkill -f "uvicorn bernoulli.server" && echo stopped || echo "nothing to stop"'

# Start the EC2 box, update ClouDNS A record, wait for the public endpoint
# to be reachable. Daily "come online" command.
serve-up:
    #!/usr/bin/env bash
    set -euo pipefail
    INSTANCE_ID=$(terraform -chdir=deploy output -raw instance_id)
    echo "[1/4] starting ec2 $INSTANCE_ID ..."
    aws ec2 start-instances --instance-ids $INSTANCE_ID > /dev/null
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID
    IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    echo "[2/4] got ip $IP"
    echo "[3/4] updating cloudns: bernoulli-internal.nuwire.dev -> $IP"
    source ~/.credentials
    ~/bin/cloudns_mgr set bernoulli-internal nuwire.dev $IP
    # keep local ssh config in sync so `just shell` etc. still work
    python3 -c "import re,os; p=os.path.expanduser('~/.ssh/config'); c=open(p).read(); open(p,'w').write(re.sub(r'(Host bernoulli\s*\n\s*HostName )\S+', r'\g<1>$IP', c, count=1))"
    echo "[4/4] waiting for public endpoint to come up (nginx re-resolves DNS every 30s + container restarts via docker policy)..."
    for i in $(seq 1 60); do
        if curl -sf https://bernoulli-api.nuwire.dev/healthz >/dev/null 2>&1; then
            echo "ready: https://bernoulli-api.nuwire.dev"
            exit 0
        fi
        sleep 5
    done
    echo "WARN: public endpoint didn't come up in 5 min; try 'just serve-status'"
    exit 1

# Stop the EC2 box. The ClouDNS record stays stale (no need to clean up —
# nothing can reach a dead backend anyway). Next `serve-up` overwrites it.
serve-down:
    #!/usr/bin/env bash
    set -euo pipefail
    INSTANCE_ID=$(terraform -chdir=deploy output -raw instance_id)
    aws ec2 stop-instances --instance-ids $INSTANCE_ID > /dev/null
    echo "stopped; run 'just serve-up' to bring back"

# EC2 state + current DNS value + endpoint health in one command.
serve-status:
    #!/usr/bin/env bash
    set -euo pipefail
    INSTANCE_ID=$(terraform -chdir=deploy output -raw instance_id)
    STATE=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].State.Name' --output text)
    echo "ec2:       $STATE"
    source ~/.credentials 2>/dev/null || true
    DNS=$(~/bin/cloudns_mgr get bernoulli-internal nuwire.dev 2>/dev/null | head -1 || echo "none")
    echo "dns:       bernoulli-internal.nuwire.dev -> $DNS"
    if curl -sf https://bernoulli-api.nuwire.dev/healthz >/dev/null 2>&1; then
        echo "healthz:   ok (https://bernoulli-api.nuwire.dev)"
    else
        echo "healthz:   unreachable"
    fi
