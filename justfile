# justfile — Bernoulli daily operations
#
# This file holds the recipes you reach for while operating the serving box:
# bringing it up, launching uvicorn, monitoring, smoke-tests, bringing it down.
# Development recipes (tests, lint, benchmarks, model pulls, docker builds, etc.)
# live in submodules — see `just dev --list` and `just bench --list`.

host := env_var_or_default("HOST", "bernoulli")

mod dev
mod bench

# Default: show the task list
default:
    @just --list

# ---------- shell + sync (daily companions to the serve-* flow) ----------

# ssh into the dev box
shell:
    ssh {{host}}

# rsync repo to the dev box (no delete)
sync:
    ./deploy/sync.sh

# rsync --delete — mirrors local state exactly (deletes remote files not in local)
sync-mirror:
    ./deploy/sync.sh --mirror

# instance state, GPU, HF cache size
status:
    @echo "--- terraform ---"
    @terraform -chdir=deploy output 2>/dev/null || echo "(not applied)"
    @echo
    @echo "--- instance ---"
    @ssh -o ConnectTimeout=5 {{host}} 'nvidia-smi | head -14; echo; df -h /opt/dlami/nvme | tail -1; echo; du -sh /opt/dlami/nvme/hf-cache 2>/dev/null' 2>/dev/null || echo "(unreachable)"

# ---------- public-endpoint lifecycle (dev-tier Salesforce-callable) ----------
# These recipes operate the g5.xlarge as a reverse-proxied public endpoint at
# https://bernoulli-api.nuwire.dev — TLS terminated on ssd2, forwarded to the
# EC2 box's ephemeral public IP (tracked via `bernoulli-internal.nuwire.dev`
# in ClouDNS, updated on each `serve-up`).
#
# Flow: `just dev up` (once) → `just serve-provision` (once per EC2 instance) →
# `just serve-up` (daily) → `just serve-start` (daily, launches uvicorn) ...
# `just serve-down` (end of day; stop also kills the process).
#
# No Docker, no systemd — minimum ceremony. The uvicorn process runs under
# nohup; if it crashes, re-run `serve-start`. For restart-on-crash + boot
# auto-start, add a systemd unit later (see M11 in product/milestones.md).
# The Dockerfile + `just dev docker-*` recipes stay for future production use.

# Stepwise, resumable bring-up for the serving box. Does the work of what
# used to be `serve-provision` + `serve-start`, with explicit short timeouts
# on each step so the UEFI-boot-loop class of failures (see
# product/reports/2026-10-18.md) fails loudly instead of silently hanging.
# Delegates to deploy/bringup.sh. Idempotent — safe to re-run on any step
# failure. BERNOULLI_API_KEYS must be in ~/.credentials.
serve-provision:
    #!/usr/bin/env bash
    set -euo pipefail
    source ~/.credentials
    ./deploy/bringup.sh

# Launch uvicorn on the box via nohup, with API keys from ~/.credentials.
# Idempotent: kills any existing uvicorn first. Returns when /healthz on the
# box responds locally (which means the model is loaded in VRAM).
# Supports BERNOULLI_HOST=<ip> override for debugging; defaults to the
# `bernoulli` ssh alias (updated by `serve-up`).
#
# All SSH calls use explicit timeouts (ConnectTimeout + overall `timeout`)
# so a stuck connection can never cost more than ~10s per probe. Loop caps
# at 3 min total.
serve-start:
    #!/usr/bin/env bash
    set -euo pipefail
    source ~/.credentials
    : "${BERNOULLI_API_KEYS:?set BERNOULLI_API_KEYS in ~/.credentials (comma-separated 64-char hex tokens)}"
    HOST="${BERNOULLI_HOST:-{{host}}}"
    SSH_OPTS="-o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2"
    echo "[1/3] stopping any existing uvicorn on $HOST..."
    timeout 15 ssh $SSH_OPTS "$HOST" 'pkill -f "uvicorn bernoulli.server" 2>/dev/null; sleep 1' || true
    echo "[2/3] seeding ~/.bernoulli.env + launching uvicorn..."
    # Write env file on the box (chmod 600) so start-serve.sh can source it.
    # Updated on every serve-start so key rotation flows through naturally.
    # start-serve.sh handles the setsid --fork + exec pattern for clean detach.
    printf 'BERNOULLI_SCORER=hf\nBERNOULLI_MAX_MODEL_LEN=8192\nBERNOULLI_API_KEYS=%s\n' "$BERNOULLI_API_KEYS" \
        | timeout 10 ssh $SSH_OPTS "$HOST" 'cat > ~/.bernoulli.env && chmod 600 ~/.bernoulli.env'
    timeout 15 ssh $SSH_OPTS "$HOST" "setsid --fork bash /home/ubuntu/bernoulli/deploy/start-serve.sh"
    sleep 2
    echo "[3/3] waiting for /healthz on the box (HFScorer loads ~729 weight shards; 4-5 min cold)..."
    for i in $(seq 1 60); do
        if timeout 8 ssh $SSH_OPTS "$HOST" 'curl -sf -m 3 http://127.0.0.1:8000/healthz' >/dev/null 2>&1; then
            echo "ready on $HOST:8000 after ${i} probes (~$((i*10))s); logs: ssh $HOST 'tail -f ~/serve.log'"
            exit 0
        fi
        sleep 10
    done
    echo "WARN: /healthz didn't come up in 10 min; check 'ssh $HOST tail ~/serve.log'"
    exit 1

# Stop uvicorn only (box stays running). Useful for restarting the server
# after a code change without a full stop/start cycle.
# `just serve-down` additionally stops the EC2 box.
serve-stop:
    timeout 15 ssh -o ConnectTimeout=10 {{host}} 'pkill -f "uvicorn bernoulli.server" && echo stopped || echo "nothing to stop"'

# Poll the public /healthz every N seconds up to T minutes; ring the terminal
# bell on success (1 ding) or timeout (3 dings). Companion to `serve-start`:
# run it in a side terminal to get an audible notification when the box is
# actually serving — cold start loads ~729 weight shards and takes 4-5 min.
# Defaults: public endpoint, 15s interval, 10m budget. Overrides:
#   just serve-monitor http://127.0.0.1:8000 10 5
serve-monitor url="https://bernoulli-api.nuwire.dev" interval="15" timeout_min="10":
    #!/usr/bin/env bash
    set -uo pipefail
    URL="{{url}}"
    INTERVAL={{interval}}
    MAX_ITERS=$(( {{timeout_min}} * 60 / INTERVAL ))
    echo "monitoring $URL/healthz every ${INTERVAL}s (timeout {{timeout_min}}m, ${MAX_ITERS} probes)..."
    for i in $(seq 1 $MAX_ITERS); do
        status=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$URL/healthz" 2>/dev/null || echo "err")
        elapsed=$(( (i - 1) * INTERVAL ))
        if [ "$status" = "200" ]; then
            printf '\a'
            echo "ok  [${elapsed}s]  $URL/healthz -> 200"
            exit 0
        fi
        echo "wait [${elapsed}s]  $URL/healthz -> $status"
        [ $i -lt $MAX_ITERS ] && sleep $INTERVAL
    done
    printf '\a\a\a'
    echo "TIMEOUT after {{timeout_min}}m — $URL/healthz still not 200"
    exit 1

# Smoke-test the running server against the request fixtures in samples/.
# Prints: /healthz probe (no auth), /v1/models probe (with auth), then each
# samples/*.json posted to /v1/decide with the response pretty-printed.
# Uses the FIRST api key from BERNOULLI_API_KEYS; `jq` is required.
#
# Default url = the ssd2-fronted public endpoint. For local-on-box testing:
#   just shell → curl http://127.0.0.1:8000/...
# or override the URL:
#   just serve-check http://127.0.0.1:8000
serve-check url="https://bernoulli-api.nuwire.dev":
    #!/usr/bin/env bash
    set -uo pipefail
    source ~/.credentials
    : "${BERNOULLI_API_KEYS:?set BERNOULLI_API_KEYS in ~/.credentials}"
    KEY=$(echo "$BERNOULLI_API_KEYS" | cut -d, -f1)
    URL="{{url}}"
    HEADER="Authorization: Bearer $KEY"
    SENTINEL="__HTTP_STATUS__"

    show() {
        # read combined output (body then one line "__HTTP_STATUS__ ...") on stdin;
        # split and print nicely.
        local combined status body
        combined=$(cat)
        status=$(printf '%s\n' "$combined" | awk -v s="$SENTINEL" '$0 ~ "^" s {sub("^" s " ",""); print; exit}')
        body=$(printf '%s\n' "$combined" | awk -v s="$SENTINEL" '$0 ~ "^" s {exit} {print}')
        echo "  http $status"
        printf '%s\n' "$body" | jq -C 2>/dev/null || printf '%s\n' "$body"
    }

    echo "=== GET /healthz (no auth) ==="
    curl -sS -w "\n${SENTINEL} %{http_code} · %{time_total}s" "$URL/healthz" | show
    echo

    echo "=== GET /v1/models (bearer auth) ==="
    curl -sS -w "\n${SENTINEL} %{http_code} · %{time_total}s" -H "$HEADER" "$URL/v1/models" | show
    echo

    for f in samples/*.json; do
        name=$(basename "$f" .json)
        echo "=== POST /v1/decide — $name ==="
        echo "--- questions ---"
        jq -r '.questions | map("  [\(.type)] \(.prompt)") | join("\n")' "$f"
        echo "--- response ---"
        curl -sS -w "\n${SENTINEL} %{http_code} · %{time_total}s" \
            -X POST -H "$HEADER" -H "content-type: application/json" \
            --data @"$f" "$URL/v1/decide" | show
        echo
    done

# Start the EC2 box, update ClouDNS A record, wait for the public endpoint
# to be reachable. Daily "come online" command.
serve-up:
    #!/usr/bin/env bash
    set -euo pipefail
    INSTANCE_ID=$(terraform -chdir=deploy output -raw instance_id)
    echo "[1/5] starting ec2 $INSTANCE_ID ..."
    aws ec2 start-instances --instance-ids $INSTANCE_ID > /dev/null
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID
    echo "[2/5] waiting for AWS health checks (both system + instance; this is what catches UEFI boot loops) ..."
    aws ec2 wait instance-status-ok --instance-ids $INSTANCE_ID
    IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    echo "[3/5] got ip $IP (both AWS health checks passed)"
    echo "[4/5] updating cloudns: bernoulli-internal.nuwire.dev -> $IP"
    source ~/.credentials
    ~/bin/cloudns_mgr set bernoulli-internal nuwire.dev $IP > /dev/null
    # keep local ssh config in sync so `just shell` + bringup.sh work
    python3 -c "import re,os; p=os.path.expanduser('~/.ssh/config'); c=open(p).read(); open(p,'w').write(re.sub(r'(Host bernoulli\s*\n\s*HostName )\S+', r'\g<1>$IP', c, count=1))"
    echo "[5/5] box is up and healthy. Run 'just serve-provision' (first time per instance) or 'just serve-start' (if already provisioned)."

# Stop the EC2 box. Blocks until AWS confirms "stopped" state — which is
# the state at which compute billing stops. After that we double-check with
# a describe-instances call. Rings 1 bell on success, 3 on timeout.
# The ClouDNS record stays stale (nothing reaches a dead backend anyway);
# next serve-up overwrites it. EBS (~$8/mo for 80GB gp3) keeps accruing.
serve-down:
    #!/usr/bin/env bash
    set -euo pipefail
    INSTANCE_ID=$(terraform -chdir=deploy output -raw instance_id)
    echo "[1/3] requesting stop of $INSTANCE_ID ..."
    aws ec2 stop-instances --instance-ids $INSTANCE_ID > /dev/null
    echo "[2/3] waiting for 'stopped' state (compute billing ends here; typical 2-5 min)..."
    if ! timeout 600 aws ec2 wait instance-stopped --instance-ids $INSTANCE_ID; then
        printf '\a\a\a'
        STATE=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
            --query 'Reservations[0].Instances[0].State.Name' --output text)
        echo "WARN: didn't reach 'stopped' in 10 min; current state: $STATE"
        echo "       check the console; may need manual intervention."
        exit 1
    fi
    STATE=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].State.Name' --output text)
    echo "[3/3] confirmed state: $STATE (no further compute charges; EBS ~\$8/mo continues)"
    printf '\a'

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

# ---------- web (landing page deploy) ----------

# deploy the bernoulli.live landing page (delegates to web/justfile)
deploy-web:
    cd web && just deploy
