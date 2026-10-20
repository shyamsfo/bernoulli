# bench.just — evaluation + benchmark recipes (invoked as `just bench <name>`)
# Imported as a module by the root justfile. Covers:
# - eval harness (per-dataset reports under evals/reports/, calibration fitting)
# - benchmark suites (academic + use-case sweeps that compare bernoulli to
#   generative / deberta / bge-m3-lr baselines, results under benchmarks/<spec>/results/)
# - load-test the running HTTP API

host := env_var_or_default("HOST", "bernoulli")

# ---------- eval harness (pre-M8 — per-dataset snapshot reports) ----------

# run one dataset end-to-end; usage: just bench eval sst2 reverse
eval dataset debias="reverse":
    uv run python -m evals.run_eval --dataset {{dataset}} --debias {{debias}} --out evals/reports/{{dataset}}.md

# fit calibration from one or more eval sidecars; usage: just bench fit-cal evals/reports/sst2.json
fit-cal +sidecars:
    uv run python -m evals.fit_calibration --from {{sidecars}} --out calibration/qwen2.5-vl-7b.json

# run one dataset with calibration applied; usage: just bench eval-calibrated sst2
eval-calibrated dataset debias="reverse" cal="calibration/qwen2.5-vl-7b.json":
    uv run python -m evals.run_eval --dataset {{dataset}} --debias {{debias}} --calibrate {{cal}} --out evals/reports/{{dataset}}.calibrated.md

# pull eval reports from the dev box back into the repo
reports-pull:
    scp '{{host}}:~/bernoulli/evals/reports/*.md' '{{host}}:~/bernoulli/evals/reports/*.json' evals/reports/

# ---------- benchmark sweep (M8+ — baselines matrix, server-backed) ----------

# Run one benchmark through benchmarks.run with the current eval-only defaults.
# Usage: just bench benchmark academic/sst2 bernoulli,generative
benchmark spec="academic/sst2" baselines="bernoulli" limit="" train_limit="2000" url="http://127.0.0.1:8000":
    uv run python -m benchmarks.run {{spec}} \
        --baselines {{baselines}} \
        {{ if limit == "" { "" } else { "--limit " + limit } }} \
        --train-limit {{train_limit}} \
        --bernoulli-url {{url}}

# Sweep the academic suite × the applicable baselines. One per-benchmark failure
# doesn't stop the rest; stdout gets a success/fail summary at the end.
# Default --limit 500 keeps a full-matrix run under ~2h on g5.xlarge/7B.
# Override with: just bench sweep-academic 2000 (or "" for no cap).
# Requires the server running at :8000 (via root `just serve-start` on the box,
# or `just dev serve-local` for a local run) — both `bernoulli` and `generative`
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

# ---------- load test ----------

# load test the serving HTTP API; usage: just bench loadtest  (needs the server up)
loadtest url="http://127.0.0.1:8000":
    uv run python -m evals.loadtest --url {{url}} --out evals/reports/loadtest.md
