# Parking Lot

Unscheduled work items — bugs, chores, ideas, research notes — captured so they don't get lost. Add items as you go; `/ds-work-parking-lot` surfaces them and suggests what to tackle.

One item per line. Tags are freeform but the conventions are:
- type: `[bug]` `[chore]` `[idea]` `[research]`
- size: `[S]` `[M]` `[L]`
- milestone link: `→M{N}` (e.g. `→M2`)

## Open

<!-- newest first; format: - [ ] YYYY-MM-DD — title  [tags] -->
<!-- Resolved 2026-10-08: adapter's score-task legend was 0-indexed while the RatingQuestion(scale=(1,n)) made the server render options as `A) 1`, `B) 2`, ..., `D) n`. Model matched legend-digit to rendered-option-digit, picked the letter one below correct; _shape_probs then shifted the 1-based response back to 0-based so the answer landed systematically one below `expected`. Fix was a one-liner: 1-index the legend. Ordinal on standard tier: 4/12 → 12/12. Standard tier overall: 0.833 → 0.944. Weighted 231-cohort: 0.675 → 0.706. -->
<!-- - [ ] 2026-10-08 — JevBench **ordinal** family (score-type tasks)... [bug] [S] →M10 -->

- [ ] 2026-10-29 — `deploy/start-serve.sh` truncates `~/serve.log` on every launch (uses `>` not `>>`). Every `just serve-start` wipes history; a 422 or OOM from before the restart is unrecoverable. Fix is one character + logrotate so the file doesn't grow unbounded (daily, gzip, keep ~30 days). Caught this trying to investigate three 422s from the Chrome extension — context was gone. [bug] [S] →M15

- [ ] 2026-10-29 — Something on ssd2 (208.87.128.216) sends constant unauth'd `GET /` to `bernoulli-api.nuwire.dev`, each one logging a 401 in serve.log. Suspect `uptime_kuma` or similar probing the root instead of `/healthz`. Fix: find the ssd2 config and point it at `/healthz` (which is auth-exempt). No functional impact, just log noise that'd obscure real traffic once the event log lands. [chore] [S] →M15

- [ ] 2026-10-29 — `just serve-stop` recipe's `pkill -f "uvicorn bernoulli.server"` matches its own ssh wrapper command string, so after killing uvicorn successfully it also kills the ssh session itself and the recipe exits 255. Benign (uvicorn is actually gone), but it looks like a failure. Fix: use `pkill -f "uvicorn bernoulli.server:app"` (narrower match) or use a pidfile. [bug] [S]

- [ ] 2026-10-18 — Mirror the model snapshot (Qwen2.5-VL-7B-Instruct @ cc59489...) to an S3 bucket + add an S3-pull path to `deploy/bringup.sh` that falls back to `hf download` on cache miss. Payoff: fresh-instance provisioning drops from ~2:30 (HF download) to ~20-30s (same-region S3 → EC2). Pick up when `terraform destroy && apply` cycles get painful (production/M11, or if we move to spot instances AWS can reclaim), or if HF Hub ever has an outage or rate-limits us mid-session. Setup: 15 min (bucket + one-time upload + IAM role + ~10 line bringup.sh edit). Ongoing cost: ~$0.35/mo for 15 GB S3 standard storage, zero same-region egress.  [chore] [S] →M11

- [ ] 2026-10-07 — Chunked scoring is **not reorder-stable** on Banking77. Smoke sweep at N=20 showed `stability_reorder = 0.000` — every single example's top-1 moved when options were permuted. Expected architecturally: chunked aggregation depends on which 26-letter chunk each option falls into, so a reorder shuffles those chunks and the global softmax changes. Tied to the pre-existing "chunked + debias is awkward" open question in `CLAUDE.md` decisions log. Fix options: (a) report chunked-scoring reorder-stability as `n/a` on the leaderboard with a footnote, same as the binary-reorder convention; (b) implement a cross-chunk debias (compute logits per chunk, aggregate without chunking); (c) use tournament rather than independent chunks. (a) is the honest short-term; (b) is the structural fix. [bug] [M] →M8
<!-- Resolved 2026-10-07: `/v1/generate` server endpoint added, `Generative` is now an HTTP baseline sharing the server's loaded scorer. See bernoulli/generative.py + bernoulli/server.py. -->
<!-- - [ ] 2026-10-06 — Generative baseline shares the GPU with the running server. [bug] [M] →M8 -->

- [ ] 2026-10-05 — Audio modality — not in vision_and_roadmap.md (which defines state as text + images only). If we want it later, scope and add as its own milestone. [idea] [L]
- [ ] 2026-10-05 — Revisit CI story (add non-GPU GHA workflow for ruff + mypy) if we add a code reviewer or need PR status checks. Keep bernoulli as the real test runner.  [chore] [S]
- [ ] 2026-10-05 — Add a periodic `terraform destroy && apply` smoke test to prove `user_data.sh` is reproducible after any AMI/driver drift  [chore] [S]
- [ ] 2026-10-05 — Compare candidate backbones (Qwen3.6-35B-A3B MoE, Qwen3.6-27B dense, Qwen3.5-VL small, alternative open VLMs) on calibration behavior at quantized precisions  [research] [L] →M4
- [ ] 2026-10-05 — Decide dev-tier backbone (smallest VLM runnable on 16–24GB GPU or CPU) and record revision hash  [research] [S] →M1
- [ ] 2026-10-05 — Investigate hierarchical / tournament scoring for >26-option choices (Banking77 is the test case)  [research] [M] →M3
- [ ] 2026-10-05 — Confirm `allowed_token_ids` + prefix caching compatibility on candidate backbone × pinned vLLM version  [research] [M] →M4
- [ ] 2026-10-05 — Quantization vs calibration drift: does int4/int8 shift ECE meaningfully?  [research] [M] →M4

## Done / Dropped (last 10)

<!-- newest first; auto-trimmed to 10 -->
