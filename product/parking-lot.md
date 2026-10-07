# Parking Lot

Unscheduled work items — bugs, chores, ideas, research notes — captured so they don't get lost. Add items as you go; `/ds-work-parking-lot` surfaces them and suggests what to tackle.

One item per line. Tags are freeform but the conventions are:
- type: `[bug]` `[chore]` `[idea]` `[research]`
- size: `[S]` `[M]` `[L]`
- milestone link: `→M{N}` (e.g. `→M2`)

## Open

<!-- newest first; format: - [ ] YYYY-MM-DD — title  [tags] -->
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
