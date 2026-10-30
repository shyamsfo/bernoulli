# Scaling options — "the product got popular, what now?"

**Status:** research, not a decision. Captures the thinking from a 2026-10-20 conversation about what we'd do when dozens of users start hitting the single `g5.xlarge` serving box. Feeds M4e+ (vLLM step-up) and whatever future hardening / HA milestone covers multi-node serving.

## Framing

Today's serving stack is one g5.xlarge (A10G 24 GB) running `BERNOULLI_SCORER=hf` behind ssd2 nginx. Requests are serialized through the GPU — no continuous batching, no prefix caching. Fine for M4-era "the thing works end-to-end" load; not fine for dozens of concurrent users.

"Dozens of users" is a vague bar. Concretely:
- **20 users × 1 QPS each = 20 QPS** sustained, which the HFScorer path cannot carry (each `/v1/decide` is N forward passes serialised; realistic sustained throughput is maybe 2-5 QPS on this box).
- **20 users doing an occasional request** (burstier, maybe 1-5 QPS average) is more realistic for the product, and sits right at the edge of what today's box can serve.

So the question is really: what do we reach for first, and what do we reserve for later?

## Options (in order of effort)

### Option 1 — Flip the scorer to vLLM

**Shape.** `BERNOULLI_SCORER=vllm` instead of `hf`. The Scorer protocol was designed for this swap (decisions log 2026-10-05: "scorer dispatch via `load_scorer(settings)` factory"). Same box, same model, same API.

**Why it helps.** vLLM gives us two throughput multipliers for free:
- **Continuous batching.** Multiple in-flight requests get merged into shared forward passes. Instead of request-at-a-time GPU utilization, we approach saturated utilization. Typically **5–10×** throughput on bursty workloads.
- **Prefix caching.** The system prompt / scaffolding tokens get cached once across all requests. For our shape (short response distribution, longer prompt scaffolding), hit rate should be very high.

**Effort.** Already planned as part of M4+. The scorer code exists. The Docker image is built. The main unresolved bit is the single-GPU vLLM config for our model + max-model-len. Realistically 1-2 days of focused work.

**Risks.**
- **Cold start.** vLLM startup is ~90s vs. HFScorer's ~4-5 min — net win. But crashes in the engine subprocess leave a zombie holding ~18 GB VRAM (see CLAUDE.md Gotchas). Operational hazard, not a correctness one.
- **Max model length tradeoff.** On 24 GB VRAM, we have to drop from 32768 → 8192 to leave room for KV cache. Already captured in Gotchas. For our prompt sizes (dozens-to-hundreds of tokens), 8192 is comfortable.

**Verdict.** This is the first lever and probably the only one needed for the "dozens of users" bar. Do this before considering any other scaling move.

---

### Option 2 — Scale up (bigger single box)

**Shape.** Keep the single-box architecture; move to a bigger GPU or multi-GPU instance.

Three realistic variants:
- **`g6e.xlarge` (L40S 48 GB).** Already the planned production backbone target. Lets us run the 32B AWQ model (better quality) *and* keep a bigger KV cache for more concurrency. ~$1.86/hr on-demand. This is the step we're already waiting on for M4e, pending capacity.
- **`g5.12xlarge` or `g6.12xlarge` (4× A10G / 4× L4).** Tensor parallelism via vLLM across the 4 GPUs. Near-linear throughput scaling. ~$5-7/hr. Overkill for "dozens of users" but the first option if we need materially more capacity than a single-GPU box.
- **`p4d.24xlarge` and above.** Not relevant at this scale. Mentioned for completeness.

**Why it helps.**
- **One DNS record, one `serve-up`, one `serve-info`.** The whole ops story stays identical to today. Scaling up is literally changing a terraform variable + re-provisioning.
- **Prefix cache stays warm across all requests.** This is subtle but matters — in a multi-node setup, requests get sharded across nodes, which halves (or worse) the prefix cache hit rate per node.
- **Model-quality step comes for free.** The g6e.xlarge step is the one we need for the 32B production backbone anyway. If scaling and quality both get solved by one move, that's the right move.

**Effort.** Low. The provisioning scripts handle any single-box instance type (via the `availability_zone` + `instance_type` variables). Multi-GPU adds a vLLM config knob (`tensor_parallel_size=N`) but no fundamental change.

**Risks.**
- **Still one point of failure.** Instance crashes → everyone sees downtime. Fine while we're in "friends-and-nerds + early customers" mode, not fine when there's an uptime SLO.
- **Capacity.** We've been bitten by `InsufficientInstanceCapacity` on g6/g6e across all us-east-1 AZs. If we commit to a bigger box and AWS doesn't have one, we're stuck. Mitigated by multi-AZ-fallback logic (already in `variables.tf`) but still a real failure mode.

**Verdict.** This is the second lever after vLLM. Do it when (a) vLLM alone can't carry sustained load, or (b) we want to step up to the 32B production backbone for quality reasons (M4e), whichever comes first.

---

### Option 3 — Scale out (N nodes behind nginx LB)

**Shape.** Multiple EC2 instances, each running the full stack, with ssd2's nginx fronting a multi-upstream block instead of a single `bernoulli-internal.nuwire.dev` resolver. The gr-slots config already does this exact pattern across `gr1/gr2/gr3`, so the nginx side is a well-trodden path.

**Why it helps.**
- **Horizontal headroom.** Add nodes until throughput meets demand.
- **Rolling upgrades.** Can take a node out, re-provision, bring it back without downtime.
- **Redundancy.** A single instance crash degrades to N-1 capacity rather than full outage.

**Effort.** High. Each new node needs:
- Its own terraform instance (easy — one more `aws_instance` resource).
- The model cache (hard — NVMe is ephemeral and `bringup.sh` is written for the single-box case; we'd need either to pre-bake the cache into an AMI, or accept 90s model-pull on every node start).
- Its own DNS record *or* internal LB membership (not hard, but new).
- Monitoring / health checks across N nodes rather than one.

Plus ssd2 nginx:
- Multi-upstream block + health check pattern. Easy.

**Risks / downsides.**
- **Prefix cache fragmentation.** vLLM's prefix cache is per-process. N nodes = each node sees ~1/N of the request stream, so cache hit rate per node drops. The throughput multiplier from Option 1 shrinks the more we shard.
- **Operational complexity.** Debugging "which node served this request" is a thing we don't have today. Any scorer-level state (not currently a problem, but something to watch) becomes a cross-node coordination problem.
- **Cost scales linearly.** 2 nodes = 2× hourly bill. Vertical scaling (Option 2) typically gets better $/throughput at this scale.

**Verdict.** Reserve for when we have (a) load that genuinely exceeds a single big box, or (b) an uptime SLO that one box can't meet. For "dozens of users," neither applies. Revisit when the product has an actual availability commitment.

---

## The main tradeoff

**Ops simplicity vs. reliability / headroom.** Today's one-box story is pleasant: one `serve-up`, one DNS record, one place to look at logs, one GPU to check. Scaling up preserves that. Scaling out breaks it — in exchange for horizontal headroom and redundancy we don't currently need.

The practical path is almost certainly:

1. **vLLM** (planned, free throughput multiplier)
2. **Bigger box** (planned anyway for M4e / quality step-up)
3. **Horizontal only when forced** (SLO or single-box capacity ceiling)

Steps 1 and 2 are already in the roadmap for other reasons. Step 3 is a response to a problem we don't have yet.

## Things that would change this analysis

- **A real uptime SLO.** "99.9% monthly" means a single box can't be the whole stack — one AWS maintenance event busts the budget. Would push us to Option 3 regardless of load.
- **Multi-tenant workloads with wildly different prompt shapes.** Would hurt vLLM's prefix cache hit rate even on a single box, reducing Option 1's gain.
- **A dramatic model size step.** 70B+ models don't fit on a single consumer-class GPU and force tensor parallelism (Option 2 multi-GPU) or distributed inference.
- **Spot pricing becoming attractive.** If we accept interruptions, a primary + spot standby pattern becomes cheaper per throughput than scaling up. Not relevant until we have a real spend to optimise.
