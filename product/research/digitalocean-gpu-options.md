# DigitalOcean GPU options for Bernoulli

**Status**: research (findings as of 2026-10-08).
**Feeds**: M4e (production backbone step-up — currently parked on AWS g6e capacity) and M11 (public verification endpoint — infra shape still open).

## Why look at DO

M4e has been parked for weeks because AWS g6e.xlarge capacity in us-east-1 is tight across every AZ we've tried. The L40S 48 GB GPU is the right silicon for `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` (AWQ-int4 ≈ 20 GB, fits L40S with 25 GB+ headroom for KV cache + batching). DO's catalog lists an equivalent SKU; a different cloud with different capacity dynamics is a legitimate backup.

The quick ask: can we spin up an L40S 48 GB on DO *today*, and at what price vs the AWS equivalent?

## Findings

Enumerated `/v2/sizes` with the account token on 2026-10-08. 19 GPU-flagged SKUs listed. Headlines:

| What it is | DO SKU | $/hr | $/mo | Available to this account? |
|---|---|---:|---:|---|
| **L40S 48 GB** (matches planned AWS g6e.xlarge) | `gpu-l40sx1-48gb` | $1.57 | $1,168 | ❌ listed, 0 regions |
| RTX 6000 Ada 48 GB | `gpu-6000adax1-48gb` | $1.57 | $1,168 | ❌ listed, 0 regions |
| RTX 4000 Ada 20 GB (tighter than our A10G) | `gpu-4000adax1-20gb` | $0.76 | $565 | ❌ listed, 0 regions |
| H100 80 GB | `gpu-h100x1-80gb` | $4.41 | $3,281 | ❌ listed, 0 regions |
| H200 141 GB | `gpu-h200x1-141gb` | $4.47 | $3,326 | ❌ listed, 0 regions |
| B300 288 GB (Blackwell, spot) | `gpu-b300x1-288gb-spot` | $8.00 | $5,952 | ❌ listed, 0 regions |
| **AMD MI325 256 GB** | `gpu-mi325x1-256gb` | $3.80 | $2,827 | ✅ **nyc2, tor1** |
| AMD MI300 192 GB | `gpu-mi300x1-192gb` | $2.59 | $1,927 | ❌ listed, 0 regions |
| AMD MI350 288 GB (spot) | `gpu-mi350x1-288gb-spot` | $2.46 | $1,830 | ✅ ric1 |
| AMD MI355 288 GB (spot) | `gpu-mi355x1-288gb-spot` | $2.97 | $2,210 | ✅ mem1 |

(Full dump preserved below.)

**Price comparison on the exact silicon we want**:

| | AWS g6e.xlarge (on-demand) | DO `gpu-l40sx1-48gb` |
|---|---|---|
| GPU | L40S 48 GB | L40S 48 GB |
| vCPU / RAM | 4 / 32 GB | 8 / 64 GB |
| $/hr | $1.86 | $1.57 (~**15 % cheaper**) |
| Capacity today | ❌ tight across us-east-1 AZs | ❌ gated behind quota request |

## The "available=true, regions=[]" pattern

Every NVIDIA SKU in the dump reports `available: true` but with an empty `regions` list. This is DO's inventory-gating signal: the SKU exists globally, but **your account has not been allocated capacity in any region yet**. New DO accounts default to zero NVIDIA GPU access; you have to request quota through DO Support.

AMD SKUs (MI300 family and Blackwell B300) have some regions visible out of the box (`nyc2`, `tor1`, `ric1`, `mem1`, `mkc1`) — but those are the exact SKUs that don't help us because of the next point.

## AMD vs NVIDIA for our stack

Our scorer stack is `torch` + `transformers` + optional `vLLM`, all built against **CUDA**. AMD MI300 / MI325 / MI350 / MI355 need **ROCm**. The implications:

- `torch`'s ROCm wheels exist but you're on a different package channel; the version pinning in `pyproject.toml` needs a parallel entry.
- `transformers` is largely architecture-agnostic (it hands tensors to torch) — minimal impact.
- `vLLM`'s ROCm support exists but is less mature than its CUDA path; prefix caching + continuous batching + paged attention are the three things we care about and all three are listed as supported, but not all production-tested across the same breadth of models.
- **The entire benchmark corpus would need to be re-run on ROCm** to confirm numeric parity. Not a bug risk — same math — but it's an audit tax.

**Call**: don't pivot to AMD unless NVIDIA access on DO is firmly denied. The 15 % price saving on L40S is not worth the ROCm transition for M4e. For a *future* multi-cloud strategy where AMD inventory outpaces NVIDIA, revisit.

## How to apply for DO GPU quota

DO does not expose a self-serve "add GPU quota" button on the dashboard. The path is a support ticket.

1. **Open a ticket**: https://cloudsupport.digitalocean.com → "Submit a ticket" → category "Account / Billing / Limits" (exact labels drift; pick the one closest to "limit increase" or "feature access").

2. **In the ticket body**, include:
   - Account email (`shyam.maniyedath@gmail.com`)
   - Specific SKU(s) requested — write out `gpu-l40sx1-48gb` and the region(s). Ask for 1-2 preferred regions by name. DO's L40S inventory is thought to live in `tor1` and `nyc2` as of late 2026 but you should ask them to confirm available regions in the response.
   - **Use case**: "I'm running a 7B/32B open-weight decision model (bernoulli-live, github.com/shyamsfo/bernoulli). Current dev box is AWS g5.xlarge/A10G 24 GB; production target is Qwen2.5-VL-32B-AWQ, which needs 48 GB. AWS g6e.xlarge capacity in us-east-1 is tight and DO is a backup. Expected utilization: ~hours/day during dev + evaluation runs, scaling up for a public endpoint when we go live."
   - **Expected monthly spend**: honest estimate — a single L40S running ~8h/day is roughly $380/mo; a dedicated 24/7 endpoint would be ~$1,200/mo. Give them a range they can underwrite.
   - **Business context, briefly**: "Open-weight model project, open-source MIT-equivalent, infra decision influenced by backup options vs AWS capacity constraints." DO's GPU team cares about distinguishing serious workloads from crypto / spam / hobbyist curiosity.

3. **Set expectations**: lead time is typically 2-5 business days for a first-tier NVIDIA L40S/6000 Ada allocation. H100/H200 can take longer and may require a sales conversation. The MI325/MI300 family doesn't need this since you already have access to those regions.

4. **If approved**: the region will show up in the SKU's `regions` list next time you hit `/v2/sizes`. At that point `doctl compute droplet create` + the usual Terraform workflow Just Works.

5. **If denied or delayed**: fall back to patience on AWS g6e. The ROCm pivot to AMD is a last resort, documented above.

## Decision for Bernoulli

- **Today**: file the DO support ticket for `gpu-l40sx1-48gb` access in `tor1` + `nyc2`. Zero cost to ask; lead time of a few days is parallel to other work.
- **If/when DO approves**: that becomes the primary M4e deployment target. 15 % cheaper than AWS g6e, decoupled from the us-east-1 capacity problem.
- **If DO denies**: continue waiting on AWS g6e. The dev-tier 7B on g5 still works for everything in M2-M8.
- **Don't**: pivot to AMD MI325 unless both NVIDIA paths firmly fail. The ROCm audit tax outweighs the 15 % saving.

## Raw inventory dump (2026-10-08)

```
slug                              vCPU   RAM MB   disk GB   $/hr      $/mo    regions
gpu-4000adax1-20gb                   8    32768      500    0.760     565.44  (none)
gpu-l40sx1-48gb                      8    65536      500    1.570   1,168.08  (none)
gpu-6000adax1-48gb                   8    65536      500    1.570   1,168.08  (none)
gpu-mi350x1-288gb-spot              24   262144      720    2.460   1,830.24  ric1
gpu-mi300x1-192gb                   20   245760      720    2.590   1,926.96  (none)
gpu-mi355x1-288gb-spot              24   262144      720    2.970   2,209.68  mem1
gpu-mi325x1-256gb                   20   163840      720    3.800   2,827.20  nyc2,tor1
gpu-h100x1-80gb                     20   245760      720    4.410   3,281.04  (none)
gpu-h200x1-141gb                    24   245760      720    4.470   3,325.68  (none)
gpu-b300x1-288gb-lc-spot            28   458752      720    8.000   5,952.00  mkc1
gpu-b300x1-288gb-spot               28   458752      720    8.000   5,952.00  (none)
gpu-mi350x8-2304gb-spot            192  2097152     2046   19.680  14,641.92  ric1
gpu-mi300x8-1536gb                 160  1966080     2046   20.720  15,415.68  (none)
gpu-mi355x8-2304gb-spot            192  2097152     2046   23.760  17,677.44  mem1
gpu-mi325x8-2048gb                 160  1310720     2046   30.400  22,617.60  nyc2,tor1
gpu-h100x8-640gb                   160  1966080     2046   35.280  26,248.32  (none)
gpu-h200x8-1128gb                  192  1966080     2046   35.760  26,605.44  (none)
gpu-b300x8-2304gb-lc-spot          224  3670016     2046   64.000  47,616.00  mkc1
gpu-b300x8-2304gb-spot             224  3670016     2046   64.000  47,616.00  (none)
```

Account: `shyam.maniyedath@gmail.com` · droplet limit: 15 · team: My Team.
Token source: `~/.credentials` → `DIGITAL_OCEAN_TOKEN` (write-scoped, rotated 2026-10-08).
