# Bernoulli — OSS repo notes for Claude Code

This is the open-source core: library, API, benchmarks, public docs. The
operational surface (serve-up/down recipes, deployment config, website,
planning docs, Chrome/Salesforce integrations) lives in a private sister
repo (`bernoulli-ops`). This file is the reference for OSS-side work.

## Workflow

Task runner is `just`. Dev + benchmark recipes live in submodules:

- `just --list dev` — code quality (ci/test/lint/fmt), local serving (serve-local),
  Docker image build + run, model pulls.
- `just --list bench` — eval harness (eval, eval-calibrated, fit-cal) and
  benchmark sweeps (benchmark, sweep-academic, sweep-usecase, loadtest).

```bash
# code quality, locally
uv sync
just dev ci
just dev test-fast

# run a local server (uses HFScorer; loads the model into the process)
just dev serve-local

# or build + run the vLLM-backed Docker image
just dev docker-build && just dev docker-run
```

GPU-marked tests auto-skip when no CUDA device is visible; non-GPU tests run anywhere.

## Config

Env-driven via pydantic-settings, prefix `BERNOULLI_`. Defaults in `bernoulli/config.py`:

| var | default |
|---|---|
| `BERNOULLI_MODEL_ID` | `Qwen/Qwen2.5-VL-7B-Instruct` (dev + early M4; step up to 32B AWQ later when capacity frees up) |
| `BERNOULLI_MODEL_REVISION` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| `BERNOULLI_DTYPE` | `bfloat16` |
| `BERNOULLI_DEVICE` | `cuda` |
| `BERNOULLI_MAX_MODEL_LEN` | `32768` (set to `8192` when running vLLM on 24GB cards — see Gotchas) |
| `BERNOULLI_SCORER` | `hf` (HFScorer, M2+) / `vllm` (VLLMScorer, M4+) |
| `BERNOULLI_DEFAULT_DEBIAS` | `reverse` |
| `BERNOULLI_CALIBRATION_PATH` | unset (set to `calibration/qwen2.5-vl-7b.json` to apply fitted T at serving time) |

## Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-10-05 | Dev-tier backbone: `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc59489...` | Fits L4 24GB in bf16 with headroom, public (no gating), well-tested logit behavior around single-letter answers. Flexible — see `bernoulli/scorer.py` interface. |
| 2026-10-05 | No LiteLLM in the Scorer path | The core technique reads *logits over label tokens* with `allowed_token_ids` + prefix caching. OpenAI-compat APIs don't expose any of that uniformly. Scorer stays as `HFScorer` / `VLLMScorer`. LiteLLM may show up in `evals/baselines.py` later for the generative-baseline comparison in M3. |
| 2026-10-05 | AWS g6.xlarge for dev; no auto-shutdown | User explicitly chose ease > cost. $0.80/hr running, $6.40/mo stopped. |
| 2026-10-05 | CI on bernoulli, not GHA | The real tests need a GPU; GHA free runners can't run them, so GHA would gate on nothing meaningful. Clean-env check comes from `terraform destroy && apply` instead. |
| 2026-10-05 | Letters A–I for rating labels (not digits) | **Deviation from vision doc §4.** Qwen's tokenizer makes `" 1"`…`" 9"` TWO tokens (space + digit), which breaks the single-token-answer invariant. Letters A–I are single-token (one per rating value, up to scale width 9). Ratings map A→low, B→low+1, … and the response distribution keys are integer strings per the API spec. |
| 2026-10-05 | Model class is hard-coded to `Qwen2_5_VLForConditionalGeneration` | Only one backbone in the picture for M2–M3. When M4 introduces the production backbone, generalize via a small registry (dispatch on `AutoConfig(name_or_path).model_type`). Not worth building the registry for one model. **Superseded 2026-10-08** — see row below. |
| 2026-10-08 | Backbone load is now config-driven, not hard-coded | `bernoulli/scorer.py:_load_backbone()` reads `AutoConfig(model_id)` and routes VLMs (non-None `vision_config` sub-config) to `AutoModelForImageTextToText`, text-only models to `AutoModelForCausalLM`. Covers Qwen2.5-VL / Qwen3-VL / Llama-3.2-Vision / Gemma-3-Vision / Pixtral / LLaVA / Mistral / Gemma / Llama / etc. without any backbone-specific code. The old M4-era TODO is closed; swapping backbones is now purely a `BERNOULLI_MODEL_ID` + `BERNOULLI_MODEL_REVISION` change (plus the single-token-labels assertion at scorer construction). |
| 2026-10-05 | `AutoProcessor` deferred to M4 | For Qwen2.5-VL the processor pulls `Qwen2VLVideoProcessor`, which hard-requires `torchvision`. M2 is text-only so we use `AutoTokenizer` directly. Images (and the processor + torchvision dep) land in M4. |
| 2026-10-05 | Milestones re-ordered: text-only shippable product before multimodal | User direction. New order: M4 production serving (text-only) → M5 hardening (v1.0) → M6 multimodal → M7 LoRA (optional). Audio parked. |
| 2026-10-05 | Production backbone: `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` @ `66c370b7`, instance `g6e.xlarge` (L40S 48GB) | AWQ int4 is ~20 GB — fits L40S 48GB with ~25 GB headroom for KV cache + batching at 32k context. L40S has native int8/int4 math. Single-GPU keeps vLLM config simple. Instance cost ~$1.86/hr on-demand. Dev-tier 7B is still runnable on the same box via `BERNOULLI_MODEL_ID` override. |
| 2026-10-05 | Deferred the step-up to M4e — stay on g6.xlarge + 7B through M4a-d | g6e.xlarge capacity in us-east-1 is tight today (us-east-1c and 1a both reported InsufficientInstanceCapacity when the plan was fresh). The vLLM scorer, FastAPI server, request batching, Dockerfile, load test, and HTTP eval harness are all backbone-agnostic and run fine on 7B / L4 24GB. Revisit the g6e capacity + 32B pull when the production step-up actually matters. Added `availability_zone` variable (default us-east-1a) + 4-min create timeout so capacity issues surface fast. |
| 2026-10-05 | Fell back from g6.xlarge → g5.xlarge | g6.xlarge capacity in us-east-1 also went dark (1a, 1b, 1d all reported InsufficientInstanceCapacity). g5.xlarge (A10G 24GB, ~$1.01/hr) has broader capacity and is a wash for 7B inference. Will re-visit at M4e. |
| 2026-10-05 | Docker base = `vllm/vllm-openai:v0.31.0-cu129-ubuntu2404` | Tried `nvidia/cuda:13.0.0-runtime-ubuntu24.04` first; Triton's JIT needs `libcuda.so.1` on gcc's search path at engine init, and `nvidia-container-toolkit` injects the lib at runtime but doesn't configure the linker. vLLM's prebuilt image has torch + vllm + CUDA + Triton kernels pre-baked and tested together — saves the yak shave. |
| 2026-10-05 | Scorer dispatch via `load_scorer(settings)` factory | `BERNOULLI_SCORER` env flips between HFScorer and VLLMScorer. Factory imports vllm lazily so the hf path doesn't pay for the heavy import at module load. |
| 2026-10-05 | Batching done at the Scorer protocol level (`score_batch`) | `decide()` and `chunked_choice()` collect all permutation / chunk prompts up front and call `score_batch` once. On vLLM that's one engine call with continuous batching + prefix caching. HFScorer loops. Within-request batching, continuous batching across requests is vLLM-automatic. |
| 2026-10-10 | 32B production-size backbone measured on Lambda H100 PCIe 80 GB | One-shot rental ($9) to replace the "production step-up parked on capacity" caveat with real numbers. Weighted 231-task JevBench cohort: 7B 0.706 → 32B **0.801** (+9.5 pp); hard-tier **+18 pp** (0.423 → 0.604); Brier roughly halved on standard + hard tiers. Pivoted from `Qwen2.5-VL-32B-Instruct-AWQ` to the base unquantized variant mid-run — transformers 5.x's AWQ loader (gptqmodel) doesn't handle Qwen2.5-VL's partial quantization (vision tower stays fp16, only LM is AWQ), and vLLM 0.26's AWQ kernels need CUDA 13 while Lambda H100 driver tops at CUDA 12.8. H100 80 GB has room for base fp16 32B (65 GB weights). Full writeup: `benchmarks/jevbench/results/2026-10-10-32b.md`. AWQ-quantization effect is a separate open question, deferred until L40S hardware is available. |
| 2026-10-29 | HFScorer passes `logits_to_keep=1` to the backbone forward | We only read `logits[0, -1, :]` — the last-position vector — but the backbone was projecting *all* positions through `lm_head`. On Qwen2.5-VL's ~150k vocab at full 8192 context that's a ~2.5 GB bf16 allocation at the final linear, which OOM'd on A10G 24GB when the Chrome extension sent long ToS pages (3 OOMs in production serve.log before the fix). The kwarg restricts the projection to just the last position (~300 KB). Determinism GPU test confirms numerically identical output. Falls back to no-kwarg path via `inspect.signature` check if an exotic backbone doesn't accept the parameter. |

## Gotchas

- **vLLM max_model_len on 24GB GPUs**: default `BERNOULLI_MAX_MODEL_LEN=32768` won't fit on A10G/L4 24GB after loading 7B weights. Set `BERNOULLI_MAX_MODEL_LEN=8192` for vLLM on these cards; HFScorer doesn't care.
- **Zombie `VLLM::EngineCore`**: if the vLLM engine init crashes partway (e.g., config error), the engine subprocess can survive the Python exit holding ~18 GB on the GPU. Check `nvidia-smi --query-compute-apps=pid --format=csv,noheader` and `sudo kill -9 <pid>`.
- **CUDA alignment on torch/vision/audio**: when bumping the pytorch index version (`[tool.uv.sources]` → `pytorch-cu130` etc.), bump all three. `transformers>=5.x` imports torchaudio at import time, so a mismatched CUDA version silently breaks every scorer load with a cryptic error from inside torchaudio.
- **torchcodec in the vLLM Docker image**: the `vllm/vllm-openai` base ships transformers 5.x which imports torchcodec, which tries to load libnvrtc.so.13 (cu13 lib) against the image's cu12.9 CUDA. Dockerfile uninstalls torchcodec after the pip install step — fine for text-only. If image support lands later, you'll need either a cu13 vllm base or to put torchcodec back with the right CUDA libs.
- **vLLM image needs `ffmpeg`**: transformers 5.x's import chain pulls libav* via torchcodec. Keep ffmpeg in the Docker image even though we uninstall torchcodec, in case other imports need it.
- **Qwen VL thinking mode**: Qwen hybrid-reasoning models (3+) can emit `<think>` before the answer. The scorer must disable thinking in the chat template and assert next token is a label.
- **Label alphabet runs out at 26**: Banking77 (77-way intent) needs chunked / tournament scoring.

# clean up a stuck vllm (zombie EngineCore pinning GPU memory)
ssh bernoulli 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r sudo kill -9'
```
