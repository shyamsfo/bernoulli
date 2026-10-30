
    ▎ Create product/operations/benchmark-primer.md in the Bernoulli repo. Audience: me, future-me, a stakeholder, or a new
    ▎ engineer who wants to understand Bernoulli's benchmark story end-to-end without reading the whole M8 milestone or
    ▎ benchmarks/README.md methodology section. The doc should be readable on a large screen (not console-constrained), ~1200-1500
    ▎  words, explanatory tone, not promotional.
    ▎
    ▎ Structure, with specific content for each section:
    ▎
    ▎ 1. Why Bernoulli benchmarks at all. The product claim is "calibrated probabilities from one forward pass, cheaper and more
    ▎ honest than text-generation-plus-parsing, competitive with the model classes that actually compete, stable under
    ▎ option-order and prompt-wording perturbations." Benchmarks are how we prove each clause. Open with that framing so a reader
    ▎ understands benchmarks aren't ritual — each one answers a specific product-claim question.
    ▎
    ▎ 2. The six academic benchmarks (M8), one paragraph each. For each: dataset name, HF path, size, what question type it
    ▎ exercises (Choice / Binary / Rating), and what architectural dimension it stresses:
    ▎
    ▎ - SST-2 (stanfordnlp/sst2) — 2-way sentiment, 872 eval / 67k train. The headline / easy-win benchmark; 2-option
    ▎ letter-labeled Choice.
    ▎ - AG News (fancyzhx/ag_news) — 4-way topic over short news snippets, 7.6k eval / 120k train. Tests the standard multi-option
    ▎  Choice path (letters A-D).
    ▎ - Banking77 (mteb/banking77) — 77-way customer-support intent, 3,076 eval / 10k train. Tests chunked scoring: options exceed
    ▎  the 26-letter alphabet, so bernoulli/chunked.py splits into 3 chunks of ≤26 letters and softmaxes globally. Also shows how
    ▎ the architecture degrades at scale.
    ▎ - TweetEval-emotion (cardiffnlp/tweet_eval config emotion) — 4-way (anger/joy/optimism/sadness) over tweets, 1,421 eval /
    ▎ 3,257 train. Out-of-domain text style (noisy social-media text) vs the formal domains of the others.
    ▎ - PAWS (google-research-datasets/paws config labeled_final) — binary paraphrase detection over adversarial sentence pairs,
    ▎ 8k eval / 49k train. First benchmark using BinaryQuestion; stitches sentence1 + sentence2 into one state.
    ▎ - arXiv post-cutoff — custom-built from arXiv API via make_dataset.py. 4-way category classification (cs.CL / cs.CV /
    ▎ math.PR / econ.EM) over papers published strictly after Qwen2.5-VL's training cutoff (default 2025-01-01).
    ▎ Anti-contamination guard: its existence prevents the "the model memorized the test set" objection to every other number.
    ▎
    ▎ 3. The five use-case benchmarks (M9), one paragraph each. Same shape; emphasize what landing-page use-case card each
    ▎ cross-references:
    ▎
    ▎ - WildGuardTest (allenai/wildguardmix config wildguardtest) — response-harm binary over (prompt, response) pairs. 1,725
    ▎ eval. Maps to Guardrails card.
    ▎ - ToxicChat (lmsys/toxic-chat config toxicchat1123) — toxicity binary over real LMSYS/Vicuna user queries. 5,082 eval /
    ▎ 5,083 train. Known for distribution-shift from Jigsaw/PerspectiveAPI training data. Guardrails card.
    ▎ - XSTest (natolambert/xstest-v2-copy) — over-refusal probe, 250 handcrafted safe-vs-contrast prompts. No training split by
    ▎ design. Guardrails card.
    ▎ - CLINC150-OOS (clinc_oos config plus) — in-scope vs out-of-scope binary, 5,500 eval / 15,250 train. First "none of these"
    ▎ probability benchmark. Support-triage card.
    ▎ - Yelp 1-5 stars (yelp_review_full) — 5-way star rating, 50k eval / 650k train. First RatingQuestion benchmark. Content
    ▎ moderation / rating card.
    ▎
    ▎ 4. The four baselines, with the key insight about Generative. Address the natural question "aren't these published numbers I
    ▎  can just quote?":

    ▎ 5. Why we can't just download published numbers. Three concrete reasons, each with an example:
    ▎
    ▎ - Framing drift. DeBERTa-zeroshot's published SST-2 accuracy uses a specific hypothesis template over ["positive",
    ▎ "negative"]; our framing uses the option strings as candidate labels with "This text is about {}." as the template. Numbers
    ▎ are related but not identical. For apples-to-apples, inputs must match.
    ▎ - Metrics not reported upstream. Our methodology commits to reorder/reword stability, coverage curves at 95/90/80/50%
    ▎ autodecision rates, 10-bin ECE (JevBench-compatible), per-call latency, cost per 1k decisions. Published papers rarely
    ▎ report these, especially under our exact protocol. Running locally is unavoidable.
    ▎ - Reproducibility footer. Each results/<date>.md records model revision + bernoulli git SHA + dataset revision + seed +
    ▎ hardware. Published numbers have none of this for our configuration.
    ▎
    ▎ Caveat: for some baseline numbers (DeBERTa's base SST-2 accuracy on standard GLUE validation), a published figure can serve
    ▎ as a sanity check. But for the full column — stability, coverage, latency, ECE at 10 bins — there's no shortcut.
    ▎
    ▎ 6. The DeBERTa-on-CPU scheduling problem, with the dual-box fix. Explain the architectural reality: on a single-GPU dev box
    ▎ the Bernoulli server owns the GPU, DeBERTa gets CPU, and CPU latency scales linearly with candidate_labels count (one
    ▎ forward pass per hypothesis). SST-2 (2 opts) is ~7s/call → tolerable. 4-way is ~14s/call → multi-hour per benchmark with
    ▎ stability. Banking77 (77 opts) is ~270s/call → multi-day. First smoke sweep tried this and had to be killed.
    ▎
    ▎ The fix is dual-box scheduling, not a methodology compromise:
    ▎
    ▎ - CPU-only baselines (DeBERTa, BGE-m3+LR) are standalone — they don't call the Bernoulli server. Running them on a separate
    ▎ CPU-heavy box (e.g., c7i.4xlarge, 16 vCPU, AVX-512, ~$0.71/hr) is scientifically identical as long as the model revision +
    ▎ dataset revision + seed are pinned (which the runner already enforces).
    ▎ - The reproducibility footer records hardware: <hostname> per baseline run, so splitting across boxes is honest reporting,
    ▎ not a confound.
    ▎ - This frees the GPU box for Bernoulli-HTTP and Generative-HTTP sweeps without contention; the two sweeps run in parallel.
    ▎ - Still can't fix Banking77 × DeBERTa at full --limit even on a fast CPU box, but a 16-vCPU AVX-512 box is ~5× faster than
    ▎ our g5's CPU, bringing Banking77 from "multi-day" to "~hours". Accept partial coverage or move DeBERTa to a cheap GPU host
    ▎ when the comparison matters.
    ▎ - This mirrors how a real customer would deploy — nobody runs DeBERTa and a 7B VLM on the same production node.
    ▎
    ▎ 7. What the published result eventually looks like. Sketch the shape of benchmarks/README.md's leaderboard summary — the
    ▎ 4-column table per benchmark, plus a short reading guide:

    ▎
    ▎ - Bernoulli wins calibration if its ECE(10) column is lowest.
    ▎ - Bernoulli wins vs. generation if its ECE/NLL beats same-model generative at the same accuracy — direct validation of
    ▎ logit-read vs. text-parse.
    ▎ - Bernoulli wins necessity if it beats BGE-m3+LR, showing embeddings-plus-classifier wasn't enough.
    ▎ - Bernoulli wins model-class if it beats DeBERTa-zeroshot despite being a general-purpose VLM, not a purpose-built
    ▎ classifier.
    ▎
    ▎ Links out to benchmarks/README.md (full methodology), each benchmark subfolder (per-benchmark deep dive),
    ▎ vision_and_roadmap.md §5 (the original calibration claim), and the GPU-contention parking-lot item (now resolved by
    ▎ /v1/generate).
    ▎
    ▎ Style notes:
    ▎
    ▎ - Use tables where comparing 4+ things.
    ▎ - Code snippets in backticks only when citing exact HF paths or module names.
    ▎ - End with a one-paragraph "where this fits in the roadmap" — benchmarks support M5 (regression gate) and M10 (JevBench
    ▎ submission); they're not an endpoint themselves.
    ▎ - No emoji. No marketing language. Explanatory tone, like you're walking someone through the codebase.