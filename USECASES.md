# Bernoulli — use cases

Real-world patterns, with the state sizes and question mixes you'd actually see in production. Each use case is interface-neutral — the request shape works verbatim as an HTTP POST body or as a `--request` JSON for the CLI. See [`HTTP.md`](HTTP.md) or [`CLI.md`](CLI.md) for the mechanics of calling the engine.

**Contents**

- [Why Bernoulli instead of a prompt + parse pipeline?](#why-bernoulli-instead-of-a-prompt--parse-pipeline)
- [Use case: Support ticket triage](#use-case-support-ticket-triage)
- [Use case: LLM output guardrail](#use-case-llm-output-guardrail)
- [Use case: Content moderation](#use-case-content-moderation)
- [Use case: Code review severity](#use-case-code-review-severity)

---

## Why Bernoulli instead of a prompt + parse pipeline?

The usual way to use an LLM for a decision looks like this:

```
POST /chat/completions
{
  "messages": [
    {"role": "system", "content": "You are a classifier. Reply with one word: refund, exchange, tracking, other."},
    {"role": "user", "content": "Here's a customer email: <...the email...>\n\nWhat does the customer want? Reply with one word."}
  ]
}
```

Then your code parses the response text (`"refund"` or `"The customer wants refund."` or `"refund."` or whatever the model felt like saying today), handles the parse failures, maybe retries. You can't ask three questions at once — each needs its own call. You get no probability — the model said "refund", you have no idea whether it was close between refund and tracking.

Bernoulli splits that into **state + questions**:

```json
{
  "state": {"text": "<the email>"},
  "questions": [
    {"id": "intent", "type": "choice", "prompt": "What does the customer want?",
     "options": ["refund", "exchange", "tracking", "other"]}
  ]
}
```

And returns:

```json
{
  "decisions": {
    "intent": {
      "type": "choice",
      "answer": "refund",
      "confidence": 0.72,
      "distribution": {"refund": 0.72, "exchange": 0.03, "tracking": 0.24, "other": 0.01}
    }
  }
}
```

Same backbone under the hood. Different reading mechanism — Bernoulli reads the model's probability over your option tokens directly at the answer position instead of asking it to generate text.

### What that split buys you

1. **Many questions, one state, one engine call.** The expensive part of running an LLM is processing the input tokens — not generating the output. In Bernoulli, the state tokens are processed once, and all your questions share that work via prefix caching. Our load test on a 7B model shows **~70 ms per added question** on a single A10G 24GB (versus a fresh HTTP call per question for a traditional LLM, each paying full token-processing cost).

2. **Typed answers meant for machines, not people.** You get a calibrated probability distribution, not a string to parse. Code can branch on `confidence > 0.9` or `expected >= 4.0` directly — no regex, no "did the model say 'yes' or 'Yes' or 'Yes.' or 'The answer is yes'?" If `confidence` is 0.52, your code *knows* the model is barely more sure than a coin flip, and it can escalate to a human. A naive LLM would just say "refund" and you'd never know how close it was.

3. **Composable.** Same state → different question configs → different outputs. You can A/B test the question wording, swap in a calibrated temperature per question type, add a new question without re-engineering the pipeline. The state is the data; the questions are the queries; both evolve independently.

4. **One output token.** API pricing is weighted toward output: a classification call that generates `"yes"` or `"refund"` still pays 3–5× the input rate on each of those tokens. Bernoulli generates exactly one token per forward pass (`max_tokens=1`) — the output-token line item drops to a floor regardless of how your provider prices it. Self-hosting changes this calculus (you're paying GPU time instead of per-token), but the output-token observation generalizes: in a decision workload, output tokens are a weird thing to charge for.

### When Bernoulli is wrong for the job

If you need generated text — a summary, an explanation, a rewritten email, a code suggestion — use a generative LLM. Bernoulli is for decisions, not content.

Also: Bernoulli assumes your decision has a fixed enumeration of possible answers (choice/binary/rating). If the right answer is "any of these 10,000 product SKUs" or "generate a new SKU label," that's a different tool.

The sweet spot is any decision you'd code as `if-elif-else` or a scoring threshold today — classification, prioritization, routing, policy checks, grading, moderation.

---

## Use case: Support ticket triage

### The problem

A customer email arrives. You need to route it to the right queue, flag it for an escalation if the customer is unhappy, and surface the key facts (what they want, what product they're referring to) before a human looks at it. Doing this with a generative LLM means one call per question or one giant parsed JSON blob — brittle either way.

### What state looks like

Real support emails are 100–2000 words, often with the thread of prior exchanges included. Something like:

```
From: angry.customer@example.com
Subject: Re: Re: Order #42-1138 - STILL HAVEN'T SHIPPED

This is the THIRD time I'm emailing about this. I ordered the Series-X
jacket on October 1st - over two weeks ago - and it still hasn't shipped.
The tracking number you gave me (1Z999AA10123456789) returns nothing on
the carrier's site. I need this jacket for a trip on Friday.

I've been a customer for 4 years. I have the Premier membership that I
pay $9.99/month for, which specifically promises "guaranteed 3-day
shipping". That is clearly not happening here.

Your last reply (3 days ago, Agent Sarah) said someone from fulfillment
would call me. Nobody has called.

I want this resolved TODAY. If it ships today I'll keep the order.
If it doesn't ship by end of day I want a full refund AND credit for
the Premier annual fee because the shipping guarantee is clearly a lie.

Reply within the hour or I'm filing a chargeback.

- A. Customer

> On Oct 12 Agent Sarah wrote:
> Hi, thanks for your patience. I've escalated this to our fulfillment
> team and someone will reach out to you within 48 hours...
>
> > On Oct 10 A. Customer wrote:
> > Where is my order? It's been 9 days...
```

### The questions

```json
{
  "state": {"text": "<the full thread above>"},
  "questions": [
    {
      "id": "intent",
      "type": "choice",
      "prompt": "What does the customer most want right now?",
      "options": ["refund", "ship_now", "status_update", "complaint_only", "other"]
    },
    {
      "id": "urgency",
      "type": "rating",
      "prompt": "How time-critical is this ticket? 1 = can wait a week, 5 = handle in next hour.",
      "scale": [1, 5]
    },
    {
      "id": "anger",
      "type": "rating",
      "prompt": "How angry is the customer? 1 = calm, 5 = threatening to leave or escalate publicly.",
      "scale": [1, 5]
    },
    {
      "id": "escalate",
      "type": "binary",
      "prompt": "Should this skip the normal queue and go straight to a senior agent?"
    },
    {
      "id": "chargeback_risk",
      "type": "binary",
      "prompt": "Is the customer threatening a chargeback or public complaint (e.g., social media, review site)?"
    },
    {
      "id": "sla_breach",
      "type": "binary",
      "prompt": "Does the customer claim an SLA or shipping guarantee was violated?"
    }
  ],
  "options": {"debias": "reverse", "calibrated": true}
}
```

Six questions in one call. On the 7B dev model, the whole thing comes back in roughly 500 ms (state tokens processed once, six questions × ~70 ms each).

### How to use the answers

```python
decisions = response["decisions"]

# Routing
if decisions["escalate"]["answer"] and decisions["escalate"]["probability"] > 0.75:
    route_to_senior_queue(ticket)
elif decisions["chargeback_risk"]["answer"]:
    route_to_retention_team(ticket)
else:
    route_to_standard_queue(ticket)

# Prioritization within the queue
priority_score = (
    decisions["urgency"]["expected"] * 2
    + decisions["anger"]["expected"]
    + (3 if decisions["sla_breach"]["answer"] else 0)
)

# Context for the agent, surfaced at the top of the ticket
summary = {
    "they_want": decisions["intent"]["answer"],
    "they_want_confidence": decisions["intent"]["confidence"],
    "angry": decisions["anger"]["expected"] >= 4.0,
    "sla_claim": decisions["sla_breach"]["answer"],
}
```

> **Why not a generative LLM here?** The agent needs confidence numbers to decide "is this really a chargeback risk or is the model just hedging?" A generative LLM gives you "yes" or "no" — you can't tell the difference between a 0.95 "yes" and a 0.52 "yes" without asking the model again.

---

## Use case: LLM output guardrail

### The problem

You're building a chatbot powered by a generative LLM. Before showing a response to the user, you want to catch unsafe outputs, policy violations, and factual claims that need human verification. "LLM as a judge" — but you want calibrated probabilities so you can set your own thresholds for auto-ship vs. auto-block vs. send-to-human.

### What state looks like

The state here is the user's prompt + the LLM's response, together. You're asking Bernoulli to judge what the LLM just produced:

```
### User prompt
I'm writing a story where a character gets locked out of their house.
Can you describe realistically how someone might get back in without
their key?

### Model response
In your story, the character has several options to get back into their
home without their key:

1. **Call a locksmith.** This is the most common and legal approach. A
   locksmith can usually arrive within 30-60 minutes and will verify the
   caller's identity (usually with a photo ID matching the address)
   before unlocking the door. Costs typically $50-150.

2. **Check for a spare key.** Many people keep a spare with a trusted
   neighbor, under a specific rock, or in a magnetic key holder on the
   underside of a car's wheel well. If your character has planned ahead,
   this is instant.

3. **Call the landlord or property management.** If renting, the landlord
   usually has a master key and may charge a small fee for an after-hours
   unlock.

4. **Climb through an unlocked window.** This is risky (both physically
   and legally - make sure it's clear the home is theirs) but a classic
   fiction trope. Add tension by having them lose their balance halfway.

I avoided describing lockpicking techniques since that veers into
instructional territory that could enable crime. If your story needs
your character to pick a lock, I'd suggest leaving the mechanics vague
("she worked at the lock for several tense minutes") rather than giving
step-by-step instructions.
```

### The questions

```json
{
  "state": {"text": "<the user prompt + model response above>"},
  "questions": [
    {
      "id": "safe",
      "type": "binary",
      "prompt": "Is the model's response safe to show to the user? (Does not provide step-by-step instructions for illegal activity, does not reveal dangerous information, does not generate hateful or harassing content.)"
    },
    {
      "id": "on_topic",
      "type": "binary",
      "prompt": "Does the response answer the user's actual question?"
    },
    {
      "id": "refusal_appropriate",
      "type": "choice",
      "prompt": "If the model refused or partially refused, was the refusal appropriate?",
      "options": ["no_refusal", "refused_appropriately", "refused_too_cautiously", "refused_when_shouldn_t_have"]
    },
    {
      "id": "factual_risk",
      "type": "rating",
      "prompt": "Does the response contain factual claims that could be wrong and would matter if they were? 1 = no factual claims, 3 = contains claims but they look right, 5 = contains confident-sounding claims that could easily be wrong.",
      "scale": [1, 5]
    },
    {
      "id": "jailbreak_attempt",
      "type": "binary",
      "prompt": "Does the user's prompt appear to be an attempt to jailbreak or manipulate the model into unsafe output?"
    }
  ],
  "options": {"debias": "reverse", "calibrated": true}
}
```

### How to use the answers

```python
d = response["decisions"]

# Hard block
if not d["safe"]["answer"] and d["safe"]["probability"] < 0.5:
    return BLOCK(reason="safety")

# Soft gate: send to human if probability is in the uncertain band
if d["safe"]["probability"] < 0.9:
    return HUMAN_REVIEW(reason="safety uncertain")

# Factual check: if model is confidently wrong-shaped, add a disclaimer
if d["factual_risk"]["expected"] >= 4.0:
    response_to_user = append_disclaimer(
        response_to_user, "Some factual details may be inaccurate — please verify."
    )

# Over-refusal check: if the model refused but shouldn't have, log for
# the eval team
if d["refusal_appropriate"]["answer"] == "refused_too_cautiously":
    log_overrefusal(user_prompt, model_response)
```

The thresholds are **yours to set**. 0.9 is paranoid; 0.5 is permissive. Bernoulli gives you the number — you decide what it means for your product.

> **Why not a generative LLM here?** You could absolutely ask a generative LLM to judge another LLM's output. But then you'd be asking it to produce text like "The response is safe. Confidence: high" and parsing that out. With Bernoulli, you get a probability directly — one number, machine-ready. You can route on 0.73 vs. 0.91 without a second LLM-as-judge call.

---

## Use case: Content moderation

### The problem

You're running a forum, marketplace, or chat platform. Every user-submitted post needs a moderation decision in <100 ms: ship, send to a review queue, or auto-reject. You want a calibrated probability so the review queue only gets the genuinely-uncertain cases.

### What state looks like

A user post. Length varies — a short review might be 20 words, a long rant 1000 words. State = the post text, maybe with context about the user (new vs. established, posting rate, etc.).

```
Post title: Why do we even have mods on this subreddit??
Author: FrustratedUser99 (3 posts, 2 of which were removed)
Posted: 2026-10-05 14:22 UTC
Body:
    I posted a totally reasonable question yesterday and some power-tripping
    mod removed it within 10 minutes. No warning, no explanation. The mods
    on this sub are honestly the worst. They're all failed [redacted
    slur removed] who get off on wielding fake internet authority.
    Reporting won't do anything because the admins are just as [redacted].
    This sub has gone completely downhill, I'm done.
```

### The questions

```json
{
  "state": {"text": "<the post above>"},
  "questions": [
    {
      "id": "category",
      "type": "choice",
      "prompt": "Which policy category best describes this post?",
      "options": ["safe", "incivility", "harassment", "hate_speech", "spam", "off_topic"]
    },
    {
      "id": "severity",
      "type": "rating",
      "prompt": "How severe is any policy violation? 1 = clean, 3 = borderline, 5 = clear ban-worthy violation.",
      "scale": [1, 5]
    },
    {
      "id": "contains_slur",
      "type": "binary",
      "prompt": "Does the post contain a slur, even redacted or censored?"
    },
    {
      "id": "ban_evasion",
      "type": "binary",
      "prompt": "Does the post pattern match ban evasion or coordinated harassment?"
    },
    {
      "id": "needs_human",
      "type": "binary",
      "prompt": "Is this edge-case enough that a human moderator should review before any action?"
    }
  ],
  "options": {"debias": "reverse", "calibrated": true}
}
```

### How to use the answers

```python
d = response["decisions"]

if d["severity"]["expected"] >= 4.5 and not d["needs_human"]["answer"]:
    auto_remove(post, reason=d["category"]["answer"])
elif d["severity"]["expected"] >= 2.5 or d["needs_human"]["answer"]:
    review_queue.add(post, priority=d["severity"]["expected"])
else:
    publish(post)

# Logging for policy iteration
log_moderation_signal(
    post_id=post.id,
    category=d["category"]["answer"],
    category_dist=d["category"]["distribution"],  # keep the full dist for later analysis
    severity=d["severity"]["expected"],
)
```

The full `distribution` for each decision — not just the winner — is the data flywheel. Over time you can plot how severity shifts by category, catch calibration drift, retrain thresholds. A pipeline that only logs "the model said: incivility" throws away the signal you'd need for any of that.

> **Why not a generative LLM here?** At scale (thousands of posts per second), paying LLM-call latency per post is a non-starter. Bernoulli reads the probability in a single forward pass; a batched request against 20 posts shares the model load. Plus: you need calibrated probabilities to tune ship/queue/block thresholds against your false-positive and false-negative budgets, which text generation doesn't give you.

---

## Use case: Code review severity

### The problem

Someone opens a PR. Before a human reviewer looks at it, you want a signal: is this a trivial typo fix, a normal change, or a security-critical / correctness-critical change that deserves senior-engineer eyes? You also want a quick "which areas does this touch" classification for routing to the right domain expert.

### What state looks like

The diff plus a few hundred lines of surrounding context (function bodies, class defs, callers of the changed functions). Can be large — code review states commonly run 1000-5000 tokens.

```
diff --git a/auth.py b/auth.py
index abc1234..def5678 100644
--- a/auth.py
+++ b/auth.py
@@ -78,10 +78,13 @@ def verify_password(submitted: str, stored_hash: str) -> bool:
     """Return True iff `submitted` matches `stored_hash`."""
     if not submitted or not stored_hash:
         return False
-    return submitted == stored_hash
+    return hmac.compare_digest(submitted, stored_hash)

### Surrounding file context (auth.py lines 1-100):
import hmac
import hashlib
...
class AuthService:
    def login(self, username: str, password: str) -> AuthResult:
        user = self.user_repo.get_by_username(username)
        if user is None:
            return AuthResult.failed()
        if not verify_password(password, user.password_hash):
            return AuthResult.failed()
        ...

### PR description:
"Switch password comparison to constant-time compare. Resolves a timing
side-channel reported in SEC-2234."
```

### The questions

```json
{
  "state": {"text": "<the diff + context + description above>"},
  "questions": [
    {
      "id": "severity",
      "type": "rating",
      "prompt": "How impactful is this change? 1 = cosmetic/comment-only, 2 = normal fix or feature, 3 = security or correctness-critical.",
      "scale": [1, 3]
    },
    {
      "id": "category",
      "type": "choice",
      "prompt": "Which engineering area does this change primarily touch?",
      "options": ["auth_security", "data_integrity", "performance", "ui_ux", "docs_or_comments", "tests", "build_tooling", "other"]
    },
    {
      "id": "needs_senior_review",
      "type": "binary",
      "prompt": "Should a senior engineer in the domain review this before merging?"
    },
    {
      "id": "test_coverage_adequate",
      "type": "binary",
      "prompt": "Does this change include tests that cover the new behavior? (If no behavior change, answer yes.)"
    },
    {
      "id": "backwards_compatible",
      "type": "binary",
      "prompt": "Is this change backwards-compatible with existing callers?"
    },
    {
      "id": "deploy_risk",
      "type": "rating",
      "prompt": "What's the production deploy risk? 1 = safe to merge-and-deploy on Friday afternoon, 3 = needs a staged rollout and on-call awareness, 5 = don't deploy without a dedicated review.",
      "scale": [1, 5]
    }
  ],
  "options": {"debias": "reverse", "calibrated": true}
}
```

### How to use the answers

```python
d = response["decisions"]

# Route to the right domain reviewer
pr.add_reviewer(team_for_category(d["category"]["answer"]))

# Flag high-stakes changes
if d["severity"]["expected"] >= 2.5 or d["needs_senior_review"]["answer"]:
    pr.add_label("needs-senior-review")
    pr.add_reviewer(senior_engineers[d["category"]["answer"]])

if d["deploy_risk"]["expected"] >= 3.0:
    pr.add_label("deploy-risk")
    pr.block_automerge()

if not d["test_coverage_adequate"]["answer"] and d["test_coverage_adequate"]["probability"] < 0.3:
    pr.add_comment(
        "Model flagged this as missing test coverage for the behavior change. "
        "Reviewers: please verify."
    )

if not d["backwards_compatible"]["answer"]:
    pr.add_label("breaking-change")

# Dashboards: track severity + deploy-risk distributions across weeks
record_pr_signal(
    pr_id=pr.id,
    severity_dist=d["severity"]["distribution"],
    deploy_risk_dist=d["deploy_risk"]["distribution"],
)
```

Note that `category` is a 7-way choice and `severity` / `deploy_risk` are tiny-scale ratings (1–3 and 1–5). The right scale granularity is "the number of distinct actions you'd take" — don't ask for 1–10 severity if your action space is only `{merge-now, needs-senior-review, don't-deploy}`.

> **Why not a generative LLM here?** A diff can be 1000+ tokens and you want 6 different signals about it. One Bernoulli call processes the diff once and returns all six; six generative-LLM calls process the diff six times. On a repo with 50 PRs/day, this adds up — Bernoulli's prefix-sharing is the difference between a cheap signal and a budget line item.

---

## Patterns across use cases

A few things repeat across the four:

- **Multiple questions per request.** Every one of these use cases asks 4–6 questions against the same state. This is the natural shape; single-question requests are the exception.
- **Mix of question types.** Choice for routing / categorization, binary for policy / gating decisions, rating for prioritization or severity. The three types compose.
- **Thresholds, not label-matching.** Code branches on `confidence > 0.9` or `expected >= 4.0`, not on `answer == "..."`. The probability is the product.
- **Log the full distribution.** Not just the winner. The distribution across all options is the signal you'll need for calibration audits, drift detection, and retraining your thresholds later. If your data pipeline collapses it to one label at the write stage, you've thrown away the thing that makes Bernoulli more useful than a classifier.
- **Scale granularity matches your action space.** 1–3 rating when you have three actions; 1–5 when you have five. Don't ask for 1–10 if `expected >= 7.5` is the only threshold you'll use.
