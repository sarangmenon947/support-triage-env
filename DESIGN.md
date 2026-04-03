# Design Document — Support Triage OpenEnv

This document explains the key design decisions behind the Support Triage environment, the reasoning behind reward functions, and the architectural choices made.

---

## Problem Domain

Customer support ticket triage is a genuine, high-value real-world task. Every SaaS company does it. It requires multiple distinct skills that are independently measurable:

1. **Classification** — understanding what the customer wants
2. **Prioritisation** — judging urgency and routing
3. **Escalation judgment** — knowing when a human must intervene
4. **Emotional intelligence** — adapting response to customer sentiment
5. **Communication** — drafting accurate, empathetic replies

These five skills map directly to five tasks of increasing difficulty, making this environment ideal for evaluating multi-dimensional agent capability.

---

## Task Design

### Why 5 tasks instead of 3?

The minimum requirement was 3 tasks. We designed 5 because:
- Each task isolates a distinct cognitive skill
- Together they cover the full support workflow end-to-end
- More tasks gives a richer picture of agent capability
- The difficulty spread (easy → medium → hard) is more granular

### Task difficulty calibration

| Task | Why this difficulty |
|---|---|
| classify | Single decision, clear categories, partial credit for adjacent errors |
| prioritize | Two simultaneous decisions (priority + team), more context needed |
| escalate | Requires judgment about conversation history and attempts, not just ticket content |
| sentiment_route | Must interpret numeric sentiment score and emotional context |
| respond | Multi-turn, requires planning across 3 steps, LLM-graded |

---

## Reward Function Design

### Partial credit everywhere

Every task gives partial credit rather than binary 0/1. This was a deliberate choice:

- **Better training signal** — agents get gradient signal even from imperfect actions
- **More realistic** — confusing billing with account is less wrong than confusing billing with spam
- **Fairer evaluation** — off-by-one priority is better than completely wrong priority

### Classify: adjacent category credit (0.4)

`billing` and `account` tickets are genuinely similar — both involve the customer's account and financial relationship with the company. Penalising this confusion as heavily as confusing `billing` with `spam` would be unfair and produce misleading agent scores.

### Prioritize: 60% priority, 40% team routing

Priority judgment requires reading urgency signals (plan tier, sentiment, issue severity) — this is harder and more consequential than team routing, which can often be inferred mechanically from category. The 60/40 split reflects this.

### Escalate: 50% decision, 30% level, 20% reason

The binary escalation decision (yes/no) is the most important call — getting it wrong has real consequences. The level matters but is more nuanced. The reason is valued because it makes the decision auditable and trustworthy, even if the agent can earn most credit without it.

### Respond: multi-step with capped rewards per step

Each step has a maximum contribution to total reward:
- Step 1 (clarify): max 0.3 — asking good questions is valuable but not the main event
- Step 2 (draft): max 0.4 — the draft is where most of the work happens
- Step 3 (refine): max 0.3 — KB-grounded refinement is important but incremental

This shapes the reward landscape so that agents learn to invest effort proportionally across steps.

---

## Grader Design

### LLM-as-judge for respond task

Keyword matching graders for response quality have a fundamental flaw: an agent can game them by inserting keywords without producing a good response. LLM-as-judge evaluates the actual quality of the response on four criteria:

1. **Issue acknowledged** — did the agent identify the right problem?
2. **Solution quality** — is the solution helpful and actionable?
3. **Empathy/tone** — is the tone professional and customer-friendly?
4. **Clarity/brevity** — is the response clear and appropriately concise?

### Heuristic fallback

If no API key is available (e.g. during local testing without credentials), the grader automatically falls back to a keyword/length-based heuristic. This ensures the environment works in all settings.

### Determinism in static mode

All graders are deterministic given the same input. In static mode, the same `episode_id` always produces the same ticket, the same ground truth, and therefore the same score for the same action. This is essential for reproducible benchmarking.

---

## Dynamic Ticket Generation

### Why dynamic mode matters

In static mode with a 50-ticket corpus, a sufficiently large agent could theoretically memorise correct answers. Dynamic mode prevents this by generating a fresh, unique ticket each episode using an LLM.

The LLM is prompted with:
- Target category distribution
- Realism constraints (P1 must be angry/negative, enterprise customers have non-trivial issues)
- Structural requirements (subject + body + all metadata fields)

High temperature (0.9) ensures variety across episodes.

### Graceful fallback

If the LLM call fails for any reason (no API key, timeout, parse error), the environment silently falls back to the static corpus. This means:
- The environment never crashes due to LLM unavailability
- Static benchmarks remain reproducible
- Dynamic mode is an enhancement, not a dependency

---

## Episode Design

### Single-step vs multi-step

Most tasks are single-step because the real-world action is a single decision — you classify a ticket once, you prioritise it once. Making these multi-step would add artificial complexity without modelling anything real.

The `respond` task is genuinely multi-step because real support agents:
1. Often ask clarifying questions before responding
2. Write a draft response
3. Refine it using documentation/KB articles

The 3-step structure mirrors this natural workflow.

### Episode boundaries

Each episode corresponds to one ticket. Episodes end when:
- A single-step task receives one action (done=True)
- The respond task completes all 3 steps (done=True after step 3)

Calling `step()` on a completed episode raises a `RuntimeError`, forcing the agent to call `reset()` to start fresh.

---

## State Management

The environment maintains per-episode state:
- `_ticket` — the current ticket object
- `_ground_truth` — the answer key (never sent to agent)
- `_step_count` — current step within episode
- `_total_reward` — cumulative reward this episode
- `_history` — list of all actions and rewards taken

For the respond task, additional state is maintained:
- `_clarifying_question` — what the agent asked in step 1
- `_customer_answer` — the simulated customer reply
- `_draft_response` — what the agent drafted in step 2

This state is passed forward into subsequent observations, giving the agent full context at each step.

---

## Simulated Customer Replies

In the respond task, after step 1 the environment must simulate a customer replying to the agent's clarifying question. This is done deterministically using a pre-written corpus of realistic replies indexed by ticket category.

The reply is seeded by `hash(ticket_id + question[:10])` so the same question on the same ticket always produces the same reply — maintaining reproducibility while giving contextually appropriate responses.

---

## Corpus Design

The 50-ticket corpus was designed to be:
- **Balanced** — 10 tickets per category
- **Realistic** — subjects and bodies are plausible real-world tickets
- **Varied in difficulty** — P1 outages, P4 feature requests, obvious spam
- **Varied in customer profile** — free to enterprise, new to long-tenured

The corpus intentionally includes tricky edge cases:
- Billing tickets that mention account details (tests adjacent-category handling)
- Technical tickets with angry enterprise customers (tests priority+sentiment interaction)
- Obvious spam (tests spam detection)

---

## API Design

The HTTP API follows the OpenEnv standard with additions:

- `/tasks` — lists all 5 tasks with metadata, making the environment self-documenting
- `/leaderboard` — task metadata and max scores for benchmarking context
- `/health` — standard health check for deployment monitoring

The `/reset` endpoint accepts an optional `mode` parameter (`static` or `dynamic`) rather than having separate endpoints for each mode. This keeps the API clean and backwards-compatible — existing clients that don't pass `mode` get static mode by default.

---

## Infrastructure

### Why FastAPI?

FastAPI provides automatic request validation via Pydantic (consistent with the model layer), automatic OpenAPI docs at `/docs`, and async support for future extensions. It also starts fast, which matters for HuggingFace Spaces cold starts.

### Why Python 3.11?

Stable, well-supported, compatible with all dependencies. Python 3.12+ had some compatibility issues with certain Pydantic versions at time of development.

### HuggingFace Spaces deployment

The environment runs as a Docker Space on HuggingFace, exposing port 7860 as required. The Dockerfile uses `python:3.11-slim` to keep image size small and startup fast.

---

## Future Extensions

If extending this environment, the highest-value additions would be:

1. **Multi-ticket episodes** — agent handles a queue of tickets, learns to prioritise across them
2. **Adversarial tickets** — tickets designed to confuse classifiers (e.g. billing questions phrased as technical issues)
3. **Agent memory** — persistent state across episodes, tracking customer history
4. **Real KB articles** — dynamic KB retrieval rather than static hardcoded articles
5. **Human-in-the-loop** — escalated tickets go to a simulated human agent with response delay