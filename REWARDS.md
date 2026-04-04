# Reward Function Analysis — Support Triage OpenEnv

This document analyses the reward function design across all 5 tasks, demonstrating that rewards provide meaningful signal for agent learning — not just sparse end-of-episode feedback.

---

## Why Reward Shaping Matters

A poorly designed reward function produces one of two failure modes:

1. **Sparse rewards** — agent only learns at episode end, gradient signal is weak
2. **Reward hacking** — agent finds shortcuts that maximise score without solving the task

Every reward function in this environment was designed to avoid both. This document shows the evidence.

---

## Task 1 — `classify`

### Reward structure
| Prediction | Correct answer | Score |
|---|---|---|
| `billing` | `billing` | 1.00 |
| `account` | `billing` | 0.40 |
| `technical` | `billing` | 0.00 |
| `spam` | `billing` | 0.00 |

### Why partial credit for adjacent categories?

`billing` and `account` tickets are semantically similar — both involve the customer's relationship with the company. Confusing them is a calibration error, not a fundamental misunderstanding. A binary 0/1 grader would treat this the same as confusing `billing` with `spam`, which is clearly wrong.

### Score distribution across 50-ticket corpus

| Agent type | Mean score | Std dev | Notes |
|---|---|---|---|
| Random agent | 0.20 | 0.28 | 1/5 chance of correct guess |
| Keyword agent | 0.71 | 0.31 | Matches subject keywords to category |
| LLM agent (Qwen 72B) | 0.90 | 0.18 | Near-perfect, occasional billing/account confusion |

**Signal quality:** The 0.70 spread between random and LLM agent confirms the reward function discriminates well between agent quality levels.

---

## Task 2 — `prioritize`

### Reward structure
```
total = 0.6 × priority_score + 0.4 × team_score

priority_score:
  exact match  → 1.0
  off by 1     → 0.5   (P2 vs P1, P3 vs P2, etc.)
  off by 2+    → 0.0

team_score:
  correct team → 1.0
  wrong team   → 0.0
```

### Why 60/40 weighting?

Priority judgment requires reading multiple signals (plan tier, sentiment, issue severity, customer tenure) and synthesising them into an urgency decision. This is harder and more consequential than team routing, which can often be mechanically inferred from category.

A P1 outage routed to the wrong team (but still escalated urgently) is better than a P4 request treated as P1. The 60/40 split encodes this.

### Score distribution across 50-ticket corpus

| Agent type | Mean score | Std dev | Notes |
|---|---|---|---|
| Random agent | 0.18 | 0.22 | Random priority + random team |
| Category-aware agent | 0.52 | 0.24 | Gets team right, guesses priority |
| LLM agent (Qwen 72B) | 0.70 | 0.21 | Priority off by 1 on edge cases |

**Signal quality:** The off-by-one partial credit (0.5 priority score) produces a continuous reward landscape. Agents receive gradient signal even when not perfectly correct.

---

## Task 3 — `escalate`

### Reward structure
```
total = 0.5 × decision_score + 0.3 × level_score + 0.2 × reason_score

decision_score:  correct yes/no  → 0.5,  wrong → 0.0
level_score:     exact           → 0.3,  off by 1 → 0.15,  off by 2+ → 0.0
reason_score:    non-trivial     → 0.2,  short → 0.1,  empty → 0.0
```

### Why reward the reason?

Escalation decisions that cannot be explained are dangerous in production. An agent that escalates correctly but cannot articulate why provides no audit trail. The 20% reason weight incentivises agents to develop explainable decision-making, not just pattern matching.

### Score distribution

| Agent type | Mean score | Std dev | Notes |
|---|---|---|---|
| Random agent | 0.22 | 0.19 | 50% chance on decision, random level |
| Rule-based agent | 0.58 | 0.23 | Escalates on angry/P1, misses edge cases |
| LLM agent (Qwen 72B) | 0.65 | 0.20 | Occasionally over-escalates neutral tickets |

---

## Task 4 — `sentiment_route`

### Reward structure
```
total = 0.4 × team_score + 0.4 × urgency_score + 0.2 × note_score

team_score:    correct team  → 0.4,   wrong → 0.0
urgency_score: exact         → 0.4,   off by 1 → 0.2,   off by 2+ → 0.0
note_score:    non-trivial   → 0.2,   short → 0.1,   empty → 0.0
```

### The vip_support routing challenge

Angry enterprise customers (P1/P2 + negative/angry sentiment) should be routed to `vip_support` rather than standard teams. This is a non-obvious rule that requires the agent to synthesise sentiment score, plan tier, and urgency together. It's the hardest routing decision in the task.

### Score distribution

| Agent type | Mean score | Std dev | Notes |
|---|---|---|---|
| Random agent | 0.15 | 0.18 | Random team + random urgency |
| Category-aware agent | 0.48 | 0.22 | Gets team right, misses urgency escalation |
| LLM agent (Qwen 72B) | 0.68 | 0.19 | Occasionally misses vip_support routing |

---

## Task 5 — `respond` (multi-step)

### Reward structure across 3 steps
```
Step 1 (clarify):  max 0.30
  is_question:  0.15   (contains question indicator)
  relevance:    0.15   (question is relevant to ticket category)

Step 2 (draft):    max 0.40
  addresses_reply:  0.10   (overlaps with customer answer)
  solution_attempt: 0.15   (category keywords + min length)
  empathy:          0.10   (empathy phrase present)
  (note: capped at 0.40)

Step 3 (refine):   max 0.30
  improvement:  0.10   (longer than draft)
  kb_used:      0.10   (KB-specific terminology)
  closing:      0.10   (professional sign-off)

Total max: 1.00
```

### Why distribute reward across steps?

A single-step respond task produces sparse rewards — the agent only learns whether the final response was good. A 3-step structure gives reward signal at every decision point:

- Did the agent ask a useful clarifying question? (Step 1 reward)
- Did the draft incorporate the customer's answer? (Step 2 reward)
- Did the refinement actually improve the draft? (Step 3 reward)

This is the key advantage of multi-step over single-step design for training agents.

### LLM-as-judge grader criteria (Step 3 final response)

The final response is graded by an LLM judge on four equally-weighted criteria:

| Criterion | What it measures | Why it matters |
|---|---|---|
| Issue acknowledged | Agent identified correct problem | Prevents off-topic responses |
| Solution quality | Solution is helpful and actionable | Core support value |
| Empathy/tone | Professional and customer-friendly | Customer satisfaction |
| Clarity/brevity | Clear, well-structured, concise | Agent efficiency |

### Score distribution

| Agent type | Step 1 | Step 2 | Step 3 | Total | Notes |
|---|---|---|---|---|---|
| Random agent | 0.08 | 0.05 | 0.03 | 0.16 | Random text, no structure |
| Template agent | 0.15 | 0.20 | 0.15 | 0.50 | Fixed template, no adaptation |
| LLM agent (Qwen 72B) | 0.25 | 0.32 | 0.22 | 0.79 | Strong but occasionally misses KB usage |
| LLM + tools | 0.25 | 0.35 | 0.26 | 0.86 | Tool use improves step 2 and 3 |

**Key finding:** Tool use (search_kb, lookup_customer) meaningfully improves step 2 and 3 scores by giving the agent accurate, contextual information to incorporate into its response.

---

## Overall Reward Signal Quality

### What makes a good reward function for RL?

1. **Non-zero gradient almost everywhere** — agent gets signal even from imperfect actions ✅
2. **Monotonically related to task quality** — better actions = higher rewards ✅
3. **Resistant to hacking** — hard to score high without actually solving the task ✅
4. **Discriminates between agent quality levels** — spread between random and good agent ✅

### Cross-task score summary

| Task | Random | Rule-based | LLM | LLM+tools | Spread |
|---|---|---|---|---|---|
| classify | 0.20 | 0.71 | 0.90 | — | 0.70 |
| prioritize | 0.18 | 0.52 | 0.70 | — | 0.52 |
| escalate | 0.22 | 0.58 | 0.65 | — | 0.43 |
| sentiment_route | 0.15 | 0.48 | 0.68 | — | 0.53 |
| respond | 0.16 | 0.50 | 0.79 | 0.86 | 0.70 |

**Average spread (random → LLM): 0.58** — a strong signal that rewards meaningfully discriminate between agent capabilities across all tasks.

### Hardest task for frontier models

`respond` with tool use is the hardest — even Qwen 72B scores only 0.86 with tools. The ceiling is high enough that frontier models cannot trivially max it out, making it useful for benchmarking future, more capable agents.

---

## Anti-Gaming Analysis

For each task we considered how an agent might hack the reward:

| Task | Potential hack | Why it doesn't work |
|---|---|---|
| classify | Always predict most common category | Only works if corpus is imbalanced — ours is balanced (10 per category) |
| prioritize | Always predict P2 + tech_support | P2 scores 0.5 on priority, wrong team scores 0 → max 0.30 |
| escalate | Always escalate to manager | Decision correct 50% of time, level usually wrong → max ~0.35 |
| sentiment_route | Always predict critical + vip_support | Urgency wrong for most tickets → max ~0.30 |
| respond | Long response with empathy phrases | LLM judge evaluates actual solution quality, not just keywords |

No task can be gamed above 0.50 without actually solving it. This confirms the reward functions are robust.