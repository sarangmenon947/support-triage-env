---
title: Support Triage OpenEnv
emoji: 🎫
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Support Triage OpenEnv

> **Customer Support Ticket Triage** — an OpenEnv-compliant environment where AI agents learn to classify, prioritise, escalate, sentiment-route, and respond to real-world support tickets across **5 tasks** of increasing difficulty, with **LLM-powered grading** and **dynamic ticket generation**.

---

## Why This Environment?

Every software company processes thousands of support tickets daily. Handling them well requires a combination of skills that are genuinely hard for AI agents:

- **Understanding** — what is the customer actually asking?
- **Judgment** — how urgent is it, and who should handle it?
- **Emotional intelligence** — how angry is the customer, and how should that change routing?
- **Escalation logic** — when does a ticket need a human, and at what level?
- **Communication** — drafting empathetic, accurate, KB-grounded replies

This environment trains and evaluates agents across all five dimensions using a corpus of 50 realistic synthetic tickets, LLM-based graders, and an optional dynamic ticket generation mode that produces infinite episode variety.

---

## Tasks (Easy to Hard)

| # | Task ID | Difficulty | Steps | Description |
|---|---------|-----------|-------|-------------|
| 1 | `classify` | Easy | 1 | Classify ticket into one of 5 categories |
| 2 | `prioritize` | Medium | 1 | Assign P1-P4 priority + route to correct team |
| 3 | `escalate` | Medium | 1 | Decide escalation need, level, and reason |
| 4 | `sentiment_route` | Medium | 1 | Route emotionally charged ticket with urgency flag |
| 5 | `respond` | Hard | 3 | Multi-turn: clarify, draft, refine using KB articles |

---

### Task 1 — `classify` (Easy)

**Observation:** Ticket (subject, body, customer metadata)

**Action:**
```json
{"category": "<billing|technical|account|feature_request|spam>"}
```

**Grading:**
- 1.0 — exact category match
- 0.4 — adjacent category (e.g. billing and account)
- 0.0 — wrong category

---

### Task 2 — `prioritize` (Medium)

**Observation:** Ticket + customer context (plan tier, tenure, sentiment)

**Action:**
```json
{"priority": "<P1|P2|P3|P4>", "assigned_team": "<team_name>"}
```

**Valid teams:** `billing_team`, `tech_support`, `account_team`, `product_team`, `spam_filter`

**Grading:** 60% priority accuracy + 40% team routing
- Priority: 1.0 exact, 0.5 off by 1 level, 0.0 off by 2+
- Team: 1.0 correct, 0.0 wrong

---

### Task 3 — `escalate` (Medium)

**Observation:** Ticket + conversation history + number of prior agent attempts

**Action:**
```json
{"should_escalate": true, "escalation_level": "<none|L1|L2|L3|manager>", "reason": "<explanation>"}
```

**Grading:**
- 50% correct escalation decision (yes/no)
- 30% correct escalation level (off-by-one = partial credit)
- 20% reason provided and non-trivial

---

### Task 4 — `sentiment_route` (Medium)

**Observation:** Ticket + numeric sentiment score (-1.0 to 1.0) + detected anger keywords

**Action:**
```json
{"assigned_team": "<team>", "urgency_flag": "<low|normal|high|critical>", "de_escalation_note": "<message>"}
```

**Valid teams:** `billing_team`, `tech_support`, `account_team`, `product_team`, `spam_filter`, `vip_support`

**Grading:**
- 40% correct team (including vip_support for angry high-value customers)
- 40% correct urgency flag (off-by-one = partial credit)
- 20% de-escalation note present and non-trivial

---

### Task 5 — `respond` (Hard, 3-step)

A multi-turn episode that mirrors how real support agents work:

**Step 1 — Clarify** (max reward: 0.3)

Agent reads the ticket and asks one clarifying question.
```json
{"clarifying_question": "<question>"}
```
Graded on: is it a real question? is it relevant to the ticket category?

**Step 2 — Draft** (max reward: 0.4)

Environment simulates customer reply. Agent drafts a response.
```json
{"draft_response": "<draft>"}
```
Graded on: addresses customer reply, solution attempt, empathy.

**Step 3 — Refine** (max reward: 0.3)

Environment provides KB articles. Agent refines the draft into a final response.
```json
{"response_text": "<final polished reply>"}
```
Graded on: improvement over draft, KB usage, proper closing.

The respond grader uses **LLM-as-judge** (with heuristic fallback) scoring four criteria: issue acknowledged, solution quality, empathy/tone, clarity/brevity.

---

## Episode Modes

### Static mode (default, deterministic)
```json
POST /reset {"task": "classify", "mode": "static"}
```
Picks from a corpus of 50 realistic synthetic tickets. Same `episode_id` always produces the same ticket — fully reproducible for benchmarking.

### Dynamic mode (LLM-powered, infinite variety)
```json
POST /reset {"task": "classify", "mode": "dynamic"}
```
Calls an LLM to generate a brand new, never-seen-before ticket each episode. Agents cannot memorise answers — they must truly generalise. Falls back to static mode gracefully if no API key is available.

---

## Reward Function Summary

| Property | Value |
|---|---|
| Range | 0.0 to 1.0 per episode |
| Partial credit | Yes — all tasks have partial scoring |
| Multi-step rewards | Yes — respond gives reward at each of 3 steps |
| Deterministic | Yes in static mode; infinite variety in dynamic mode |
| Grader type | LLM-as-judge for respond; deterministic rules for others |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET` | `/state` | Get current episode state |
| `GET` | `/tasks` | List all 5 tasks with descriptions |
| `GET` | `/leaderboard` | Task metadata and max scores |
| `GET` | `/health` | Health check |

### /reset
```json
POST /reset
{
  "task": "classify",
  "episode_id": "optional-seed-string",
  "mode": "static"
}
```

### /step
```json
POST /step
{
  "task": "classify",
  "data": {"category": "billing"}
}
```

---

## Quick Start

### Local (no Docker)

```bash
# 1. Clone and install
git clone https://github.com/sarangmenon947/support-triage-env
cd support-triage-env
pip install -r requirements.txt

# 2. Start the server
uvicorn server:app --host 0.0.0.0 --port 7860

# 3. Health check
curl http://localhost:7860/health

# 4. Run a classify episode
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task": "classify"}'

curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"task": "classify", "data": {"category": "billing"}}'

# 5. Try dynamic mode
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task": "classify", "mode": "dynamic"}'
```

### Docker

```bash
docker build -t support-triage .
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_xxxx \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  support-triage
```

### Inference script

```bash
export HF_TOKEN=hf_xxxx
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export PING_URL=http://localhost:7860

python inference.py
```

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via HF Inference Router (static mode):

| Task | Score | Notes |
|------|-------|-------|
| classify | ~0.90 | Near-perfect with good prompting |
| prioritize | ~0.70 | Priority judgment occasionally off by 1 |
| escalate | ~0.65 | Escalation level calibration varies |
| sentiment_route | ~0.68 | vip_support routing occasionally missed |
| respond (3-step) | ~0.72 | LLM grader; KB usage in step 3 is key |
| **Average** | **~0.73** | |

---

## Project Structure

```
support-triage-env/
├── models.py         # Pydantic typed models for all 5 tasks
├── data.py           # 50-ticket corpus + dynamic LLM generation + KB articles
├── graders.py        # Deterministic + LLM-as-judge graders for all 5 tasks
├── environment.py    # SupportTriageEnv with static and dynamic modes
├── server.py         # FastAPI HTTP server (OpenEnv interface, port 7860)
├── inference.py      # Baseline LLM agent (all 5 tasks, dynamic mode for respond)
├── openenv.yaml      # OpenEnv metadata
├── pyproject.toml    # Project config + dependencies
├── uv.lock           # Locked dependencies
├── requirements.txt  # pip-compatible requirements
├── Dockerfile        # HF Spaces-compatible Docker image
└── tests/
    └── test_env.py   # Unit + integration tests
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| 5 tasks easy to hard | Clear difficulty progression for benchmarking agent capabilities |
| Partial credit rewards | Richer training signal than binary pass/fail |
| Multi-step respond | Mirrors real agent workflow; tests multi-turn planning |
| LLM-as-judge for respond | Industry best practice; more nuanced than keyword matching |
| Dynamic ticket generation | Prevents memorisation; tests true generalisation |
| Graceful fallbacks | Dynamic mode falls back to static if no API key available |
| Deterministic by episode_id | Reproducible evaluation across runs |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/test_env.py -v
```