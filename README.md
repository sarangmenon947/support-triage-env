# Support Triage OpenEnv

> **Customer Support Ticket Triage** — an OpenEnv-compliant environment where AI agents learn to classify, prioritise, and respond to real-world support tickets.

---

## Why This Environment?

Every software company processes thousands of support tickets. Doing it well requires:
- **Understanding** the nature of each ticket (billing? technical? spam?)
- **Judgment** about urgency and the right team to handle it
- **Communication skill** to draft empathetic, accurate replies

This environment lets you train and evaluate agents on all three stages, using a corpus of 50 realistic synthetic tickets across 5 categories, with deterministic graders and partial-credit rewards.

---

## Tasks (Easy → Hard)

| # | Task ID | Difficulty | Description |
|---|---------|-----------|-------------|
| 1 | `classify` | Easy | Classify a ticket into one of 5 categories |
| 2 | `prioritize` | Medium | Assign P1–P4 priority + route to correct team |
| 3 | `respond` | Hard | Draft a full customer reply using knowledge-base articles |

### Task 1 — `classify`

**Observation:** Ticket (subject, body, customer metadata)  
**Action:** `{"category": "<billing|technical|account|feature_request|spam>"}`  
**Grading:**  
- 1.0 — exact category match  
- 0.4 — adjacent category (e.g. `billing` ↔ `account`)  
- 0.0 — wrong category  

### Task 2 — `prioritize`

**Observation:** Ticket + customer context (plan tier, tenure, sentiment)  
**Action:** `{"priority": "<P1|P2|P3|P4>", "assigned_team": "<team_name>"}`  
**Valid teams:** `billing_team`, `tech_support`, `account_team`, `product_team`, `spam_filter`  
**Grading:** 60% priority accuracy + 40% team routing  
- Priority: 1.0 (exact), 0.5 (off by 1 level), 0.0 (off by 2+)  
- Team: 1.0 (correct), 0.0 (wrong)  

### Task 3 — `respond`

**Observation:** Ticket + 3 knowledge-base articles  
**Action:** `{"response_text": "<full reply>"}`  
**Grading (4 × 25%):**  
1. Issue acknowledged (ticket topic mentioned in reply)  
2. Solution provided (substantive response, ≥ 80 words = full credit)  
3. Empathy/tone (contains at least one empathy phrase)  
4. Proper closing (professional sign-off present)  

---

## Observation & Action Spaces

### Observation (all tasks)
```json
{
  "task": "classify",
  "data": {
    "ticket": {
      "ticket_id": "TKT-ABCDEF",
      "subject": "Invoice shows double charge",
      "body": "Hi, I was charged twice ...",
      "customer_plan": "pro",
      "customer_since_days": 180,
      "previous_tickets": 2,
      "sentiment": "negative"
    },
    "valid_categories": ["billing", "technical", "account", "feature_request", "spam"]
  },
  "step": 0,
  "done": false
}
```

For `respond`, `data` also contains a `knowledge_base` array of articles.

### Action
```json
{ "task": "classify", "data": { "category": "billing" } }
```

---

## Reward Function

| Property | Value |
|---|---|
| Range | 0.0 – 1.0 |
| Episode type | Single-step |
| Partial credit | Yes (prioritize off-by-one, adjacent classify, respond rubric) |
| Deterministic | Yes — same episode_id always produces same ticket and same score |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step`  | Submit an action |
| `GET`  | `/state` | Get current episode state |
| `GET`  | `/tasks` | List all tasks |
| `GET`  | `/health`| Health check |

### /reset
```json
POST /reset
{ "task": "classify", "episode_id": "optional-seed-string" }
```

### /step
```json
POST /step
{ "task": "classify", "data": { "category": "billing" } }
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

# 3. In another terminal — run a quick smoke test
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task": "classify"}'

curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"task": "classify", "data": {"category": "billing"}}'
```

### Docker

```bash
docker build -t support-triage .
docker run -p 7860:7860 support-triage
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

## Deploying to Hugging Face Spaces

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Push (creates a Docker Space)
huggingface-cli repo create support-triage-env --type space --space-sdk docker
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/support-triage-env
git push hf main
```

Tag your Space with **`openenv`** in the HF Space settings.

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via HF Inference Router:

| Task | Score | Notes |
|------|-------|-------|
| classify | ~0.90 | Near-perfect with good prompting |
| prioritize | ~0.70 | Priority judgment occasionally off by 1 |
| respond | ~0.75 | Occasionally misses closing phrase |
| **Average** | **~0.78** | |

---

## Project Structure

```
support-triage-env/
├── models.py         # Pydantic typed models (Observation, Action, Reward, State)
├── data.py           # 50-ticket corpus + KB articles, deterministic by episode_id
├── graders.py        # Deterministic graders for all 3 tasks
├── environment.py    # SupportTriageEnv — reset() / step() / state()
├── server.py         # FastAPI HTTP server (OpenEnv interface)
├── inference.py      # Baseline LLM agent script
├── openenv.yaml      # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── tests/
    └── test_env.py   # Unit + integration tests
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/test_env.py -v
```

---