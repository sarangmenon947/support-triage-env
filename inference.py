"""
Inference Script — Support Triage OpenEnv
==========================================
Baseline agent that calls an LLM to solve all 5 tasks.
Uses external tools (search_kb, lookup_customer, check_order_status, get_similar_tickets)
during the respond task to demonstrate real tool-use capability.

MANDATORY environment variables:
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  The name of the local Docker image (used with from_docker_image()).

Defaults reflect the active inference setup:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

Stdout format (strictly followed):
    [START] task=<task> env=support_triage model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─────────────────────────────────────────────
#  Config  (matches mandatory checklist)
# ─────────────────────────────────────────────

# Defaults reflect the active inference setup
API_BASE_URL     = os.getenv("API_BASE_URL",     "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",       "Qwen/Qwen2.5-72B-Instruct")

# No default — must be supplied at runtime
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Use HF_TOKEN as the API key for all LLM calls
API_KEY  = HF_TOKEN or os.getenv("API_KEY", "")
PING_URL = os.getenv("PING_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = "support_triage"

TASKS = ["classify", "prioritize", "escalate", "sentiment_route", "respond"]
SUCCESS_THRESHOLD = 0.5      # score >= 0.5 counts as success

# ─────────────────────────────────────────────
#  Logging helpers  (exact required format)
# ─────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rw = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rw}", flush=True)

# ─────────────────────────────────────────────
#  Server helpers
# ─────────────────────────────────────────────

def server_reset(task: str, mode: str = "static", session_id: str = "default") -> Dict[str, Any]:
    r = requests.post(f"{PING_URL}/reset", json={"task": task, "mode": mode, "session_id": session_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def server_step(task: str, action_data: Dict[str, Any], session_id: str = "default") -> Dict[str, Any]:
    r = requests.post(f"{PING_URL}/step", json={"task": task, "data": action_data, "session_id": session_id}, timeout=60)
    r.raise_for_status()
    return r.json()


def server_tool_call(tool_name: str, tool_args: Dict[str, Any], session_id: str = "default") -> Dict[str, Any]:
    """Call a tool on the environment server."""
    try:
        r = requests.post(
            f"{PING_URL}/tool",
            json={"tool_name": tool_name, "tool_args": tool_args, "session_id": session_id},
            timeout=30
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────
#  Prompt builders
# ─────────────────────────────────────────────

def build_classify_prompt(obs_data: Dict[str, Any]) -> str:
    ticket = obs_data.get("ticket", {})
    cats   = obs_data.get("valid_categories", [])
    return textwrap.dedent(f"""
        You are a customer support manager. Classify the following support ticket.

        Subject: {ticket.get('subject', '')}
        Body: {ticket.get('body', '')}

        Valid categories: {', '.join(cats)}

        Respond with a JSON object only — no markdown, no extra text:
        {{"category": "<one of the valid categories>"}}
    """).strip()


def build_prioritize_prompt(obs_data: Dict[str, Any]) -> str:
    ticket    = obs_data.get("ticket", {})
    prios     = obs_data.get("valid_priorities", [])
    teams     = obs_data.get("valid_teams", [])
    return textwrap.dedent(f"""
        You are a support team lead. Assign priority and route this ticket.

        Subject:          {ticket.get('subject', '')}
        Body:             {ticket.get('body', '')}
        Customer plan:    {ticket.get('customer_plan', 'free')}
        Customer tenure:  {ticket.get('customer_since_days', 0)} days
        Previous tickets: {ticket.get('previous_tickets', 0)}
        Sentiment:        {ticket.get('sentiment', 'neutral')}

        Priority levels: {', '.join(prios)}  (P1=critical, P4=low)
        Teams:           {', '.join(teams)}

        Respond with a JSON object only — no markdown, no extra text:
        {{"priority": "<P1|P2|P3|P4>", "assigned_team": "<team_name>"}}
    """).strip()


def build_respond_step1_prompt(obs_data):
    ticket = obs_data.get("ticket", {})
    return (
        "You are a customer support agent. Read this ticket and ask ONE clarifying question "
        "to better understand the issue before responding.\n\n"
        f"Subject: {ticket.get('subject', '')}\n"
        f"Body: {ticket.get('body', '')}\n\n"
        "Respond with JSON only: {\"clarifying_question\": \"<your question here>\"}"
    )

def build_respond_step2_prompt(obs_data):
    ticket   = obs_data.get("ticket", {})
    question = obs_data.get("clarifying_question", "")
    answer   = obs_data.get("customer_answer", "")
    return (
        "You are a customer support agent. Draft a helpful response based on the clarification.\n\n"
        f"Ticket: {ticket.get('subject', '')}\n"
        f"Your question: {question}\n"
        f"Customer answer: {answer}\n\n"
        "Write a draft (50-100 words). Respond with JSON only: {\"draft_response\": \"<draft here>\"}"
    )

def build_respond_step3_prompt(obs_data):
    ticket = obs_data.get("ticket", {})
    draft  = obs_data.get("draft_response", "")
    kb     = obs_data.get("knowledge_base", [])
    kb_text = "\n".join(f"[{a['article_id']}] {a['title']}: {a['content']}" for a in kb)
    return (
        "You are a customer support agent. Refine your draft using the KB articles.\n\n"
        f"Ticket: {ticket.get('subject', '')}\n"
        f"Draft: {draft}\n\n"
        f"KB Articles:\n{kb_text}\n\n"
        "Write a polished final reply (80-150 words). Respond with JSON only: {\"response_text\": \"<final reply>\"}"
    )



def build_escalate_prompt(obs_data: dict) -> str:
    ticket  = obs_data.get("ticket", {})
    history = obs_data.get("conversation_history", [])
    attempts = obs_data.get("agent_attempts", 0)
    history_text = "\n".join(history) if history else "No prior conversation."
    return (
        "You are a support team lead. Decide if this ticket needs escalation.\n\n"
        f"Subject: {ticket.get('subject', '')}\n"
        f"Body: {ticket.get('body', '')}\n"
        f"Customer plan: {ticket.get('customer_plan', 'free')}\n"
        f"Sentiment: {ticket.get('sentiment', 'neutral')}\n"
        f"Agent attempts so far: {attempts}\n"
        f"Conversation history:\n{history_text}\n\n"
        "Valid escalation levels: none, L1, L2, L3, manager\n\n"
        "Respond with JSON only:\n"
        '{"should_escalate": true/false, "escalation_level": "<level>", "reason": "<brief reason>"}'
    )


def build_sentiment_route_prompt(obs_data: dict) -> str:
    ticket  = obs_data.get("ticket", {})
    score   = obs_data.get("sentiment_score", 0.0)
    keywords = obs_data.get("keywords_detected", [])
    return (
        "You are a support routing specialist. Route this emotionally charged ticket.\n\n"
        f"Subject: {ticket.get('subject', '')}\n"
        f"Body: {ticket.get('body', '')}\n"
        f"Customer plan: {ticket.get('customer_plan', 'free')}\n"
        f"Sentiment score: {score} (-1.0=very angry, 0=neutral, 1.0=very positive)\n"
        f"Anger keywords detected: {', '.join(keywords) if keywords else 'none'}\n\n"
        "Valid teams: billing_team, tech_support, account_team, product_team, spam_filter, vip_support\n"
        "Valid urgency flags: low, normal, high, critical\n"
        "(Use vip_support for angry high-value customers, critical for P1+angry)\n\n"
        "Respond with JSON only:\n"
        '{"assigned_team": "<team>", "urgency_flag": "<urgency>", "de_escalation_note": "<brief calming message for customer>"}'
    )

def call_llm(client: Optional[OpenAI], prompt: str) -> str:
    """Call the LLM and return raw response text. Returns empty string if client unavailable."""
    if client is None:
        return ""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON only."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERROR: {e}"


def parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        # strip opening and closing fence
        inner = [l for l in lines if not l.startswith("```")]
        clean = "\n".join(inner).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}

# ─────────────────────────────────────────────
#  Run one task episode
# ─────────────────────────────────────────────

def run_task(client: OpenAI, task: str) -> float:
    """Run a single episode. For respond: 3 steps. Others: 1 step."""
    log_start(task=task, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    try:
        # Use dynamic ticket generation for the respond task to showcase variety
        mode = "dynamic" if task == "respond" else "static"
        obs = server_reset(task, mode=mode)

        if task in ("classify", "prioritize", "escalate", "sentiment_route"):
            # Single-step tasks
            obs_data = obs.get("data", {})
            if task == "classify":
                prompt = build_classify_prompt(obs_data)
            elif task == "prioritize":
                prompt = build_prioritize_prompt(obs_data)
            elif task == "escalate":
                prompt = build_escalate_prompt(obs_data)
            else:
                prompt = build_sentiment_route_prompt(obs_data)

            raw = call_llm(client, prompt)
            action_data = parse_json(raw)
            if not action_data:
                error_msg = "JSON parse failed"
                action_data = {}

            result = server_step(task, action_data)
            reward = float(result.get("reward", 0.0))
            done   = bool(result.get("done", True))
            rewards.append(reward)
            steps_taken = 1
            score = reward
            log_step(step=1, action=json.dumps(action_data).replace(" ", ""),
                     reward=reward, done=done, error=error_msg)

        else:
            # respond — 3-step episode with tool use
            done = False
            step = 0
            current_obs = obs
            session_id = f"inference-{task}-{steps_taken}"

            while not done and step < 3:
                step += 1
                obs_data = current_obs.get("data", {})
                respond_step = obs_data.get("respond_step", step)

                # Before step 3 (refine), call tools to gather information
                tool_context = ""
                if respond_step == 3 or respond_step == 2:
                    ticket = obs_data.get("ticket", {})
                    ticket_id = ticket.get("ticket_id", "")

                    # Tool 1: search KB
                    kb_result = server_tool_call(
                        "search_kb",
                        {"query": ticket.get("subject", "")},
                        session_id=session_id
                    )
                    # Tool 2: lookup customer
                    customer_result = server_tool_call(
                        "lookup_customer",
                        {"ticket_id": ticket_id},
                        session_id=session_id
                    )
                    # Tool 3: get similar tickets
                    similar_result = server_tool_call(
                        "get_similar_tickets",
                        {"subject": ticket.get("subject", ""), "category": "technical"},
                        session_id=session_id
                    )

                    kb_articles = kb_result.get("results", [])
                    customer_info = customer_result.get("customer", {})
                    similar_tickets = similar_result.get("similar_tickets", [])

                    tool_context = (
                        f"\n\n--- TOOL RESULTS ---\n"
                        f"KB Articles found: {len(kb_articles)}\n"
                        + "\n".join(f"[{a.get('id','')}] {a.get('title','')}: {a.get('content','')}" for a in kb_articles[:2])
                        + f"\nCustomer health: {customer_info.get('account_health', 'unknown')}"
                        + f"\nSimilar resolved tickets: {len(similar_tickets)}"
                    )

                if respond_step == 1:
                    prompt = build_respond_step1_prompt(obs_data)
                elif respond_step == 2:
                    prompt = build_respond_step2_prompt(obs_data) + tool_context
                else:
                    prompt = build_respond_step3_prompt(obs_data) + tool_context

                raw = call_llm(client, prompt)
                action_data = parse_json(raw)
                if not action_data:
                    error_msg = "JSON parse failed"
                    action_data = {}

                result = server_step(task, action_data, session_id=session_id)
                reward = float(result.get("reward", 0.0))
                done   = bool(result.get("done", False))
                current_obs = result.get("observation", {})

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=json.dumps(action_data).replace(" ", ""),
                         reward=reward, done=done, error=error_msg)
                error_msg = None

            score = sum(rewards)

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        error_msg = str(e)
        log_step(step=max(steps_taken, 1), action="{}", reward=0.0,
                 done=True, error=error_msg)

    finally:
        log_end(success=success, steps=steps_taken, score=score,
                rewards=rewards if rewards else [0.0])

    return score


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    # Validate required environment variables before starting
    if not API_KEY:
        print("[WARN] HF_TOKEN/API_KEY not set. LLM calls will fail. "
              "Set HF_TOKEN environment variable.", flush=True)

    # Use a placeholder key if none provided — lets environment
    # calls proceed even if LLM calls fail gracefully
    safe_api_key = API_KEY if API_KEY else "placeholder-no-key-set"

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=safe_api_key)
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI client: {e}", flush=True)
        # Still run tasks so [START]/[STEP]/[END] logs are emitted
        client = None

    all_scores = {}
    for task in TASKS:
        try:
            score = run_task(client, task)
        except Exception as e:
            print(f"[ERROR] Task '{task}' failed: {e}", flush=True)
            score = 0.0
            # Emit required log lines even on failure
            log_start(task=task, model=MODEL_NAME)
            log_step(step=1, action="{}", reward=0.0, done=True, error=str(e))
            log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        all_scores[task] = score
        print(f"[INFO] Task '{task}' score: {score:.3f}", flush=True)

    avg = sum(all_scores.values()) / len(all_scores)
    print(f"[INFO] Average score across all tasks: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()