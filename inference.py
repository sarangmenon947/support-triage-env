"""
Inference Script — Support Triage OpenEnv
==========================================
Baseline agent that calls an LLM to solve all three tasks.

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
#  Config 
# ─────────────────────────────────────────────

API_BASE_URL     = os.getenv("API_BASE_URL",     "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",       "Qwen/Qwen2.5-72B-Instruct")

HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_KEY  = HF_TOKEN or os.getenv("API_KEY", "")
PING_URL = os.getenv("PING_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = "support_triage"

TASKS = ["classify", "prioritize", "respond"]
SUCCESS_THRESHOLD = 0.5      

# ─────────────────────────────────────────────
#  Logging helpers
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

def server_reset(task: str) -> Dict[str, Any]:
    r = requests.post(f"{PING_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()

def server_step(task: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{PING_URL}/step", json={"task": task, "data": action_data}, timeout=60)
    r.raise_for_status()
    return r.json()

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


def build_respond_prompt(obs_data: Dict[str, Any]) -> str:
    ticket = obs_data.get("ticket", {})
    kb     = obs_data.get("knowledge_base", [])
    kb_text = "\n\n".join(
        f"[{a['article_id']}] {a['title']}\n{a['content']}" for a in kb
    )
    return textwrap.dedent(f"""
        You are a customer support agent. Write a helpful, empathetic reply to this ticket.
        Use the knowledge-base articles below where relevant.

        --- TICKET ---
        Subject: {ticket.get('subject', '')}
        Body:    {ticket.get('body', '')}
        Customer plan:  {ticket.get('customer_plan', 'free')}
        Sentiment:      {ticket.get('sentiment', 'neutral')}

        --- KNOWLEDGE BASE ---
        {kb_text}

        Write a professional support reply (80–150 words).
        Respond with a JSON object only — no markdown, no extra text:
        {{"response_text": "<your full reply here>"}}
    """).strip()


def call_llm(client: OpenAI, prompt: str) -> str:
    """Call the LLM and return raw response text."""
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

def run_task(client: OpenAI, task: str) -> float:
    """Run a single episode of the given task. Returns episode score."""
    log_start(task=task, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    try:
        obs = server_reset(task)
        obs_data = obs.get("data", {})

        if task == "classify":
            prompt = build_classify_prompt(obs_data)
        elif task == "prioritize":
            prompt = build_prioritize_prompt(obs_data)
        else:
            prompt = build_respond_prompt(obs_data)

        raw_response = call_llm(client, prompt)
        action_data  = parse_json(raw_response)

        if not action_data:
            error_msg = "JSON parse failed"
            action_data = {}

        result = server_step(task, action_data)
        reward  = float(result.get("reward", 0.0))
        done    = bool(result.get("done", True))
        info    = result.get("info", {})
        error_msg = error_msg  

        rewards.append(reward)
        steps_taken = 1
        score = reward

        action_str = json.dumps(action_data).replace(" ", "")
        log_step(step=1, action=action_str, reward=reward, done=done, error=error_msg)

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        error_msg = str(e)
        log_step(step=max(steps_taken, 1), action="{}", reward=0.0, done=True, error=error_msg)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards if rewards else [0.0])

    return score


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = {}
    for task in TASKS:
        score = run_task(client, task)
        all_scores[task] = score
        print(f"[INFO] Task '{task}' score: {score:.3f}", flush=True)

    avg = sum(all_scores.values()) / len(all_scores)
    print(f"[INFO] Average score across all tasks: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()