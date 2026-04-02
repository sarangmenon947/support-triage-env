"""
Graders for all three tasks.
All graders are deterministic and return a float in [0.0, 1.0].
"""

from typing import Dict, Any


# ─────────────────────────────────────────────
#  Category adjacency map for partial credit
# ─────────────────────────────────────────────

_CATEGORY_NEIGHBORS: Dict[str, list] = {
    "billing":         ["account"],
    "technical":       [],
    "account":         ["billing"],
    "feature_request": [],
    "spam":            [],
}

_PRIORITY_RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

_TEAM_FOR_CATEGORY = {
    "billing":         "billing_team",
    "technical":       "tech_support",
    "account":         "account_team",
    "feature_request": "product_team",
    "spam":            "spam_filter",
}


# ─────────────────────────────────────────────
#  Task 1 — classify
# ─────────────────────────────────────────────

def grade_classify(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score the classify action.
    - 1.0 for exact match
    - 0.4 for adjacent category (billing ↔ account)
    - 0.0 otherwise
    """
    predicted = str(action_data.get("category", "")).strip().lower()
    correct   = ground_truth["category"].lower()

    if predicted == correct:
        score = 1.0
        feedback = "Correct category."
    elif predicted in _CATEGORY_NEIGHBORS.get(correct, []):
        score = 0.4
        feedback = f"Adjacent category. Expected '{correct}', got '{predicted}'."
    else:
        score = 0.0
        feedback = f"Wrong category. Expected '{correct}', got '{predicted}'."

    return {"score": round(score, 2), "feedback": feedback,
            "breakdown": {"category_match": score}}


# ─────────────────────────────────────────────
#  Task 2 — prioritize
# ─────────────────────────────────────────────

def grade_prioritize(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score the prioritize action (60% priority, 40% team routing).

    Priority:
      - Exact match = 1.0
      - Off by 1 level = 0.5
      - Off by 2+ = 0.0

    Team:
      - Correct team = 1.0
      - Wrong team = 0.0
    """
    pred_priority = str(action_data.get("priority", "")).strip().upper()
    pred_team     = str(action_data.get("assigned_team", "")).strip().lower()

    correct_priority = ground_truth["priority"].upper()
    correct_team     = ground_truth["assigned_team"].lower()

    pred_rank    = _PRIORITY_RANK.get(pred_priority, 99)
    correct_rank = _PRIORITY_RANK.get(correct_priority, 99)
    diff = abs(pred_rank - correct_rank)

    if diff == 0:
        priority_score = 1.0
        pf = "Priority correct."
    elif diff == 1:
        priority_score = 0.5
        pf = f"Priority off by 1. Expected {correct_priority}, got {pred_priority}."
    else:
        priority_score = 0.0
        pf = f"Priority wrong. Expected {correct_priority}, got {pred_priority}."

    team_score = 1.0 if pred_team == correct_team else 0.0
    tf = "Team correct." if team_score == 1.0 else f"Wrong team. Expected '{correct_team}', got '{pred_team}'."

    total = round(0.6 * priority_score + 0.4 * team_score, 2)

    return {
        "score": total,
        "feedback": f"{pf} {tf}",
        "breakdown": {"priority": priority_score, "team_routing": team_score},
    }


# ─────────────────────────────────────────────
#  Task 3 — respond  (LLM-based grader with heuristic fallback)
# ─────────────────────────────────────────────

import os
import json as _json

_EMPATHY_PHRASES = [
    "understand", "sorry", "apologise", "apology", "frustrat", "inconvenien",
    "appreciate", "thank you for", "we're here to help", "happy to help",
]

_CLOSING_PHRASES = [
    "let me know", "feel free", "don't hesitate", "reach out", "any questions",
    "here to help", "best regards", "kind regards", "sincerely",
]

_CATEGORY_KEYWORDS: Dict[str, list] = {
    "billing":         ["invoice", "charge", "payment", "refund", "billing", "receipt"],
    "technical":       ["error", "issue", "fix", "troubleshoot", "technical", "support", "problem"],
    "account":         ["account", "password", "login", "access", "settings", "profile"],
    "feature_request": ["feedback", "feature", "team", "future", "roadmap", "noted"],
    "spam":            ["spam", "flag", "reported", "block", "review"],
}

_LLM_GRADE_PROMPT = """You are an expert customer support quality analyst.
Score the following support response on four criteria.
Each score must be a float between 0.0 and 1.0.

TICKET SUBJECT: {subject}
TICKET CATEGORY: {category}
AGENT RESPONSE: {response}

Score these four criteria:
1. issue_acknowledged (0.0-1.0): Did the agent clearly identify and address the customer's specific issue?
2. solution_quality (0.0-1.0): Did the agent provide a helpful, accurate, and actionable solution?
3. empathy_tone (0.0-1.0): Was the tone empathetic, professional, and customer-friendly?
4. clarity_brevity (0.0-1.0): Was the response clear, well-structured, and appropriately concise?

Respond with valid JSON only, no markdown:
{{"issue_acknowledged": 0.0, "solution_quality": 0.0, "empathy_tone": 0.0, "clarity_brevity": 0.0, "reasoning": "brief explanation"}}"""


def _heuristic_respond(text: str, category: str) -> Dict[str, Any]:
    """Fallback heuristic grader used when LLM grader is unavailable."""
    words = text.lower().split()
    kws = _CATEGORY_KEYWORDS.get(category, [])
    issue_score = 1.0 if any(kw in text.lower() for kw in kws) else 0.0
    if len(words) >= 80:
        solution_score = 1.0
    elif len(words) >= 50:
        solution_score = 0.6
    elif len(words) >= 20:
        solution_score = 0.3
    else:
        solution_score = 0.0
    empathy_score = 1.0 if any(p in text.lower() for p in _EMPATHY_PHRASES) else 0.0
    closing_score = 1.0 if any(p in text.lower() for p in _CLOSING_PHRASES) else 0.0
    total = round(0.25 * issue_score + 0.25 * solution_score + 0.25 * empathy_score + 0.25 * closing_score, 2)
    return {
        "score": total,
        "feedback": f"heuristic; issue={'✓' if issue_score else '✗'} solution={'✓' if solution_score>=1 else '~'} empathy={'✓' if empathy_score else '✗'} closing={'✓' if closing_score else '✗'}",
        "breakdown": {"issue_acknowledged": issue_score, "solution_quality": solution_score, "empathy_tone": empathy_score, "clarity_brevity": closing_score},
    }


def _llm_respond(response_text: str, category: str, ticket_subject: str) -> Dict[str, Any]:
    """LLM-based grader — calls the same API used by inference.py."""
    try:
        from openai import OpenAI
        api_key      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
        api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        model_name   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

        if not api_key:
            return None \

        client = OpenAI(base_url=api_base_url, api_key=api_key)
        prompt = _LLM_GRADE_PROMPT.format(
            subject=ticket_subject,
            category=category,
            response=response_text,
        )
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a support quality analyst. Always respond with valid JSON only."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = (completion.choices[0].message.content or "").strip()

        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
        scores = _json.loads(raw)

        issue    = float(scores.get("issue_acknowledged", 0.0))
        solution = float(scores.get("solution_quality",   0.0))
        empathy  = float(scores.get("empathy_tone",       0.0))
        clarity  = float(scores.get("clarity_brevity",    0.0))
        reasoning = scores.get("reasoning", "")

        issue, solution, empathy, clarity = (
            max(0.0, min(1.0, v)) for v in (issue, solution, empathy, clarity)
        )
        total = round(0.25 * issue + 0.25 * solution + 0.25 * empathy + 0.25 * clarity, 2)

        return {
            "score": total,
            "feedback": f"llm_graded; {reasoning}",
            "breakdown": {
                "issue_acknowledged": issue,
                "solution_quality":   solution,
                "empathy_tone":       empathy,
                "clarity_brevity":    clarity,
            },
        }
    except Exception as e:
        return None  


def grade_respond(
    action_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
    ticket_subject: str = "",
) -> Dict[str, Any]:
    """
    LLM-based rubric grader for the respond task.
    Uses an LLM to score four criteria (25% each):
      1. Issue acknowledged  — agent identified the customer's problem
      2. Solution quality    — helpful and actionable solution provided
      3. Empathy/tone        — professional and empathetic tone
      4. Clarity/brevity     — clear, well-structured, concise response

    Falls back to heuristic grader if LLM is unavailable.
    """
    response_text = str(action_data.get("response_text", ""))
    category      = ground_truth["category"]

    result = _llm_respond(response_text, category, ticket_subject)

    if result is None:
        result = _heuristic_respond(response_text, category)

    return result


# ─────────────────────────────────────────────
#  Unified grading entry point
# ─────────────────────────────────────────────

def grade(task: str, action_data: Dict[str, Any], ground_truth: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    if task == "classify":
        return grade_classify(action_data, ground_truth)
    elif task == "prioritize":
        return grade_prioritize(action_data, ground_truth)
    elif task == "respond":
        return grade_respond(action_data, ground_truth, **kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


# ─────────────────────────────────────────────
#  Multi-step respond graders
# ─────────────────────────────────────────────

_QUESTION_INDICATORS = [
    "?", "what", "when", "where", "which", "who", "how", "could you",
    "can you", "did you", "have you", "please clarify", "please provide",
    "could you tell", "would you mind",
]

_CATEGORY_QUESTION_KEYWORDS: dict = {
    "billing":         ["charge", "invoice", "payment", "amount", "date", "receipt", "refund"],
    "technical":       ["error", "message", "browser", "device", "version", "steps", "when"],
    "account":         ["email", "username", "access", "role", "when", "browser", "device"],
    "feature_request": ["team", "use case", "how often", "workflow", "currently", "benefit"],
    "spam":            ["intentional", "account", "verify", "sent", "who"],
}


def grade_clarify(
    action_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
    ticket_subject: str = "",
) -> Dict[str, Any]:
    """
    Grade Step 1: Did the agent ask a good clarifying question?
    Max reward: 0.3

    Criteria:
    - Is it actually a question? (0.15)
    - Is it relevant to the ticket category? (0.15)
    """
    question = str(action_data.get("clarifying_question", "")).lower().strip()
    category = ground_truth.get("category", "")

    is_question = any(ind in question for ind in _QUESTION_INDICATORS)
    question_score = 0.15 if is_question else 0.0

    kws = _CATEGORY_QUESTION_KEYWORDS.get(category, [])
    is_relevant = any(kw in question for kw in kws)
    relevance_score = 0.15 if is_relevant else 0.0

    total = round(question_score + relevance_score, 2)
    feedback = f"clarify: is_question={'Y' if is_question else 'N'} relevant={'Y' if is_relevant else 'N'}"

    return {
        "score": total,
        "feedback": feedback,
        "breakdown": {"is_question": question_score, "relevance": relevance_score},
    }


def grade_draft(
    action_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
    ticket_subject: str = "",
    customer_answer: str = "",
) -> Dict[str, Any]:
    """
    Grade Step 2: Did the agent write a good draft response?
    Max reward: 0.4

    Criteria:
    - Addresses the customer answer (0.1)
    - Contains solution attempt (0.15)
    - Has empathetic tone (0.15)
    """
    draft = str(action_data.get("draft_response", "")).lower().strip()
    category = ground_truth.get("category", "")
    words = draft.split()

    answer_words = set(customer_answer.lower().split())
    draft_words  = set(draft.split())
    overlap = len(answer_words & draft_words)
    address_score = 0.1 if overlap >= 3 else (0.05 if overlap >= 1 else 0.0)

    kws = _CATEGORY_KEYWORDS.get(category, [])
    has_kw = any(kw in draft for kw in kws)
    has_length = len(words) >= 30
    solution_score = 0.15 if (has_kw and has_length) else (0.08 if has_length else 0.0)

    empathy_score = 0.1 if any(p in draft for p in _EMPATHY_PHRASES) else 0.0

    total = round(address_score + solution_score + empathy_score, 2)
    total = min(total, 0.4)

    feedback = f"draft: addresses_reply={'Y' if address_score>0 else 'N'} solution={'Y' if solution_score>0 else 'N'} empathy={'Y' if empathy_score>0 else 'N'}"

    return {
        "score": total,
        "feedback": feedback,
        "breakdown": {
            "addresses_reply": address_score,
            "solution_attempt": solution_score,
            "empathy": empathy_score,
        },
    }


def grade_refine(
    action_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
    ticket_subject: str = "",
    draft_response: str = "",
) -> Dict[str, Any]:
    """
    Grade Step 3: Did the agent improve the draft using KB articles?
    Max reward: 0.3

    Criteria:
    - Response is longer/better than draft (0.1)
    - Uses KB-specific information (0.1)
    - Has proper closing (0.1)
    """
    final = str(action_data.get("response_text", "")).lower().strip()
    draft = draft_response.lower().strip()

    improvement_score = 0.1 if len(final.split()) > len(draft.split()) + 5 else (
        0.05 if len(final.split()) >= len(draft.split()) else 0.0
    )

    kb_terms = [
        "settings", "navigate", "click", "go to", "follow", "steps",
        "documentation", "refer to", "knowledge base", "article",
        "instructions", "guide", "process", "procedure",
    ]
    kb_score = 0.1 if any(term in final for term in kb_terms) else 0.0
    closing_score = 0.1 if any(p in final for p in _CLOSING_PHRASES) else 0.0

    total = round(improvement_score + kb_score + closing_score, 2)
    total = min(total, 0.3)

    feedback = f"refine: improved={'Y' if improvement_score>0 else 'N'} kb_used={'Y' if kb_score>0 else 'N'} closing={'Y' if closing_score>0 else 'N'}"

    return {
        "score": total,
        "feedback": feedback,
        "breakdown": {
            "improvement": improvement_score,
            "kb_used": kb_score,
            "closing": closing_score,
        },
    }