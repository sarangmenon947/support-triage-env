"""
Graders for all 5 tasks in Support Triage OpenEnv.

All graders are deterministic and return a score in [0.0, 1.0].
All support partial credit — no binary pass/fail.

Weighting rationale:
  classify:        1.0 exact, 0.4 adjacent (billing/account are genuinely similar)
  prioritize:      60% priority (harder, more consequential) + 40% team routing
  escalate:        50% decision + 30% level + 20% reason (auditable)
  sentiment_route: 40% team + 40% urgency + 20% de-escalation note
  respond:         LLM-as-judge across 3 steps (clarify 0.3, draft 0.4, refine 0.3)
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


def _clamp_score(x: float, eps: float = 1e-3) -> float:
    """Clamp score to strictly (0, 1) — never exactly 0.0 or 1.0."""
    try:
        x = float(x)
    except Exception:
        return eps
    if x != x:  # NaN check
        return eps
    return max(eps, min(1.0 - eps, x))


def _clamp_breakdown(breakdown: Dict[str, float]) -> Dict[str, float]:
    """Clamp all values in a breakdown dict to strictly (0, 1)."""
    return {k: _clamp_score(v) for k, v in breakdown.items()}


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

    return {"score": _clamp_score(round(score, 2)), "feedback": feedback,
            "breakdown": _clamp_breakdown({"category_match": score})}


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

    # Priority score
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

    # Team score
    team_score = 1.0 if pred_team == correct_team else 0.0
    tf = "Team correct." if team_score == 1.0 else f"Wrong team. Expected '{correct_team}', got '{pred_team}'."

    # 60% priority: urgency judgment is harder and more consequential than routing
    # 40% team routing: can often be inferred mechanically from category
    total = round(0.6 * priority_score + 0.4 * team_score, 2)

    return {
        "score": _clamp_score(total),
        "feedback": f"{pf} {tf}",
        "breakdown": _clamp_breakdown({"priority": priority_score, "team_routing": team_score}),
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
        "score": _clamp_score(total),
        "feedback": f"heuristic; issue={'✓' if issue_score else '✗'} solution={'✓' if solution_score>=1 else '~'} empathy={'✓' if empathy_score else '✗'} closing={'✓' if closing_score else '✗'}",
        "breakdown": _clamp_breakdown({"issue_acknowledged": issue_score, "solution_quality": solution_score, "empathy_tone": empathy_score, "clarity_brevity": closing_score}),
    }


def _llm_respond(response_text: str, category: str, ticket_subject: str) -> Dict[str, Any]:
    """LLM-based grader — calls the same API used by inference.py."""
    try:
        from openai import OpenAI
        api_key      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
        api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        model_name   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

        if not api_key:
            return None  # no key available, fall back to heuristic

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
        # strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
        scores = _json.loads(raw)

        issue    = float(scores.get("issue_acknowledged", 0.0))
        solution = float(scores.get("solution_quality",   0.0))
        empathy  = float(scores.get("empathy_tone",       0.0))
        clarity  = float(scores.get("clarity_brevity",    0.0))
        reasoning = scores.get("reasoning", "")

        # clamp all to [0,1]
        issue, solution, empathy, clarity = (
            max(0.0, min(1.0, v)) for v in (issue, solution, empathy, clarity)
        )
        total = round(0.25 * issue + 0.25 * solution + 0.25 * empathy + 0.25 * clarity, 2)

        return {
            "score": _clamp_score(total),
            "feedback": f"llm_graded; {reasoning}",
            "breakdown": _clamp_breakdown({
                "issue_acknowledged": issue,
                "solution_quality":   solution,
                "empathy_tone":       empathy,
                "clarity_brevity":    clarity,
            }),
        }
    except Exception as e:
        return None  # fall back to heuristic on any error


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

    # Try LLM grader first
    result = _llm_respond(response_text, category, ticket_subject)

    # Fall back to heuristic if LLM unavailable
    if result is None:
        result = _heuristic_respond(response_text, category)

    return result


# ─────────────────────────────────────────────
#  Task 3 — respond (step graders)
# ─────────────────────────────────────────────

def grade_clarify(
    action_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
    ticket_subject: str = "",
) -> Dict[str, Any]:
    """
    Grade step 1 of respond: the clarifying question (max 0.30).

    Scoring:
      - is_question (0.15): Does the question contain "?"
      - relevance (0.15): Does it mention category-relevant keywords
      - Total: sum of both, capped at 0.30
    """
    question = str(action_data.get("clarifying_question", "")).strip().lower()
    category = ground_truth.get("category", "technical")
    keywords = _CATEGORY_KEYWORDS.get(category, [])

    # Check if it's a question (contains "?")
    is_question_score = 0.15 if "?" in question else 0.0

    # Check if question is relevant to category
    is_relevant = any(kw in question for kw in keywords)
    relevance_score = 0.15 if is_relevant else 0.0

    total = min(0.30, is_question_score + relevance_score)

    feedback = f"{'✓' if is_question_score else '✗'} question marker; {'✓' if relevance_score else '✗'} relevance"

    return {
        "score": _clamp_score(total),
        "feedback": feedback,
        "breakdown": _clamp_breakdown({
            "is_question": is_question_score,
            "relevance": relevance_score,
        }),
    }


def grade_draft(
    action_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
    ticket_subject: str = "",
    customer_answer: str = "",
) -> Dict[str, Any]:
    """
    Grade step 2 of respond: the draft response (max 0.40).

    Scoring:
      - addresses_reply (0.10): Does draft reference the customer_answer
      - solution_attempt (0.15): Category keywords + minimum length (30+ words)
      - empathy (0.10): Presence of empathy phrases
      - Total: sum of all, capped at 0.40
    """
    draft = str(action_data.get("draft_response", "")).strip().lower()
    customer_answer_lower = str(customer_answer).strip().lower()
    category = ground_truth.get("category", "technical")
    keywords = _CATEGORY_KEYWORDS.get(category, [])

    # Check if draft addresses the customer's answer
    addresses_score = 0.0
    if customer_answer_lower and any(word in draft for word in customer_answer_lower.split()[:5]):
        addresses_score = 0.10

    # Check for solution attempt (category keywords + length)
    has_keywords = any(kw in draft for kw in keywords)
    word_count = len(draft.split())
    solution_score = 0.15 if (has_keywords and word_count >= 30) else 0.0

    # Check for empathy
    empathy_score = 0.10 if any(p in draft for p in _EMPATHY_PHRASES) else 0.0

    total = min(0.40, addresses_score + solution_score + empathy_score)

    feedback = f"{'✓' if addresses_score else '✗'} address; {'✓' if solution_score else '✗'} solution; {'✓' if empathy_score else '✗'} empathy"

    return {
        "score": _clamp_score(total),
        "feedback": feedback,
        "breakdown": _clamp_breakdown({
            "addresses_reply": addresses_score,
            "solution_attempt": solution_score,
            "empathy": empathy_score,
        }),
    }


def grade_refine(
    action_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
    ticket_subject: str = "",
    draft_response: str = "",
) -> Dict[str, Any]:
    """
    Grade step 3 of respond: the refined response using KB (max 0.30).

    Scoring:
      - improvement (0.10): Is the response longer than the draft
      - kb_used (0.10): Uses knowledge-base terminology
      - closing (0.10): Professional closing phrase present
      - Total: sum of all, capped at 0.30
    """
    response = str(action_data.get("response_text", "")).strip().lower()
    draft_lower = str(draft_response).strip().lower() if draft_response else ""

    # Check if response is longer than draft (improvement)
    improvement_score = 0.10 if len(response) > len(draft_lower) else 0.0

    # Check for KB usage (look for certain KB-related keywords)
    # Common KB article titles might mention: "update", "refund", "reset", "configure", "invit"
    kb_keywords = ["update", "refund", "reset", "configure", "invit", "how to", "follow",
                   "process", "within", "days", "step", "click", "login", "settings"]
    kb_used = any(kw in response for kw in kb_keywords)
    kb_score = 0.10 if kb_used else 0.0

    # Check for professional closing
    closing_score = 0.10 if any(p in response for p in _CLOSING_PHRASES) else 0.0

    total = min(0.30, improvement_score + kb_score + closing_score)

    feedback = f"{'✓' if improvement_score else '✗'} improvement; {'✓' if kb_score else '✗'} KB usage; {'✓' if closing_score else '✗'} closing"

    return {
        "score": _clamp_score(total),
        "feedback": feedback,
        "breakdown": _clamp_breakdown({
            "improvement": improvement_score,
            "kb_used": kb_score,
            "closing": closing_score,
        }),
    }


# ─────────────────────────────────────────────
#  Task 4 — escalate
# ─────────────────────────────────────────────

_ESCALATION_LEVEL_RANK = {"none": 0, "L1": 1, "L2": 2, "L3": 3, "manager": 4}


def _score_escalation_reason(reason: str) -> float:
    """Score an escalation reason: non-trivial (1.0), short (0.5), empty (0.0)."""
    word_count = len(str(reason).split()) if reason else 0
    if word_count >= 20:
        return 1.0
    elif word_count >= 5:
        return 0.5
    else:
        return 0.0


def grade_escalate(
    action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score the escalate action (50% decision, 30% level, 20% reason).

    Decision:
      - Correct yes/no = 1.0
      - Wrong = 0.0

    Level:
      - Exact match = 1.0
      - Off by 1 = 0.5
      - Off by 2+ = 0.0

    Reason:
      - Non-trivial (20+ words) = 1.0
      - Short (5-19 words) = 0.5
      - Empty (< 5 words) = 0.0
    """
    pred_should_escalate = bool(action_data.get("should_escalate", False))
    pred_level = str(action_data.get("escalation_level", "none")).strip().lower()
    pred_reason = str(action_data.get("reason", "")).strip()

    correct_should_escalate = bool(ground_truth.get("should_escalate", False))
    correct_level = str(ground_truth.get("escalation_level", "none")).strip().lower()

    # Decision score
    decision_score = 1.0 if pred_should_escalate == correct_should_escalate else 0.0
    if decision_score == 1.0:
        df = "Escalation decision correct."
    else:
        df = f"Escalation decision wrong. Expected {'escalate' if correct_should_escalate else 'no escalation'}, got {'escalate' if pred_should_escalate else 'no escalation'}."

    # Level score
    pred_rank = _ESCALATION_LEVEL_RANK.get(pred_level, -1)
    correct_rank = _ESCALATION_LEVEL_RANK.get(correct_level, -1)
    if pred_rank == -1 or correct_rank == -1:
        level_score = 0.0
        lf = f"Invalid escalation level. Expected '{correct_level}', got '{pred_level}'."
    else:
        diff = abs(pred_rank - correct_rank)
        if diff == 0:
            level_score = 1.0
            lf = "Escalation level correct."
        elif diff == 1:
            level_score = 0.5
            lf = f"Escalation level off by 1. Expected '{correct_level}', got '{pred_level}'."
        else:
            level_score = 0.0
            lf = f"Escalation level wrong. Expected '{correct_level}', got '{pred_level}'."

    # Reason score
    reason_score = _score_escalation_reason(pred_reason)
    if reason_score == 1.0:
        rf = "Reason is detailed and clear."
    elif reason_score == 0.5:
        rf = "Reason is brief."
    else:
        rf = "Reason is missing or empty."

    total = round(0.5 * decision_score + 0.3 * level_score + 0.2 * reason_score, 2)

    return {
        "score": _clamp_score(total),
        "feedback": f"{df} {lf} {rf}",
        "breakdown": _clamp_breakdown({
            "decision": decision_score,
            "level": level_score,
            "reason": reason_score,
        }),
    }


# ─────────────────────────────────────────────
#  Task 5 — sentiment_route
# ─────────────────────────────────────────────

_URGENCY_RANK = {"low": 0, "normal": 1, "high": 2, "critical": 3}


def _score_de_escalation_note(note: str) -> float:
    """Score a de-escalation note: non-trivial (1.0), short (0.5), empty (0.0)."""
    word_count = len(str(note).split()) if note else 0
    if word_count >= 20:
        return 1.0
    elif word_count >= 5:
        return 0.5
    else:
        return 0.0


def grade_sentiment_route(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score the sentiment_route action (40% team, 40% urgency, 20% note).

    Team:
      - Correct team = 1.0
      - Wrong team = 0.0

    Urgency:
      - Exact match = 1.0
      - Off by 1 = 0.5
      - Off by 2+ = 0.0

    De-escalation note:
      - Non-trivial (20+ words) = 1.0
      - Short (5-19 words) = 0.5
      - Empty (< 5 words) = 0.0
    """
    pred_team = str(action_data.get("assigned_team", "")).strip().lower()
    pred_urgency = str(action_data.get("urgency_flag", "")).strip().lower()
    pred_note = str(action_data.get("de_escalation_note", "")).strip()

    correct_team = str(ground_truth.get("assigned_team", "")).strip().lower()
    correct_urgency = str(ground_truth.get("urgency_flag", "")).strip().lower()

    # Team score
    team_score = 1.0 if pred_team == correct_team else 0.0
    if team_score == 1.0:
        tf = "Team routing correct."
    else:
        tf = f"Wrong team. Expected '{correct_team}', got '{pred_team}'."

    # Urgency score
    pred_rank = _URGENCY_RANK.get(pred_urgency, -1)
    correct_rank = _URGENCY_RANK.get(correct_urgency, -1)
    if pred_rank == -1 or correct_rank == -1:
        urgency_score = 0.0
        uf = f"Invalid urgency flag. Expected '{correct_urgency}', got '{pred_urgency}'."
    else:
        diff = abs(pred_rank - correct_rank)
        if diff == 0:
            urgency_score = 1.0
            uf = "Urgency level correct."
        elif diff == 1:
            urgency_score = 0.5
            uf = f"Urgency level off by 1. Expected '{correct_urgency}', got '{pred_urgency}'."
        else:
            urgency_score = 0.0
            uf = f"Urgency level wrong. Expected '{correct_urgency}', got '{pred_urgency}'."

    # Note score
    note_score = _score_de_escalation_note(pred_note)
    if note_score == 1.0:
        nf = "De-escalation note is detailed."
    elif note_score == 0.5:
        nf = "De-escalation note is brief."
    else:
        nf = "De-escalation note is missing or empty."

    total = round(0.4 * team_score + 0.4 * urgency_score + 0.2 * note_score, 2)

    return {
        "score": _clamp_score(total),
        "feedback": f"{tf} {uf} {nf}",
        "breakdown": _clamp_breakdown({
            "team": team_score,
            "urgency": urgency_score,
            "note": note_score,
        }),
    }


# ─────────────────────────────────────────────
#  Unified grading entry point
# ─────────────────────────────────────────────

def grade(task: str, action_data: Dict[str, Any], ground_truth: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    if task == "classify":
        return grade_classify(action_data, ground_truth)
    elif task == "prioritize":
        return grade_prioritize(action_data, ground_truth)
    elif task == "escalate":
        return grade_escalate(action_data, ground_truth)
    elif task == "sentiment_route":
        return grade_sentiment_route(action_data, ground_truth)
    elif task == "respond":
        return grade_respond(action_data, ground_truth, **kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


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

    # Improvement over draft
    improvement_score = 0.1 if len(final.split()) > len(draft.split()) + 5 else (
        0.05 if len(final.split()) >= len(draft.split()) else 0.0
    )

    # KB-specific keywords (step names, article-specific terms)
    kb_terms = [
        "settings", "navigate", "click", "go to", "follow", "steps",
        "documentation", "refer to", "knowledge base", "article",
        "instructions", "guide", "process", "procedure",
    ]
    kb_score = 0.1 if any(term in final for term in kb_terms) else 0.0

    # Proper closing
    closing_score = 0.1 if any(p in final for p in _CLOSING_PHRASES) else 0.0

    total = round(improvement_score + kb_score + closing_score, 2)
    total = min(total, 0.3)

    feedback = f"refine: improved={'Y' if improvement_score>0 else 'N'} kb_used={'Y' if kb_score>0 else 'N'} closing={'Y' if closing_score>0 else 'N'}"

    return {
        "score": _clamp_score(total),
        "feedback": feedback,
        "breakdown": _clamp_breakdown({
            "improvement": improvement_score,
            "kb_used": kb_score,
            "closing": closing_score,
        }),
    }


# ─────────────────────────────────────────────
#  Task 4 — escalate grader
# ─────────────────────────────────────────────

_ESCALATION_RANK = {"none": 0, "L1": 1, "L2": 2, "L3": 3, "manager": 4}


def grade_escalate(
    action_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Grade the escalate task.
    - 50%: correct escalation decision (yes/no)
    - 30%: correct escalation level (off-by-one = partial)
    - 20%: reason provided (non-empty)
    """
    pred_escalate = bool(action_data.get("should_escalate", False))
    pred_level    = str(action_data.get("escalation_level", "none")).strip().lower()
    pred_reason   = str(action_data.get("reason", "")).strip()

    correct_escalate = ground_truth.get("should_escalate", False)
    correct_level    = ground_truth.get("escalation_level", "none").lower()

    # Decision score (50%)
    decision_score = 0.5 if pred_escalate == correct_escalate else 0.0

    # Level score (30%)
    pred_rank    = _ESCALATION_RANK.get(pred_level, 0)
    correct_rank = _ESCALATION_RANK.get(correct_level, 0)
    diff = abs(pred_rank - correct_rank)
    if diff == 0:
        level_score = 0.3
    elif diff == 1:
        level_score = 0.15
    else:
        level_score = 0.0

    # Reason score (20%)
    reason_score = 0.2 if len(pred_reason) >= 10 else (0.1 if pred_reason else 0.0)

    total = round(decision_score + level_score + reason_score, 2)
    feedback = (
        f"escalate: decision={'Y' if decision_score else 'N'} "
        f"level={'exact' if diff==0 else ('close' if diff==1 else 'wrong')} "
        f"reason={'Y' if reason_score>=0.2 else 'N'}"
    )

    return {
        "score": _clamp_score(total),
        "feedback": feedback,
        "breakdown": _clamp_breakdown({
            "decision": decision_score,
            "level": level_score,
            "reason": reason_score,
        }),
    }


# ─────────────────────────────────────────────
#  Task 5 — sentiment_route grader
# ─────────────────────────────────────────────

def grade_sentiment_route(
    action_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Grade the sentiment_route task.
    - 40%: correct team assignment
    - 40%: correct urgency flag (off-by-one = partial)
    - 20%: de-escalation note present and non-trivial
    """
    pred_team    = str(action_data.get("assigned_team", "")).strip().lower()
    pred_urgency = str(action_data.get("urgency_flag", "")).strip().lower()
    pred_note    = str(action_data.get("de_escalation_note", "")).strip()

    correct_team    = ground_truth.get("assigned_team", "").lower()
    correct_urgency = ground_truth.get("urgency_flag", "normal").lower()

    # Team score (40%)
    team_score = 0.4 if pred_team == correct_team else 0.0

    # Urgency score (40%)
    _urgency_rank = {"low": 0, "normal": 1, "high": 2, "critical": 3}
    pred_rank    = _urgency_rank.get(pred_urgency, 1)
    correct_rank = _urgency_rank.get(correct_urgency, 1)
    diff = abs(pred_rank - correct_rank)
    if diff == 0:
        urgency_score = 0.4
    elif diff == 1:
        urgency_score = 0.2
    else:
        urgency_score = 0.0

    # De-escalation note (20%)
    note_score = 0.2 if len(pred_note) >= 15 else (0.1 if pred_note else 0.0)

    total = round(team_score + urgency_score + note_score, 2)
    feedback = (
        f"sentiment_route: team={'Y' if team_score else 'N'} "
        f"urgency={'exact' if diff==0 else ('close' if diff==1 else 'wrong')} "
        f"note={'Y' if note_score>=0.2 else 'N'}"
    )

    return {
        "score": _clamp_score(total),
        "feedback": feedback,
        "breakdown": _clamp_breakdown({
            "team": team_score,
            "urgency": urgency_score,
            "de_escalation_note": note_score,
        }),
    }