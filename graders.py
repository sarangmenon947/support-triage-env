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

    total = round(0.6 * priority_score + 0.4 * team_score, 2)

    return {
        "score": total,
        "feedback": f"{pf} {tf}",
        "breakdown": {"priority": priority_score, "team_routing": team_score},
    }


# ─────────────────────────────────────────────
#  Task 3 — respond  (heuristic rubric)
# ─────────────────────────────────────────────

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


def grade_respond(
    action_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
    ticket_subject: str = "",
) -> Dict[str, Any]:
    """
    Heuristic rubric for the respond task.
    Four equally-weighted components (25% each):
      1. Issue acknowledged  — response mentions the ticket topic
      2. Solution provided   — response is substantive (>50 words)
      3. Empathy/tone        — contains at least one empathy phrase
      4. Proper closing      — contains a closing phrase
    """
    text = str(action_data.get("response_text", "")).lower()
    category = ground_truth["category"]
    words = text.split()

    kws = _CATEGORY_KEYWORDS.get(category, [])
    issue_score = 1.0 if any(kw in text for kw in kws) else 0.0

    if len(words) >= 80:
        solution_score = 1.0
    elif len(words) >= 50:
        solution_score = 0.6
    elif len(words) >= 20:
        solution_score = 0.3
    else:
        solution_score = 0.0

    empathy_score = 1.0 if any(p in text for p in _EMPATHY_PHRASES) else 0.0

    closing_score = 1.0 if any(p in text for p in _CLOSING_PHRASES) else 0.0

    total = round(
        0.25 * issue_score
        + 0.25 * solution_score
        + 0.25 * empathy_score
        + 0.25 * closing_score,
        2,
    )

    components = {
        "issue_acknowledged": issue_score,
        "solution_provided":  solution_score,
        "empathy_tone":       empathy_score,
        "proper_closing":     closing_score,
    }
    feedback = "; ".join(
        f"{k}={'✓' if v >= 1.0 else ('~' if v > 0 else '✗')}"
        for k, v in components.items()
    )

    return {"score": total, "feedback": feedback, "breakdown": components}


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