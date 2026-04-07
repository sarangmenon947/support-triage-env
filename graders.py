from typing import Dict, Any

# ─────────────────────────────────────────────
#  Constants & Shared Logic
# ─────────────────────────────────────────────

_CATEGORY_NEIGHBORS: Dict[str, list] = {
    "billing":         ["account"],
    "technical":       [],
    "account":         ["billing"],
    "feature_request": [],
    "spam":            [],
}

_PRIORITY_RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
_ESCALATION_RANK = {"none": 0, "L1": 1, "L2": 2, "L3": 3, "manager": 4}
_URGENCY_RANK = {"low": 0, "normal": 1, "high": 2, "critical": 3}

def _clamp_score(x: Any, eps: float = 0.01) -> float:
    """Strictly enforces (0, 1) range for all numerical outputs."""
    try:
        val = float(x)
    except (ValueError, TypeError):
        return eps
    if val != val:  # NaN check
        return eps
    # Hard clamp to [0.01, 0.99] to satisfy Phase 2
    return max(eps, min(1.0 - eps, val))

def _safe_breakdown(d: Dict[str, float]) -> Dict[str, float]:
    """Applies clamping to every key in a breakdown for total safety."""
    return {k: _clamp_score(v) for k, v in d.items()}

# ─────────────────────────────────────────────
#  Task 1 — Classify
# ─────────────────────────────────────────────

def grade_classify(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    pred = str(action_data.get("category", "")).strip().lower()
    corr = ground_truth["category"].lower()

    if pred == corr:
        score = 1.0
        msg = "Correct category."
    elif pred in _CATEGORY_NEIGHBORS.get(corr, []):
        score = 0.4
        msg = f"Adjacent category. Expected '{corr}', got '{pred}'."
    else:
        score = 0.0
        msg = f"Wrong category. Expected '{corr}', got '{pred}'."
    
    return {
        "score": _clamp_score(score),
        "feedback": msg,
        "breakdown": _safe_breakdown({"category_match": score})
    }

# ─────────────────────────────────────────────
#  Task 2 — Prioritize
# ─────────────────────────────────────────────

def grade_prioritize(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    pred_p = str(action_data.get("priority", "")).strip().upper()
    pred_t = str(action_data.get("assigned_team", "")).strip().lower()

    corr_p = ground_truth["priority"].upper()
    corr_t = ground_truth["assigned_team"].lower()

    p_diff = abs(_PRIORITY_RANK.get(pred_p, 99) - _PRIORITY_RANK.get(corr_p, 99))
    p_score = 1.0 if p_diff == 0 else (0.5 if p_diff == 1 else 0.0)
    t_score = 1.0 if pred_t == corr_t else 0.0

    total = (0.6 * p_score) + (0.4 * t_score)
    return {
        "score": _clamp_score(total),
        "feedback": f"Priority Match: {'Exact' if p_diff==0 else 'Partial'}, Team Match: {pred_t == corr_t}",
        "breakdown": _safe_breakdown({"priority": p_score, "team": t_score}),
    }

# ─────────────────────────────────────────────
#  Task 3 — Respond
# ─────────────────────────────────────────────

def grade_respond(action_data: Dict[str, Any], ground_truth: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    text = str(action_data.get("response_text", "")).lower()
    
    # 40% Issue Identification
    issue_kws = ["issue", "problem", "error", "bug", "trouble", "unable", "failed", "help"]
    issue_score = 1.0 if any(k in text for k in issue_kws) else 0.4
    
    # 30% Empathy & Tone
    emp_kws = ["sorry", "understand", "apolog", "frustrat", "inconvenien", "appreciate"]
    emp_score = 1.0 if any(w in text for w in emp_kws) else 0.0
    
    # 30% Clarity & Substance
    word_count = len(text.split())
    clar_score = 1.0 if word_count >= 40 else (0.5 if word_count >= 15 else 0.0)
    
    total = (0.4 * issue_score) + (0.3 * emp_score) + (0.3 * clar_score)
    return {
        "score": _clamp_score(total),
        "feedback": "Response graded on keyword presence and length heuristics.",
        "breakdown": _safe_breakdown({"issue": issue_score, "empathy": emp_score, "clarity": clar_score}),
    }

# ─────────────────────────────────────────────
#  Task 4 — Escalate
# ─────────────────────────────────────────────

def grade_escalate(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    pred_esc = bool(action_data.get("should_escalate", False))
    corr_esc = bool(ground_truth.get("should_escalate", False))
    
    dec_score = 0.5 if pred_esc == corr_esc else 0.0
    
    p_rank = _ESCALATION_RANK.get(str(action_data.get("escalation_level", "none")).lower(), 0)
    c_rank = _ESCALATION_RANK.get(str(ground_truth.get("escalation_level", "none")).lower(), 0)
    lev_score = 0.3 if p_rank == c_rank else (0.15 if abs(p_rank - c_rank) == 1 else 0.0)
    
    reason_len = len(str(action_data.get("reason", "")))
    re_score = 0.2 if reason_len >= 15 else (0.1 if reason_len >= 5 else 0.0)

    return {
        "score": _clamp_score(dec_score + lev_score + re_score),
        "feedback": f"Escalation Decision Match: {pred_esc == corr_esc}",
        "breakdown": _safe_breakdown({"decision": dec_score, "level": lev_score, "reason": re_score}),
    }

# ─────────────────────────────────────────────
#  Task 5 — Sentiment Route
# ─────────────────────────────────────────────

def grade_sentiment_route(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    # Team (40%)
    t_match = str(action_data.get("assigned_team", "")).lower() == str(ground_truth.get("assigned_team", "")).lower()
    t_score = 0.4 if t_match else 0.0
    
    # Urgency (40%)
    pred_u = str(action_data.get("urgency_flag", "")).lower()
    corr_u = str(ground_truth.get("urgency_flag", "normal")).lower()
    u_diff = abs(_URGENCY_RANK.get(pred_u, 1) - _URGENCY_RANK.get(corr_u, 1))
    u_score = 0.4 if u_diff == 0 else (0.2 if u_diff == 1 else 0.0)
    
    # Note (20%)
    n_len = len(str(action_data.get("de_escalation_note", "")))
    n_score = 0.2 if n_len >= 15 else (0.1 if n_len >= 5 else 0.0)

    return {
        "score": _clamp_score(t_score + u_score + n_score),
        "feedback": f"Team match: {t_match}, Urgency accuracy: {'High' if u_diff==0 else 'Low'}",
        "breakdown": _safe_breakdown({"team": t_score, "urgency": u_score, "note": n_score}),
    }

# ─────────────────────────────────────────────
#  Unified Entry Point
# ─────────────────────────────────────────────

def grade(task, action_data, ground_truth, **kwargs):
    tasks = {
        "classify": grade_classify,
        "prioritize": grade_prioritize,
        "respond": grade_respond,
        "escalate": grade_escalate,
        "sentiment_route": grade_sentiment_route
    }
    if task not in tasks:
        raise ValueError(f"Unknown task: {task}")
    return tasks[task](action_data, ground_truth, **kwargs)