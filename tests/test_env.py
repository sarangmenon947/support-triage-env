"""
Comprehensive test suite for Support Triage OpenEnv.
Covers all 5 tasks: classify, prioritize, escalate, sentiment_route, respond.

Run with: python -m pytest tests/test_env.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from graders import (
    grade_classify, grade_prioritize, grade_escalate,
    grade_sentiment_route, grade_clarify, grade_draft, grade_refine
)
from environment import SupportTriageEnv
from models import Action


# ─────────────────────────────────────────────
#  Task 1 — classify grader
# ─────────────────────────────────────────────

class TestClassifyGrader:
    def test_exact_match(self):
        r = grade_classify({"category": "billing"}, {"category": "billing"})
        assert r["score"] == 1.0

    def test_adjacent_category_billing_account(self):
        r = grade_classify({"category": "account"}, {"category": "billing"})
        assert r["score"] == 0.4

    def test_adjacent_category_account_billing(self):
        r = grade_classify({"category": "billing"}, {"category": "account"})
        assert r["score"] == 0.4

    def test_wrong_category(self):
        r = grade_classify({"category": "spam"}, {"category": "billing"})
        assert r["score"] == 0.0

    def test_case_insensitive(self):
        r = grade_classify({"category": "BILLING"}, {"category": "billing"})
        assert r["score"] == 1.0

    def test_missing_field_returns_zero(self):
        r = grade_classify({}, {"category": "billing"})
        assert r["score"] == 0.0

    def test_all_five_categories_exact(self):
        for cat in ["billing", "technical", "account", "feature_request", "spam"]:
            r = grade_classify({"category": cat}, {"category": cat})
            assert r["score"] == 1.0, f"Failed for: {cat}"

    def test_score_always_in_range(self):
        for cat in ["billing", "technical", "account", "feature_request", "spam", "garbage"]:
            r = grade_classify({"category": cat}, {"category": "billing"})
            assert 0.0 <= r["score"] <= 1.0


# ─────────────────────────────────────────────
#  Task 2 — prioritize grader
# ─────────────────────────────────────────────

class TestPrioritizeGrader:
    def test_exact_match(self):
        r = grade_prioritize(
            {"priority": "P1", "assigned_team": "billing_team"},
            {"priority": "P1", "assigned_team": "billing_team"},
        )
        assert r["score"] == 1.0

    def test_priority_off_by_one(self):
        r = grade_prioritize(
            {"priority": "P2", "assigned_team": "billing_team"},
            {"priority": "P1", "assigned_team": "billing_team"},
        )
        # 0.6*0.5 + 0.4*1.0 = 0.70
        assert r["score"] == pytest.approx(0.7, abs=0.01)

    def test_priority_off_by_two_is_zero(self):
        r = grade_prioritize(
            {"priority": "P3", "assigned_team": "billing_team"},
            {"priority": "P1", "assigned_team": "billing_team"},
        )
        # 0.6*0.0 + 0.4*1.0 = 0.40
        assert r["score"] == pytest.approx(0.4, abs=0.01)

    def test_wrong_priority_wrong_team(self):
        r = grade_prioritize(
            {"priority": "P4", "assigned_team": "product_team"},
            {"priority": "P1", "assigned_team": "billing_team"},
        )
        assert r["score"] == 0.0

    def test_correct_priority_wrong_team(self):
        r = grade_prioritize(
            {"priority": "P1", "assigned_team": "product_team"},
            {"priority": "P1", "assigned_team": "billing_team"},
        )
        # 0.6*1.0 + 0.4*0.0 = 0.60
        assert r["score"] == pytest.approx(0.6, abs=0.01)

    def test_score_always_in_range(self):
        r = grade_prioritize(
            {"priority": "P3", "assigned_team": "tech_support"},
            {"priority": "P2", "assigned_team": "tech_support"},
        )
        assert 0.0 <= r["score"] <= 1.0

    def test_breakdown_keys_present(self):
        r = grade_prioritize(
            {"priority": "P1", "assigned_team": "billing_team"},
            {"priority": "P1", "assigned_team": "billing_team"},
        )
        assert "priority" in r["breakdown"]
        assert "team_routing" in r["breakdown"]


# ─────────────────────────────────────────────
#  Task 3 — escalate grader
# ─────────────────────────────────────────────

class TestEscalateGrader:
    def test_perfect_score(self):
        r = grade_escalate(
            {"should_escalate": True, "escalation_level": "L2",
             "reason": "Customer is angry and issue unresolved after 3 attempts"},
            {"should_escalate": True, "escalation_level": "L2"},
        )
        assert r["score"] == 1.0

    def test_wrong_decision(self):
        r = grade_escalate(
            {"should_escalate": False, "escalation_level": "none", "reason": "Looks fine"},
            {"should_escalate": True, "escalation_level": "L2"},
        )
        # decision wrong = 0, level off by 2 = 0, reason present = 0.2
        assert r["score"] == pytest.approx(0.2, abs=0.01)

    def test_correct_decision_level_off_by_one(self):
        r = grade_escalate(
            {"should_escalate": True, "escalation_level": "L1",
             "reason": "Needs escalation due to repeated failures"},
            {"should_escalate": True, "escalation_level": "L2"},
        )
        # decision=0.5, level off by 1=0.15, reason=0.2
        assert r["score"] == pytest.approx(0.85, abs=0.01)

    def test_no_reason_loses_points(self):
        r = grade_escalate(
            {"should_escalate": True, "escalation_level": "L2", "reason": ""},
            {"should_escalate": True, "escalation_level": "L2"},
        )
        assert r["score"] == pytest.approx(0.8, abs=0.01)

    def test_score_in_range(self):
        r = grade_escalate(
            {"should_escalate": False, "escalation_level": "manager", "reason": "x"},
            {"should_escalate": True, "escalation_level": "L1"},
        )
        assert 0.0 <= r["score"] <= 1.0

    def test_breakdown_keys_present(self):
        r = grade_escalate(
            {"should_escalate": True, "escalation_level": "L1", "reason": "test"},
            {"should_escalate": True, "escalation_level": "L1"},
        )
        assert "decision" in r["breakdown"]
        assert "level" in r["breakdown"]
        assert "reason" in r["breakdown"]


# ─────────────────────────────────────────────
#  Task 4 — sentiment_route grader
# ─────────────────────────────────────────────

class TestSentimentRouteGrader:
    def test_perfect_score(self):
        r = grade_sentiment_route(
            {"assigned_team": "billing_team", "urgency_flag": "high",
             "de_escalation_note": "I completely understand your frustration and will resolve this now"},
            {"assigned_team": "billing_team", "urgency_flag": "high"},
        )
        assert r["score"] == 1.0

    def test_wrong_team(self):
        r = grade_sentiment_route(
            {"assigned_team": "tech_support", "urgency_flag": "high",
             "de_escalation_note": "We apologise for the inconvenience"},
            {"assigned_team": "billing_team", "urgency_flag": "high"},
        )
        # team wrong=0, urgency exact=0.4, note=0.2
        assert r["score"] == pytest.approx(0.6, abs=0.01)

    def test_urgency_off_by_one(self):
        r = grade_sentiment_route(
            {"assigned_team": "billing_team", "urgency_flag": "normal",
             "de_escalation_note": "We are here to help you"},
            {"assigned_team": "billing_team", "urgency_flag": "high"},
        )
        # team=0.4, urgency off by 1=0.2, note=0.2
        assert r["score"] == pytest.approx(0.8, abs=0.01)

    def test_missing_de_escalation_note(self):
        r = grade_sentiment_route(
            {"assigned_team": "billing_team", "urgency_flag": "high",
             "de_escalation_note": ""},
            {"assigned_team": "billing_team", "urgency_flag": "high"},
        )
        # team=0.4, urgency=0.4, note missing=0
        assert r["score"] == pytest.approx(0.8, abs=0.01)

    def test_score_in_range(self):
        r = grade_sentiment_route(
            {"assigned_team": "spam_filter", "urgency_flag": "critical",
             "de_escalation_note": "hi"},
            {"assigned_team": "vip_support", "urgency_flag": "low"},
        )
        assert 0.0 <= r["score"] <= 1.0


# ─────────────────────────────────────────────
#  Task 5 — respond multi-step graders
# ─────────────────────────────────────────────

class TestRespondGraders:
    def test_clarify_good_question(self):
        r = grade_clarify(
            {"clarifying_question": "Can you tell me the date and amount of the charge?"},
            {"category": "billing"},
        )
        assert r["score"] == pytest.approx(0.3, abs=0.01)

    def test_clarify_not_a_question(self):
        r = grade_clarify(
            {"clarifying_question": "I will look into your billing issue"},
            {"category": "billing"},
        )
        # no question indicator, may have relevant keyword
        assert r["score"] <= 0.15

    def test_clarify_score_in_range(self):
        r = grade_clarify(
            {"clarifying_question": "What browser are you using?"},
            {"category": "technical"},
        )
        assert 0.0 <= r["score"] <= 0.3

    def test_draft_good_response(self):
        r = grade_draft(
            {"draft_response": "Thank you for clarifying. I understand the billing issue you are experiencing. "
             "I can see the duplicate charge on your account and will process a refund immediately. "
             "This will appear within 5-7 business days."},
            {"category": "billing"},
            customer_answer="Yes I was charged twice on March 1st and 3rd",
        )
        assert r["score"] > 0.2

    def test_draft_empty_response(self):
        r = grade_draft(
            {"draft_response": ""},
            {"category": "billing"},
            customer_answer="I was charged twice",
        )
        assert r["score"] == 0.0

    def test_draft_score_capped_at_0_4(self):
        r = grade_draft(
            {"draft_response": "Thank you for clarifying the billing issue. I am sorry for the "
             "inconvenience caused by this duplicate charge. I will refund immediately."},
            {"category": "billing"},
            customer_answer="billing charge invoice",
        )
        assert r["score"] <= 0.4

    def test_refine_improves_score(self):
        r = grade_refine(
            {"response_text": "Thank you for contacting us about your billing issue. "
             "I have reviewed your account and can confirm the duplicate charge. "
             "Please navigate to Settings and follow the steps in our refund guide. "
             "A full refund will be processed within 5-7 days. "
             "Please let me know if you need any further assistance."},
            {"category": "billing"},
            draft_response="Sorry for the issue. We will refund you.",
        )
        assert r["score"] > 0.1

    def test_refine_score_capped_at_0_3(self):
        r = grade_refine(
            {"response_text": "Great response with all details, go to settings, follow steps, let me know"},
            {"category": "billing"},
            draft_response="short",
        )
        assert r["score"] <= 0.3


# ─────────────────────────────────────────────
#  Environment integration tests — all 5 tasks
# ─────────────────────────────────────────────

class TestEnvironmentClassify:
    def test_episode_completes(self):
        env = SupportTriageEnv(task="classify")
        obs = env.reset(episode_id="test-classify-001")
        assert obs.task == "classify"
        assert obs.done is False
        result = env.step(Action(task="classify", data={"category": "billing"}))
        assert result.done is True
        assert 0.0 <= result.reward <= 1.0

    def test_deterministic(self):
        env = SupportTriageEnv(task="classify")
        obs1 = env.reset(episode_id="det-test-classify")
        obs2 = env.reset(episode_id="det-test-classify")
        assert obs1.data["ticket"]["ticket_id"] == obs2.data["ticket"]["ticket_id"]

    def test_double_step_raises(self):
        env = SupportTriageEnv(task="classify")
        env.reset()
        env.step(Action(task="classify", data={"category": "billing"}))
        with pytest.raises(RuntimeError):
            env.step(Action(task="classify", data={"category": "billing"}))

    def test_reset_clears_state(self):
        env = SupportTriageEnv(task="classify")
        env.reset()
        env.step(Action(task="classify", data={"category": "billing"}))
        env.reset()
        state = env.state()
        assert state.current_step == 0
        assert state.total_reward == 0.0


class TestEnvironmentPrioritize:
    def test_episode_completes(self):
        env = SupportTriageEnv(task="prioritize")
        obs = env.reset(episode_id="test-prioritize-001")
        assert obs.task == "prioritize"
        result = env.step(Action(
            task="prioritize",
            data={"priority": "P2", "assigned_team": "tech_support"}
        ))
        assert result.done is True
        assert 0.0 <= result.reward <= 1.0

    def test_observation_has_valid_priorities(self):
        env = SupportTriageEnv(task="prioritize")
        obs = env.reset()
        assert "valid_priorities" in obs.data
        assert "P1" in obs.data["valid_priorities"]


class TestEnvironmentEscalate:
    def test_episode_completes(self):
        env = SupportTriageEnv(task="escalate")
        obs = env.reset(episode_id="test-escalate-001")
        assert obs.task == "escalate"
        assert "conversation_history" in obs.data
        assert "agent_attempts" in obs.data
        result = env.step(Action(
            task="escalate",
            data={"should_escalate": True, "escalation_level": "L2",
                  "reason": "Customer is angry and unresolved"}
        ))
        assert result.done is True
        assert 0.0 <= result.reward <= 1.0

    def test_ground_truth_has_escalation(self):
        env = SupportTriageEnv(task="escalate")
        env.reset(episode_id="test-escalate-002")
        assert "should_escalate" in env._ground_truth
        assert "escalation_level" in env._ground_truth


class TestEnvironmentSentimentRoute:
    def test_episode_completes(self):
        env = SupportTriageEnv(task="sentiment_route")
        obs = env.reset(episode_id="test-sentiment-001")
        assert obs.task == "sentiment_route"
        assert "sentiment_score" in obs.data
        assert "keywords_detected" in obs.data
        result = env.step(Action(
            task="sentiment_route",
            data={"assigned_team": "billing_team", "urgency_flag": "high",
                  "de_escalation_note": "I understand your frustration and will help immediately"}
        ))
        assert result.done is True
        assert 0.0 <= result.reward <= 1.0

    def test_sentiment_score_in_range(self):
        env = SupportTriageEnv(task="sentiment_route")
        obs = env.reset()
        score = obs.data.get("sentiment_score", 0.0)
        assert -1.0 <= score <= 1.0


class TestEnvironmentRespond:
    def test_three_step_episode(self):
        env = SupportTriageEnv(task="respond")
        obs = env.reset(episode_id="test-respond-001")
        assert obs.task == "respond"
        assert obs.done is False

        # Step 1 — clarify
        result1 = env.step(Action(
            task="respond",
            data={"clarifying_question": "Can you tell me when this charge appeared on your account?"}
        ))
        assert result1.done is False
        assert 0.0 <= result1.reward <= 0.3

        # Step 2 — draft
        result2 = env.step(Action(
            task="respond",
            data={"draft_response": "Thank you for clarifying. I understand the billing issue "
                  "and will help resolve it immediately."}
        ))
        assert result2.done is False
        assert 0.0 <= result2.reward <= 0.4

        # Step 3 — refine
        result3 = env.step(Action(
            task="respond",
            data={"response_text": "Thank you for contacting us. I completely understand your "
                  "frustration regarding this billing issue. Following our refund process, "
                  "please navigate to Settings and I will process this immediately. "
                  "Please let me know if you need anything else."}
        ))
        assert result3.done is True
        assert 0.0 <= result3.reward <= 0.3

    def test_total_reward_sums_correctly(self):
        env = SupportTriageEnv(task="respond")
        env.reset(episode_id="test-respond-002")
        r1 = env.step(Action(task="respond", data={"clarifying_question": "What error do you see?"}))
        r2 = env.step(Action(task="respond", data={"draft_response": "Thank you for the technical details. I will fix this error now."}))
        r3 = env.step(Action(task="respond", data={"response_text": "Thank you for reaching out. I understand the technical issue. Please follow the steps in our guide. Let me know if you need help."}))
        state = env.state()
        expected = round(r1.reward + r2.reward + r3.reward, 3)
        assert state.total_reward == pytest.approx(expected, abs=0.01)

    def test_step_4_raises(self):
        env = SupportTriageEnv(task="respond")
        env.reset()
        env.step(Action(task="respond", data={"clarifying_question": "What happened?"}))
        env.step(Action(task="respond", data={"draft_response": "I will help you."}))
        env.step(Action(task="respond", data={"response_text": "Final response here."}))
        with pytest.raises(RuntimeError):
            env.step(Action(task="respond", data={"response_text": "Extra step"}))

    def test_kb_articles_present_in_step3(self):
        env = SupportTriageEnv(task="respond")
        env.reset(episode_id="test-respond-003")
        env.step(Action(task="respond", data={"clarifying_question": "When did this start?"}))
        result = env.step(Action(task="respond", data={"draft_response": "I will help you with this issue right away."}))
        assert "knowledge_base" in result.observation.data


class TestEnvironmentGeneral:
    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            SupportTriageEnv(task="nonexistent_task")

    def test_step_before_reset_raises(self):
        env = SupportTriageEnv(task="classify")
        with pytest.raises(RuntimeError):
            env.step(Action(task="classify", data={"category": "billing"}))

    def test_state_reflects_history(self):
        env = SupportTriageEnv(task="classify")
        env.reset(episode_id="state-test-001")
        env.step(Action(task="classify", data={"category": "spam"}))
        state = env.state()
        assert state.current_step == 1
        assert state.done is True
        assert len(state.history) == 1

    def test_all_tasks_complete_without_error(self):
        tasks_actions = {
            "classify":        {"category": "billing"},
            "prioritize":      {"priority": "P2", "assigned_team": "tech_support"},
            "escalate":        {"should_escalate": False, "escalation_level": "none", "reason": "Resolved"},
            "sentiment_route": {"assigned_team": "billing_team", "urgency_flag": "normal",
                                "de_escalation_note": "We are happy to help you"},
        }
        for task, action_data in tasks_actions.items():
            env = SupportTriageEnv(task=task)
            env.reset(episode_id=f"smoke-test-{task}")
            result = env.step(Action(task=task, data=action_data))
            assert result.done is True
            assert 0.0 <= result.reward <= 1.0, f"Reward out of range for task: {task}"