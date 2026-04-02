"""
Tests for graders and core environment logic.
Run with: python -m pytest tests/test_env.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from graders import grade_classify, grade_prioritize, grade_respond
from environment import SupportTriageEnv
from models import Action


# ─────────────────────────────────────────────
#  Grader tests — classify
# ─────────────────────────────────────────────

class TestClassifyGrader:
    def test_exact_match(self):
        r = grade_classify({"category": "billing"}, {"category": "billing"})
        assert r["score"] == 1.0

    def test_adjacent_category(self):
        r = grade_classify({"category": "account"}, {"category": "billing"})
        assert r["score"] == 0.4

    def test_wrong_category(self):
        r = grade_classify({"category": "spam"}, {"category": "billing"})
        assert r["score"] == 0.0

    def test_case_insensitive(self):
        r = grade_classify({"category": "BILLING"}, {"category": "billing"})
        assert r["score"] == 1.0

    def test_missing_field(self):
        r = grade_classify({}, {"category": "billing"})
        assert r["score"] == 0.0

    def test_all_categories_exact(self):
        for cat in ["billing", "technical", "account", "feature_request", "spam"]:
            r = grade_classify({"category": cat}, {"category": cat})
            assert r["score"] == 1.0, f"Failed for category: {cat}"


# ─────────────────────────────────────────────
#  Grader tests — prioritize
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
        assert r["score"] == pytest.approx(0.7, abs=0.01)

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
        assert r["score"] == pytest.approx(0.6, abs=0.01)

    def test_score_in_range(self):
        r = grade_prioritize(
            {"priority": "P3", "assigned_team": "tech_support"},
            {"priority": "P2", "assigned_team": "tech_support"},
        )
        assert 0.0 <= r["score"] <= 1.0


# ─────────────────────────────────────────────
#  Grader tests — respond
# ─────────────────────────────────────────────

class TestRespondGrader:
    GOOD_RESPONSE = (
        "Thank you for reaching out. I'm sorry to hear about the double charge on your account. "
        "I understand how frustrating this can be. I have reviewed your billing history and can "
        "confirm the duplicate payment of $49. I've submitted a refund request, which will appear "
        "within 5–7 business days. If you have any further questions, please don't hesitate to reach out."
    )

    def test_perfect_response(self):
        r = grade_respond(
            {"response_text": self.GOOD_RESPONSE},
            {"category": "billing"},
        )
        assert r["score"] >= 0.75

    def test_empty_response(self):
        r = grade_respond({"response_text": ""}, {"category": "billing"})
        assert r["score"] == 0.0

    def test_score_between_0_and_1(self):
        r = grade_respond(
            {"response_text": "Sorry for the issue."},
            {"category": "technical"},
        )
        assert 0.0 <= r["score"] <= 1.0

    def test_breakdown_keys_present(self):
        r = grade_respond(
            {"response_text": self.GOOD_RESPONSE},
            {"category": "billing"},
        )
        assert "issue_acknowledged" in r["breakdown"]
        assert "solution_provided" in r["breakdown"]
        assert "empathy_tone" in r["breakdown"]
        assert "proper_closing" in r["breakdown"]


# ─────────────────────────────────────────────
#  Environment integration tests
# ─────────────────────────────────────────────

class TestEnvironment:
    def test_classify_episode(self):
        env = SupportTriageEnv(task="classify")
        obs = env.reset(episode_id="test-episode-001")
        assert obs.task == "classify"
        assert obs.done is False

        action = Action(task="classify", data={"category": "billing"})
        result = env.step(action)
        assert result.done is True
        assert 0.0 <= result.reward <= 1.0

    def test_prioritize_episode(self):
        env = SupportTriageEnv(task="prioritize")
        obs = env.reset(episode_id="test-episode-002")
        assert obs.task == "prioritize"

        action = Action(task="prioritize", data={"priority": "P2", "assigned_team": "tech_support"})
        result = env.step(action)
        assert result.done is True
        assert 0.0 <= result.reward <= 1.0

    def test_respond_episode(self):
        env = SupportTriageEnv(task="respond")
        obs = env.reset(episode_id="test-episode-003")
        assert "knowledge_base" in obs.data

        action = Action(task="respond", data={"response_text": "Thank you for contacting support. I understand your frustration. We will help you resolve this billing issue right away. Please don't hesitate to reach out if you need anything further."})
        result = env.step(action)
        assert result.done is True
        assert 0.0 <= result.reward <= 1.0

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

    def test_deterministic_reset(self):
        """Same episode_id must always produce the same ticket."""
        env = SupportTriageEnv(task="classify")
        obs1 = env.reset(episode_id="determinism-test-xyz")
        env.reset(episode_id="determinism-test-xyz")
        obs2 = env.reset(episode_id="determinism-test-xyz")
        assert obs1.data["ticket"]["ticket_id"] == obs2.data["ticket"]["ticket_id"]

    def test_state_reflects_step(self):
        env = SupportTriageEnv(task="classify")
        env.reset(episode_id="state-test-001")
        state_before = env.state()
        assert state_before.current_step == 0

        env.step(Action(task="classify", data={"category": "spam"}))
        state_after = env.state()
        assert state_after.current_step == 1
        assert state_after.done is True

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            SupportTriageEnv(task="not_a_task")