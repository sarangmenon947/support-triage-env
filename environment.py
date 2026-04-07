"""
SupportTriageEnv — OpenEnv-compliant environment.
Implements reset() / step() / state() for all 5 support triage tasks.

Tasks:
  classify        — single-step: classify ticket into 1 of 5 categories
  prioritize      — single-step: assign P1-P4 priority + route to team
  escalate        — single-step: decide escalation level and reason
  sentiment_route — single-step: route by emotional urgency
  respond         — multi-step (3 steps):
                      Step 1: clarifying question  (reward 0.0-0.3)
                      Step 2: draft response       (reward 0.0-0.4)
                      Step 3: refine with KB+tools (reward 0.0-0.3)
                    Total max reward = 1.0 across all steps

Modes:
  static  — deterministic corpus of 50 tickets, reproducible by episode_id
  dynamic — LLM generates fresh ticket each episode (infinite variety)
"""

import uuid
from typing import Any, Dict, List, Optional

from models import Observation, Action, StepResult, State
from data import (get_ticket_for_episode, get_kb_articles_for_ticket,
                  simulate_customer_reply, get_escalate_ground_truth,
                  get_sentiment_route_ground_truth, _CONVERSATION_HISTORIES,
                  generate_ticket_dynamically)
from graders import grade, grade_escalate, grade_sentiment_route, grade_clarify, grade_draft, grade_refine


_MAX_STEPS = {
    "classify":        1,
    "prioritize":      1,
    "respond":         3,
    "escalate":        1,
    "sentiment_route": 1,
}


class SupportTriageEnv:
    """
    Customer Support Ticket Triage environment.

    Five tasks (easy → hard):
      classify        — single-step: identify ticket category
      prioritize      — single-step: assign priority + route to team
      escalate        — single-step: decide escalation level and reason
      sentiment_route — single-step: route by emotional urgency
      respond         — multi-step (3 steps): clarify -> draft -> refine

    Max reward per episode = 1.0 for all tasks.
    Supports static (deterministic) and dynamic (LLM-generated) ticket modes.
    """

    VALID_TASKS = ["classify", "prioritize", "respond", "escalate", "sentiment_route"]

    def __init__(self, task: str = "classify", mode: str = "static"):
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}")
        self.task = task
        self.mode = mode  # "static" or "dynamic"
        self._episode_id: Optional[str] = None
        self._ground_truth: Optional[Dict[str, Any]] = None
        self._current_obs: Optional[Observation] = None
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False
        self._history: List[Dict[str, Any]] = []

        # respond multi-step state
        self._ticket = None
        self._kb_articles = None
        self._clarifying_question: str = ""
        self._customer_answer: str = ""
        self._draft_response: str = ""

    def reset(self, episode_id: Optional[str] = None) -> Observation:
        """Start a new episode. Returns the first observation."""
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._history = []
        self._clarifying_question = ""
        self._customer_answer = ""
        self._draft_response = ""

        if self.mode == "dynamic":
            self._ticket, self._ground_truth = generate_ticket_dynamically(
                self._episode_id
            )
        else:
            self._ticket, self._ground_truth = get_ticket_for_episode(
                self._episode_id, self.task
            )

        if self.task == "classify":
            from models import ClassifyObservation
            obs_data = ClassifyObservation(ticket=self._ticket).model_dump()

        elif self.task == "prioritize":
            from models import PrioritizeObservation
            obs_data = PrioritizeObservation(ticket=self._ticket).model_dump()

        elif self.task == "respond":
            from models import RespondStep1Observation
            obs_data = RespondStep1Observation(ticket=self._ticket).model_dump()
            self._kb_articles = get_kb_articles_for_ticket(
                self._ticket, self._ground_truth
            )

        elif self.task == "escalate":
            from models import EscalateObservation
            import random
            rng = random.Random(hash(self._episode_id))
            history = rng.choice(_CONVERSATION_HISTORIES)
            attempts = rng.randint(0, 3)
            obs_data = EscalateObservation(
                ticket=self._ticket,
                conversation_history=history,
                agent_attempts=attempts,
            ).model_dump()
            # extend ground truth with escalation answer
            esc_gt = get_escalate_ground_truth(self._ground_truth)
            self._ground_truth.update(esc_gt)

        else:  # sentiment_route
            from models import SentimentRouteObservation
            sr_gt = get_sentiment_route_ground_truth(
                self._ground_truth, self._ticket.body
            )
            self._ground_truth.update(sr_gt)
            obs_data = SentimentRouteObservation(
                ticket=self._ticket,
                sentiment_score=sr_gt["sentiment_score"],
                keywords_detected=sr_gt["keywords_detected"],
            ).model_dump()

        self._current_obs = Observation(
            task=self.task,
            data=obs_data,
            step=0,
            done=False,
        )
        return self._current_obs

    def step(self, action: Action) -> StepResult:
        """
        Process one action. Returns (observation, reward, done, info).

        Single-step tasks (done=True after step 1):
          classify, prioritize, escalate, sentiment_route

        Multi-step tasks:
          respond — 3 steps, done=True after step 3
                    cumulative reward across all steps, max = 1.0
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._current_obs is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1
        ticket_subject = self._ticket.subject if self._ticket else ""

        # ── classify / prioritize (single-step) ──────────────────────────
        if self.task in ("classify", "prioritize", "escalate", "sentiment_route"):
            if self.task == "escalate":
                result = grade_escalate(action.data, self._ground_truth)
            elif self.task == "sentiment_route":
                result = grade_sentiment_route(action.data, self._ground_truth)
            else:
                result = grade(
                    task=self.task,
                    action_data=action.data,
                    ground_truth=self._ground_truth,
                    ticket_subject=ticket_subject,
                )
            reward_val = float(result["score"])
            self._total_reward = round(self._total_reward + reward_val, 3)
            self._done = True

            next_obs = Observation(
                task=self.task,
                data=self._current_obs.data,
                step=self._step_count,
                done=True,
            )

        # ── respond (multi-step) ─────────────────────────────────────────
        else:
            if self._step_count == 1:
                # Step 1 — clarifying question
                self._clarifying_question = action.data.get("clarifying_question", "")
                result = grade_clarify(
                    action_data=action.data,
                    ground_truth=self._ground_truth,
                    ticket_subject=ticket_subject,
                )
                reward_val = float(result["score"])

                # Simulate customer reply
                self._customer_answer = simulate_customer_reply(
                    self._ticket, self._clarifying_question, self._ground_truth
                )

                from models import RespondStep2Observation
                obs_data = RespondStep2Observation(
                    ticket=self._ticket,
                    clarifying_question=self._clarifying_question,
                    customer_answer=self._customer_answer,
                ).model_dump()
                done = False

            elif self._step_count == 2:
                # Step 2 — draft response
                self._draft_response = action.data.get("draft_response", "")
                result = grade_draft(
                    action_data=action.data,
                    ground_truth=self._ground_truth,
                    ticket_subject=ticket_subject,
                    customer_answer=self._customer_answer,
                )
                reward_val = float(result["score"])

                from models import RespondStep3Observation
                obs_data = RespondStep3Observation(
                    ticket=self._ticket,
                    draft_response=self._draft_response,
                    knowledge_base=self._kb_articles,
                ).model_dump()
                done = False

            else:
                # Step 3 — refine using KB
                result = grade_refine(
                    action_data=action.data,
                    ground_truth=self._ground_truth,
                    ticket_subject=ticket_subject,
                    draft_response=self._draft_response,
                )
                reward_val = float(result["score"])
                obs_data = self._current_obs.data
                done = True
                self._done = True

            self._total_reward = round(self._total_reward + reward_val, 3)

            next_obs = Observation(
                task=self.task,
                data=obs_data,
                step=self._step_count,
                done=done,
            )

        self._current_obs = next_obs

        self._history.append({
            "step": self._step_count,
            "action": action.data,
            "reward": reward_val,
            "feedback": result.get("feedback", ""),
            "breakdown": result.get("breakdown", {}),
        })

        return StepResult(
            observation=next_obs,
            reward=reward_val,
            done=self._done,
            info={
                "feedback": result.get("feedback", ""),
                "breakdown": result.get("breakdown", {}),
                "ground_truth": self._ground_truth,
                "episode_id": self._episode_id,
                "max_steps": _MAX_STEPS[self.task],
            },
        )

    def state(self) -> State:
        """Return the current episode state."""
        return State(
            task=self.task,
            current_step=self._step_count,
            episode_id=self._episode_id or "",
            done=self._done,
            total_reward=self._total_reward,
            history=self._history,
        )