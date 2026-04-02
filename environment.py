"""
SupportTriageEnv — OpenEnv-compliant environment.
Implements reset() / step() / state() for the three support triage tasks.

Tasks:
  classify   — single-step: classify ticket into category
  prioritize — single-step: assign priority + route to team
  respond    — multi-step (3 steps):
                 Step 1: ask a clarifying question   (reward 0.0-0.3)
                 Step 2: draft a response            (reward 0.0-0.4)
                 Step 3: refine using KB articles    (reward 0.0-0.3)
               Total max reward = 1.0
"""

import uuid
from typing import Any, Dict, List, Optional

from models import Observation, Action, StepResult, State
from data import get_ticket_for_episode, get_kb_articles_for_ticket, simulate_customer_reply
from graders import grade, grade_clarify, grade_draft, grade_refine


_MAX_STEPS = {
    "classify":   1,
    "prioritize": 1,
    "respond":    3,
}


class SupportTriageEnv:
    """
    Customer Support Ticket Triage environment.

    Three tasks (easy → hard):
      classify    — single-step: identify ticket category
      prioritize  — single-step: assign priority + route to correct team
      respond     — multi-step (3 steps): clarify -> draft -> refine

    Max reward per episode = 1.0 for all tasks.
    """

    VALID_TASKS = ["classify", "prioritize", "respond"]

    def __init__(self, task: str = "classify"):
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}")
        self.task = task
        self._episode_id: Optional[str] = None
        self._ground_truth: Optional[Dict[str, Any]] = None
        self._current_obs: Optional[Observation] = None
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False
        self._history: List[Dict[str, Any]] = []

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

        self._ticket, self._ground_truth = get_ticket_for_episode(
            self._episode_id, self.task
        )

        if self.task == "classify":
            from models import ClassifyObservation
            obs_data = ClassifyObservation(ticket=self._ticket).model_dump()

        elif self.task == "prioritize":
            from models import PrioritizeObservation
            obs_data = PrioritizeObservation(ticket=self._ticket).model_dump()

        else:
            from models import RespondStep1Observation
            obs_data = RespondStep1Observation(ticket=self._ticket).model_dump()
            self._kb_articles = get_kb_articles_for_ticket(
                self._ticket, self._ground_truth
            )

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

        classify / prioritize: single-step, done=True immediately.
        respond: 3 steps, done=True after step 3.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._current_obs is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1
        ticket_subject = self._ticket.subject if self._ticket else ""

        # ── classify / prioritize (single-step) ──────────────────────────
        if self.task in ("classify", "prioritize"):
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
                self._clarifying_question = action.data.get("clarifying_question", "")
                result = grade_clarify(
                    action_data=action.data,
                    ground_truth=self._ground_truth,
                    ticket_subject=ticket_subject,
                )
                reward_val = float(result["score"])

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