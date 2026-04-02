"""
SupportTriageEnv — OpenEnv-compliant environment.
Implements reset() / step() / state() for the three support triage tasks.
"""

import uuid
from typing import Any, Dict, Optional

from models import Observation, Action, StepResult, State
from data import get_ticket_for_episode, get_kb_articles_for_ticket
from graders import grade


class SupportTriageEnv:
    """
    Customer Support Ticket Triage environment.

    Three tasks (easy → hard):
      classify    — identify ticket category
      prioritize  — assign priority + route to correct team
      respond     — draft a reply using the knowledge base

    Each episode is single-step (step once, episode ends).
    Max reward per episode = 1.0.
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
        self._history: list = []

    # ─────────────────────────────────────────
    #  reset()
    # ─────────────────────────────────────────

    def reset(self, episode_id: Optional[str] = None) -> Observation:
        """Start a new episode. Returns the first (and only) observation."""
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._history = []

        ticket, self._ground_truth = get_ticket_for_episode(self._episode_id, self.task)

        if self.task == "classify":
            from models import ClassifyObservation
            obs_data = ClassifyObservation(ticket=ticket).model_dump()

        elif self.task == "prioritize":
            from models import PrioritizeObservation
            obs_data = PrioritizeObservation(ticket=ticket).model_dump()

        else: 
            from models import RespondObservation
            articles = get_kb_articles_for_ticket(ticket, self._ground_truth)
            obs_data = RespondObservation(ticket=ticket, knowledge_base=articles).model_dump()

        self._current_obs = Observation(
            task=self.task,
            data=obs_data,
            step=0,
            done=False,
        )
        return self._current_obs

    # ─────────────────────────────────────────
    #  step()
    # ─────────────────────────────────────────

    def step(self, action: Action) -> StepResult:
        """
        Process one action and return (observation, reward, done, info).
        Each episode is single-step: done=True after the first step.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._current_obs is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1

        ticket_subject = self._current_obs.data.get("ticket", {}).get("subject", "")
        result = grade(
            task=self.task,
            action_data=action.data,
            ground_truth=self._ground_truth,
            ticket_subject=ticket_subject,
        )
        reward_val = float(result["score"])
        self._total_reward += reward_val
        self._done = True  

        self._history.append({
            "step": self._step_count,
            "action": action.data,
            "reward": reward_val,
            "feedback": result.get("feedback", ""),
            "breakdown": result.get("breakdown", {}),
        })

        next_obs = Observation(
            task=self.task,
            data=self._current_obs.data,
            step=self._step_count,
            done=True,
        )
        self._current_obs = next_obs

        return StepResult(
            observation=next_obs,
            reward=reward_val,
            done=True,
            info={
                "feedback": result.get("feedback", ""),
                "breakdown": result.get("breakdown", {}),
                "ground_truth": self._ground_truth,
                "episode_id": self._episode_id,
            },
        )

    # ─────────────────────────────────────────
    #  state()
    # ─────────────────────────────────────────

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