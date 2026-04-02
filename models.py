"""
Pydantic models for the Support Triage OpenEnv environment.
Defines all typed data structures: Observation, Action, Reward, State.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
#  Shared sub-models
# ─────────────────────────────────────────────

class Ticket(BaseModel):
    """A customer support ticket."""
    ticket_id: str
    subject: str
    body: str
    customer_plan: str = "free"         
    customer_since_days: int = 0
    previous_tickets: int = 0
    sentiment: str = "neutral"       


class KBArticle(BaseModel):
    """A knowledge-base article available to the agent."""
    article_id: str
    title: str
    content: str
    category: str


# ─────────────────────────────────────────────
#  Task 1 — classify
# ─────────────────────────────────────────────

class ClassifyObservation(BaseModel):
    ticket: Ticket
    valid_categories: List[str] = Field(
        default=["billing", "technical", "account", "feature_request", "spam"]
    )
    task: str = "classify"


class ClassifyAction(BaseModel):
    category: str  


# ─────────────────────────────────────────────
#  Task 2 — prioritize
# ─────────────────────────────────────────────

class PrioritizeObservation(BaseModel):
    ticket: Ticket
    valid_priorities: List[str] = Field(default=["P1", "P2", "P3", "P4"])
    valid_teams: List[str] = Field(
        default=["billing_team", "tech_support", "account_team", "product_team", "spam_filter"]
    )
    task: str = "prioritize"


class PrioritizeAction(BaseModel):
    priority: str 
    assigned_team: str


# ─────────────────────────────────────────────
#  Task 3 — respond
# ─────────────────────────────────────────────

class RespondObservation(BaseModel):
    ticket: Ticket
    knowledge_base: List[KBArticle]
    task: str = "respond"


class RespondAction(BaseModel):
    response_text: str  


# ─────────────────────────────────────────────
#  Generic wrappers used by the OpenEnv server
# ─────────────────────────────────────────────

class Observation(BaseModel):
    task: str
    data: Dict[str, Any]         
    step: int = 0
    done: bool = False


class Action(BaseModel):
    task: str
    data: Dict[str, Any]         


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = {}  
    feedback: str = ""


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class State(BaseModel):
    task: str
    current_step: int
    episode_id: str
    done: bool
    total_reward: float
    history: List[Dict[str, Any]] = []