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
#  Task 3 — respond  (multi-step, 3 steps)
# ─────────────────────────────────────────────
#
#  Step 1 — clarify:  Agent asks one clarifying question
#  Step 2 — draft:    Agent drafts a response (customer reply provided)
#  Step 3 — refine:   Agent refines draft using KB articles
#
# Each step has its own observation and action type.

class RespondStep1Observation(BaseModel):
    """Step 1: Agent sees ticket, must ask a clarifying question."""
    ticket: Ticket
    instruction: str = "Ask one clarifying question to better understand the customer's issue."
    task: str = "respond"
    respond_step: int = 1


class RespondStep1Action(BaseModel):
    clarifying_question: str 


class RespondStep2Observation(BaseModel):
    """Step 2: Agent sees ticket + customer's answer, must draft a response."""
    ticket: Ticket
    clarifying_question: str
    customer_answer: str  
    instruction: str = "Draft a helpful response to the customer based on their clarification."
    task: str = "respond"
    respond_step: int = 2


class RespondStep2Action(BaseModel):
    draft_response: str 


class RespondStep3Observation(BaseModel):
    """Step 3: Agent sees ticket + draft + KB articles, must produce final response."""
    ticket: Ticket
    draft_response: str      
    knowledge_base: List[KBArticle]
    instruction: str = "Refine your draft using the knowledge-base articles to produce a final, accurate response."
    task: str = "respond"
    respond_step: int = 3


class RespondStep3Action(BaseModel):
    response_text: str  # The final polished reply



# ─────────────────────────────────────────────
#  Task 4 — escalate
# ─────────────────────────────────────────────

class EscalateObservation(BaseModel):
    ticket: Ticket
    conversation_history: List[str] = []   
    agent_attempts: int = 0              
    valid_escalation_levels: List[str] = Field(
        default=["none", "L1", "L2", "L3", "manager"]
    )
    task: str = "escalate"


class EscalateAction(BaseModel):
    should_escalate: bool
    escalation_level: str  
    reason: str           


# ─────────────────────────────────────────────
#  Task 5 — sentiment_route
# ─────────────────────────────────────────────

class SentimentRouteObservation(BaseModel):
    ticket: Ticket
    sentiment_score: float = 0.0         
    keywords_detected: List[str] = []    
    valid_teams: List[str] = Field(
        default=["billing_team", "tech_support", "account_team",
                 "product_team", "spam_filter", "vip_support"]
    )
    valid_urgency_flags: List[str] = Field(
        default=["low", "normal", "high", "critical"]
    )
    task: str = "sentiment_route"


class SentimentRouteAction(BaseModel):
    assigned_team: str
    urgency_flag: str         
    de_escalation_note: str    

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