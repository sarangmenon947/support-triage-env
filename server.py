"""
FastAPI server exposing the OpenEnv HTTP interface.
Endpoints: POST /reset  POST /step  GET /state  GET /health
"""

import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from environment import SupportTriageEnv
from models import Action, Observation, StepResult, State
from tools import TOOL_DEFINITIONS, execute_tool

app = FastAPI(
    title="Support Triage OpenEnv",
    description="Customer Support Ticket Triage — OpenEnv-compliant environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session-based environment store — supports concurrent agents
_sessions: Dict[str, SupportTriageEnv] = {}
_session_tickets: Dict[str, Any] = {}  # stores ticket info per session for tool use
DEFAULT_SESSION = "default"


# ─────────────────────────────────────────────
#  Request / response schemas
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "classify"
    episode_id: Optional[str] = None
    mode: str = "static"   # "static" (deterministic) or "dynamic" (LLM-generated ticket)
    session_id: str = DEFAULT_SESSION  # supports concurrent agents


class StepRequest(BaseModel):
    task: str
    data: Dict[str, Any]
    session_id: str = DEFAULT_SESSION


class ToolCallRequest(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any]
    session_id: str = DEFAULT_SESSION


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    try:
        env = SupportTriageEnv(task=req.task, mode=req.mode)
        obs = env.reset(episode_id=req.episode_id)
        _sessions[req.session_id] = env
        # Store ticket info for tool use
        ticket = obs.data.get("ticket", {})
        _session_tickets[req.session_id] = {
            "ticket_id": ticket.get("ticket_id", ""),
            "category": env._ground_truth.get("category", "technical") if env._ground_truth else "technical",
            "customer_plan": ticket.get("customer_plan", "free"),
            "customer_since_days": ticket.get("customer_since_days", 0),
            "previous_tickets": ticket.get("previous_tickets", 0),
        }
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    action = Action(task=req.task, data=req.data)
    try:
        result = env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(session_id: str = DEFAULT_SESSION) -> Dict[str, Any]:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return env.state().model_dump()


@app.post("/tool")
def call_tool(req: ToolCallRequest) -> Dict[str, Any]:
    """Agent calls a tool to look up information before responding."""
    ticket_info = _session_tickets.get(req.session_id, {})
    if not ticket_info:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    result = execute_tool(
        tool_name=req.tool_name,
        tool_args=req.tool_args,
        ticket_id=ticket_info.get("ticket_id", ""),
        category=ticket_info.get("category", "technical"),
        customer_plan=ticket_info.get("customer_plan", "free"),
        customer_since_days=ticket_info.get("customer_since_days", 0),
        previous_tickets=ticket_info.get("previous_tickets", 0),
    )
    return result


@app.get("/tools")
def list_tools() -> Dict[str, Any]:
    """List all available tools agents can call."""
    return {"tools": TOOL_DEFINITIONS}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "classify",
                "difficulty": "easy",
                "max_steps": 1,
                "description": "Classify a support ticket into one of 5 categories.",
                "valid_categories": ["billing", "technical", "account", "feature_request", "spam"],
            },
            {
                "id": "prioritize",
                "difficulty": "medium",
                "max_steps": 1,
                "description": "Assign priority (P1-P4) and route ticket to correct team.",
                "valid_priorities": ["P1", "P2", "P3", "P4"],
                "valid_teams": ["billing_team", "tech_support", "account_team", "product_team", "spam_filter"],
            },
            {
                "id": "escalate",
                "difficulty": "medium",
                "max_steps": 1,
                "description": "Decide if ticket needs escalation, to what level, and why.",
                "valid_levels": ["none", "L1", "L2", "L3", "manager"],
            },
            {
                "id": "sentiment_route",
                "difficulty": "medium",
                "max_steps": 1,
                "description": "Route emotionally charged ticket with correct urgency flag.",
                "valid_teams": ["billing_team", "tech_support", "account_team", "product_team", "spam_filter", "vip_support"],
                "valid_urgency": ["low", "normal", "high", "critical"],
            },
            {
                "id": "respond",
                "difficulty": "hard",
                "max_steps": 3,
                "description": "Multi-step: clarify -> draft -> refine using KB articles and external tools.",
                "available_tools": ["search_kb", "lookup_customer", "check_order_status", "get_similar_tickets"],
            },
        ]
    }