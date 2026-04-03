"""
FastAPI server exposing the OpenEnv HTTP interface.
Endpoints: POST /reset  POST /step  GET /state  GET /health
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional

from environment import SupportTriageEnv
from models import Action, Observation, StepResult, State

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

_env: Optional[SupportTriageEnv] = None


# ─────────────────────────────────────────────
#  Request / response schemas
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "classify"
    episode_id: Optional[str] = None
    mode: str = "static" 


class StepRequest(BaseModel):
    task: str
    data: Dict[str, Any]


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    global _env
    try:
        _env = SupportTriageEnv(task=req.task, mode=req.mode)
        obs = _env.reset(episode_id=req.episode_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    action = Action(task=req.task, data=req.data)
    try:
        result = _env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state() -> Dict[str, Any]:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return _env.state().model_dump()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "classify",
                "difficulty": "easy",
                "description": "Given a support ticket, classify it into one of 5 categories.",
                "valid_categories": ["billing", "technical", "account", "feature_request", "spam"],
            },
            {
                "id": "prioritize",
                "difficulty": "medium",
                "description": "Assign a priority (P1–P4) and route the ticket to the correct team.",
                "valid_priorities": ["P1", "P2", "P3", "P4"],
                "valid_teams": ["billing_team", "tech_support", "account_team", "product_team", "spam_filter"],
            },
            {
                "id": "respond",
                "difficulty": "hard",
                "description": "Draft a helpful, empathetic reply using the provided knowledge-base articles.",
            },
        ]
    }