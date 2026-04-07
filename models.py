"""
models.py — Typed Pydantic models for the HospitalEnv OpenEnv environment.

Domain models:
    Action, Patient, Observation — typed schemas for the env interface.

API models:
    ResetRequest, ActionRequest, ResetResponse, StepResponse, StateResponse
"""

from pydantic import BaseModel, Field
from typing import Any, Optional, Literal


# ═══════════════════════════════════════════════════════════════════════════
#  DOMAIN MODELS (OpenEnv spec — typed Observation, Action, Reward)
# ═══════════════════════════════════════════════════════════════════════════

class Action(BaseModel):
    """Typed action model for the HospitalEnv."""
    type: Literal["assign_doctor", "send_to_icu", "wait"]
    patient_id: Optional[int] = None


class Patient(BaseModel):
    """Typed patient model within the observation."""
    id: int
    severity: int = Field(ge=1, le=5)
    wait_time: int = Field(ge=0)
    status: Literal["waiting", "treating", "treated", "icu"]
    treatment_timer: int = 0


class Observation(BaseModel):
    """Typed observation model returned by the environment."""
    patients: list[Patient]
    available_doctors: int
    available_icu_beds: int
    current_step: int
    max_steps: int
    treated_count: int
    waiting_count: int
    treating_count: int
    total_patients: int
    done: bool
    available_actions: list[dict[str, Any]]
    action_space: dict[str, Any]


class Reward(BaseModel):
    """Typed reward model — always clamped to [0.0, 1.0]."""
    value: float = Field(ge=0.0, le=1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  API REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    """Body for POST /reset — all fields optional for flexibility."""
    seed: Optional[int] = None
    num_patients: Optional[int] = None
    num_doctors: Optional[int] = None
    icu_beds: Optional[int] = None


class ActionRequest(BaseModel):
    """Body for POST /step — wraps the agent's action."""
    action: Action


class ResetResponse(BaseModel):
    """Response from POST /reset."""
    observation: Observation
    done: bool = False


class StepResponse(BaseModel):
    """Response from POST /step — includes info dict per OpenEnv spec."""
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = {}


class StateResponse(BaseModel):
    """Response from GET|POST /state."""
    observation: Observation
