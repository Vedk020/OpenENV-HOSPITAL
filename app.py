"""
app.py — FastAPI server for the HospitalEnv OpenEnv environment.

Endpoints:
    POST /reset  — Reinitialize environment, optionally with task config.
    POST /step   — Advance one time-step with an action.
    GET  /state  — Return current observation without advancing time.

This is the main entrypoint for Docker / HF Spaces deployment.
Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Body
from typing import Any, Optional
from env import HospitalEnv
from models import ResetRequest, ActionRequest, ResetResponse, StepResponse, StateResponse

app = FastAPI(
    title="HospitalEnv",
    description="OpenEnv Hospital Triage Simulation Environment",
    version="1.0.0",
)

# Global instance — reused across resets (state reinitialised, not destroyed)
env_instance = HospitalEnv()


@app.post("/reset", response_model=ResetResponse)
def reset_env(req: Optional[ResetRequest] = Body(default=None)):
    """
    Reinitialize the environment for a new episode.

    Accepts optional task configuration (num_patients, num_doctors, icu_beds, seed).
    If no body is sent, resets with default configuration.

    Returns: {"observation": {...}, "done": false}
    """
    global env_instance

    if req is not None:
        if req.num_patients is not None:
            env_instance.num_patients = req.num_patients
        if req.num_doctors is not None:
            env_instance.num_doctors = req.num_doctors
        if req.icu_beds is not None:
            env_instance.icu_beds = req.icu_beds

        obs = env_instance.reset(seed=req.seed)
    else:
        obs = env_instance.reset()

    return {
        "observation": obs,
        "done": False,
    }


@app.post("/step", response_model=StepResponse)
def step_env(req: ActionRequest):
    """
    Advance the simulation by one time-step.

    Accepts an action dict:
        {"action": {"type": "assign_doctor", "patient_id": 1}}

    Returns: {"observation": {...}, "reward": 0.7, "done": false, "info": {...}}
    """
    obs, reward, done, info = env_instance.step(req.action)
    return {
        "observation": obs,
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    }


@app.get("/state", response_model=StateResponse)
@app.post("/state", response_model=StateResponse)
def get_state():
    """
    Return the current observation snapshot without advancing time.

    Returns: {"observation": {...}}
    """
    return {
        "observation": env_instance.state(),
    }
