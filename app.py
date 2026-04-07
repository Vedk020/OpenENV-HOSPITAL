from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Optional
from env import HospitalEnv

app = FastAPI()

# Global instance initialized natively
env_instance = HospitalEnv()

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    num_patients: Optional[int] = None
    num_doctors: Optional[int] = None
    icu_beds: Optional[int] = None

class ActionRequest(BaseModel):
    action: dict[str, Any]

@app.post("/reset")
def reset_env(req: ResetRequest = None):
    global env_instance
    
    # If the evaluator specifies task configs during reset
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
        "done": False
    }

@app.post("/step")
def step_env(req: ActionRequest):
    obs, reward, done = env_instance.step(req.action)
    return {
        "observation": obs,
        "reward": float(reward),
        "done": bool(done)
    }

@app.get("/state")
@app.post("/state")
def get_state():
    return {
        "observation": env_instance.state()
    }
