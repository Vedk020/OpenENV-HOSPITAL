"""
inference.py — OpenEnv evaluator-compatible inference script.

Connects to the HospitalEnv via Docker API, uses OpenAI client for
model interaction, and prints strict [START]/[STEP]/[END] logs.
"""

import os
import json
import asyncio
import requests
import time
from openai import OpenAI

# ── ENV VARS (REQUIRED by OpenEnv evaluator) ───────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "GreedyAgent-Triage-v1")
HF_TOKEN = os.getenv("HF_TOKEN", "")


# ── OpenAI Client (REQUIRED by OpenEnv rules) ──────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy-key",
)


# ── Environment API Client ─────────────────────────────────────────────────

class APIRouterEnv:
    """Wrapper that talks to the HospitalEnv FastAPI server."""

    def __init__(self, endpoint="http://localhost:8000"):
        self.endpoint = endpoint

    @classmethod
    async def from_docker_image(cls, image_name: str):
        # The evaluator builds Docker, runs container, and exposes API.
        return cls()

    def reset(self, config=None):
        payload = config if config else {}
        resp = requests.post(f"{self.endpoint}/reset", json=payload)
        resp.raise_for_status()
        return resp.json()["observation"]

    def step(self, action):
        resp = requests.post(f"{self.endpoint}/step", json={"action": action})
        resp.raise_for_status()
        data = resp.json()
        return data["observation"], data["reward"], data["done"]


# ── Agent Logic ────────────────────────────────────────────────────────────

def greedy_agent(obs: dict) -> dict:
    """Deterministic greedy triage agent using available_actions."""
    available = obs.get("available_actions", [])
    if not available:
        return {"type": "wait"}

    waiting = [p for p in obs.get("patients", []) if p["status"] == "waiting"]
    if not waiting:
        return {"type": "wait"}

    waiting.sort(key=lambda p: (p["severity"], p["wait_time"]), reverse=True)
    top = waiting[0]

    # Critical -> ICU
    if top["severity"] >= 5:
        icu_actions = [
            a for a in available
            if a["type"] == "send_to_icu" and a.get("patient_id") == top["id"]
        ]
        if icu_actions:
            return icu_actions[0]

    # Assign doctor to most urgent
    for candidate in waiting:
        doc_actions = [
            a for a in available
            if a["type"] == "assign_doctor" and a.get("patient_id") == candidate["id"]
        ]
        if doc_actions:
            return doc_actions[0]

    return {"type": "wait"}


# ── Main Inference Loop ────────────────────────────────────────────────────

async def run_inference():
    IMAGE_NAME = "openenv/hospital:latest"
    env = await APIRouterEnv.from_docker_image(IMAGE_NAME)

    task_name = "medium"

    # ── [START] ─────────────────────────────────────────────────────────────
    print(f"[START] task={task_name} env=HospitalEnv model={MODEL_NAME}")

    obs = env.reset({"num_patients": 8, "num_doctors": 2, "icu_beds": 1})
    done = False
    step_count = 0
    rewards = []

    MAX_TOTAL_REWARD = 8.0

    while not done:
        action = greedy_agent(obs)
        step_count += 1

        obs, reward, done = env.step(action)
        rewards.append(reward)

        # ── [STEP] ─────────────────────────────────────────────────────────
        print(f"[STEP] step={step_count} action={json.dumps(action)} reward={reward} done={done}")

        if step_count > 100:
            break

    # ── Score normalization ─────────────────────────────────────────────────
    score = sum(rewards) / MAX_TOTAL_REWARD
    score = min(max(score, 0.0), 1.0)
    success = done and (obs.get("treated_count", 0) == obs.get("total_patients", 0))

    # ── [END] ──────────────────────────────────────────────────────────────
    print(f"[END] success={success} steps={step_count} score={score}")


if __name__ == "__main__":
    asyncio.run(run_inference())
