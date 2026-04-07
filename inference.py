"""
inference.py — OpenEnv evaluator-compatible inference script.

Connects to the HospitalEnv via Docker API, uses the OpenAI client
for all agent decisions, and prints strict [START]/[STEP]/[END] logs.

Required environment variables:
    API_BASE_URL   — LLM API endpoint
    MODEL_NAME     — Model identifier for inference
    OPENAI_API_KEY — API key for the OpenAI client
"""

import os
import json
import requests
from openai import OpenAI

# ── ENV VARS (REQUIRED by OpenEnv evaluator) ───────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("HF_TOKEN", ""))

# ── OpenAI Client (REQUIRED — used for ALL agent decisions) ────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=OPENAI_API_KEY if OPENAI_API_KEY else "dummy-key",
)

# ── Environment Server URL (local Docker container) ────────────────────────

ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8000")

# ── Task definitions (matching openenv.yaml) ───────────────────────────────

TASKS = {
    "easy": {"num_patients": 4, "num_doctors": 3, "icu_beds": 2},
    "medium": {"num_patients": 8, "num_doctors": 2, "icu_beds": 1},
    "hard": {"num_patients": 15, "num_doctors": 2, "icu_beds": 1},
}


# ── Environment API Client ─────────────────────────────────────────────────

class EnvClient:
    """HTTP client that talks to the HospitalEnv FastAPI server."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def reset(self, config: dict = None) -> dict:
        payload = config if config else {}
        resp = requests.post(f"{self.endpoint}/reset", json=payload)
        resp.raise_for_status()
        return resp.json()["observation"]

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        resp = requests.post(f"{self.endpoint}/step", json={"action": action})
        resp.raise_for_status()
        data = resp.json()
        return data["observation"], data["reward"], data["done"], data.get("info", {})


# ── LLM-based Agent (uses OpenAI Client for ALL decisions) ─────────────────

SYSTEM_PROMPT = """You are an expert hospital triage agent. You must decide how to allocate limited medical resources (doctors and ICU beds) to patients based on their severity and wait time.

You will receive the current hospital state and a list of available actions. You must respond with EXACTLY one valid JSON action object from the available_actions list.

Decision priorities:
1. Critical patients (severity 5) should go to ICU if beds available
2. High severity patients (4-5) should be assigned doctors first
3. Consider wait time — longer-waiting patients get priority when severity is equal
4. Only use "wait" if no better option exists

IMPORTANT: Respond with ONLY a valid JSON object. No explanation, no markdown, no extra text.
Example: {"type": "assign_doctor", "patient_id": 3}"""


def build_observation_prompt(obs: dict) -> str:
    """Format the observation into a clear prompt for the LLM."""
    patients_info = []
    for p in obs.get("patients", []):
        if p["status"] == "waiting":
            patients_info.append(
                f"  Patient {p['id']}: severity={p['severity']}, wait_time={p['wait_time']}, status={p['status']}"
            )

    treating = [p for p in obs.get("patients", []) if p["status"] == "treating"]
    treated = [p for p in obs.get("patients", []) if p["status"] in ("treated", "icu")]

    prompt = f"""Current Hospital State (Step {obs['current_step']}/{obs['max_steps']}):
- Available doctors: {obs['available_doctors']}
- Available ICU beds: {obs['available_icu_beds']}
- Patients waiting: {obs['waiting_count']}
- Patients being treated: {obs['treating_count']}
- Patients done: {obs['treated_count']}/{obs['total_patients']}

Waiting patients:
{chr(10).join(patients_info) if patients_info else '  (none)'}

Available actions:
{json.dumps(obs['available_actions'], indent=2)}

Choose ONE action from the available_actions list above. Respond with ONLY the JSON object."""

    return prompt


def llm_agent(obs: dict) -> dict:
    """
    Agent that uses the OpenAI client for every decision.

    Falls back to a safe greedy action if the LLM response can't be parsed.
    """
    prompt = build_observation_prompt(obs)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=100,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        action = json.loads(raw)

        # Validate the action has required fields
        if "type" not in action:
            raise ValueError("Missing 'type' field")

        return action

    except Exception as e:
        # Fallback: deterministic greedy if LLM fails
        return _fallback_greedy(obs)


def _fallback_greedy(obs: dict) -> dict:
    """Deterministic fallback when LLM response is unparseable."""
    available = obs.get("available_actions", [])
    if not available:
        return {"type": "wait"}

    waiting = [p for p in obs.get("patients", []) if p["status"] == "waiting"]
    if not waiting:
        return {"type": "wait"}

    waiting.sort(key=lambda p: (p["severity"], p["wait_time"]), reverse=True)
    top = waiting[0]

    if top["severity"] >= 5:
        icu = [a for a in available if a["type"] == "send_to_icu" and a.get("patient_id") == top["id"]]
        if icu:
            return icu[0]

    for c in waiting:
        doc = [a for a in available if a["type"] == "assign_doctor" and a.get("patient_id") == c["id"]]
        if doc:
            return doc[0]

    return {"type": "wait"}


# ── Main Inference Loop ────────────────────────────────────────────────────

def run_inference():
    env = EnvClient(ENV_SERVER_URL)

    for task_name, task_config in TASKS.items():
        # ── [START] ─────────────────────────────────────────────────────
        print(f"[START] task={task_name} env=HospitalEnv model={MODEL_NAME}")

        obs = env.reset(task_config)
        done = False
        step_count = 0
        rewards = []

        max_reward = float(task_config["num_patients"])

        while not done:
            action = llm_agent(obs)
            step_count += 1

            obs, reward, done, info = env.step(action)
            rewards.append(reward)

            # ── [STEP] ─────────────────────────────────────────────────
            print(f"[STEP] step={step_count} action={json.dumps(action)} reward={reward} done={done}")

            # Safety: prevent infinite loops
            if step_count > 100:
                break

        # ── Score normalization ─────────────────────────────────────────
        score = sum(rewards) / max_reward if max_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = done and (obs.get("treated_count", 0) == obs.get("total_patients", 0))

        # ── [END] ──────────────────────────────────────────────────────
        print(f"[END] success={success} steps={step_count} score={score}")


if __name__ == "__main__":
    run_inference()
