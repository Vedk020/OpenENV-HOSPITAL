import json
import asyncio
import requests
import time

class APIRouterEnv:
    """Wrapper that acts like the client for OpenEnv evaluator"""
    
    def __init__(self, endpoint="http://localhost:8000"):
        self.endpoint = endpoint

    @classmethod
    async def from_docker_image(cls, image_name: str):
        # The evaluator builds Docker, runs container, and exposes API.
        # This wrapper simulates the established OpenEnv connection approach.
        return cls()

    def reset(self, config=None):
        payload = config if config else {}
        # Allows for injection of the task variables from YAML
        resp = requests.post(f"{self.endpoint}/reset", json=payload)
        resp.raise_for_status()
        return resp.json()["observation"]

    def step(self, action):
        resp = requests.post(f"{self.endpoint}/step", json={"action": action})
        resp.raise_for_status()
        data = resp.json()
        return data["observation"], data["reward"], data["done"]

def greedy_agent(obs: dict) -> dict:
    """Agent identical logic to earlier standalone testing."""
    available = obs.get("available_actions", [])
    if not available:
        return {"type": "wait"}

    waiting = [p for p in obs.get("patients", []) if p["status"] == "waiting"]
    if not waiting:
        return {"type": "wait"}

    # Sort descending by severity, then descending by wait time
    waiting.sort(key=lambda p: (p["severity"], p["wait_time"]), reverse=True)
    top = waiting[0]

    # Critical -> ICU
    if top["severity"] >= 5:
        icu_actions = [a for a in available if a["type"] == "send_to_icu" and a.get("patient_id") == top["id"]]
        if icu_actions:
            return icu_actions[0]

    # Assign Doctor
    for candidate in waiting:
        doc_actions = [a for a in available if a["type"] == "assign_doctor" and a.get("patient_id") == candidate["id"]]
        if doc_actions:
            return doc_actions[0]

    return {"type": "wait"}


async def run_inference():
    # 1. OpenEnv Docker connection 
    IMAGE_NAME = "openenv/hospital:latest"
    env = await APIRouterEnv.from_docker_image(IMAGE_NAME)
    
    task_name = "medium"
    model_name = "GreedyAgent-Triage-v1"
    
    # 2. Strict log: [START]
    print(f"[START] task={task_name} env=HospitalEnv model={model_name}")
    
    obs = env.reset({"num_patients": 8, "num_doctors": 2, "icu_beds": 1})
    done = False
    step_count = 0
    rewards = []
    
    # Maximum possible score is derived if every timestep yields exactly 1.0 (clamped). 
    # Or specifically, if all 8 patients generate optimal ICU/Doctor allocation.
    # The true max bound would be (total_patients * optimal_treatment_reward).
    # Assuming evaluator handles maximum scaling differently, we can estimate for the formula.
    MAX_TOTAL_REWARD = 8.0 
    
    # Wait for container startup
    time.sleep(1)

    while not done:
        action = greedy_agent(obs)
        step_count += 1
        
        obs, reward, done = env.step(action)
        rewards.append(reward)
        
        # 3. Strict log: [STEP]
        print(f"[STEP] step={step_count} action={json.dumps(action)} reward={reward} done={str(done).lower()}")
        
        # Safety catch
        if step_count > 100:
            break

    # 4. Strict log calculation & formatting
    raw_score = sum(rewards) / MAX_TOTAL_REWARD
    score = min(max(raw_score, 0.0), 1.0)
    success = done and (obs.get("treated_count", 0) == obs.get("total_patients", 0))

    # 5. Strict log: [END]
    print(f"[END] success={str(success).lower()} steps={step_count} score={round(score, 4)}")

if __name__ == "__main__":
    asyncio.run(run_inference())
