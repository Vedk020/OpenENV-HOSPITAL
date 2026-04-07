"""
Grader — Evaluates agent performance over full episodes.

Submission-ready: validates reward normalization, observation stability,
action format, stalemate handling, and reproducibility.

Usage:
    from env import HospitalEnv
    from env.grader import grade_episode, validate_env

    env = HospitalEnv(seed=42)
    total_reward, steps, report = grade_episode(env, agent_fn)
    issues = validate_env(env)
"""

import copy
from typing import Any, Callable
from .hospital_env import HospitalEnv


# ── Required observation keys (FIX #6 — stability guarantee) ────────────────

_REQUIRED_OBS_KEYS = frozenset({
    "patients",
    "available_doctors",
    "available_icu_beds",
    "current_step",
    "max_steps",
    "treated_count",
    "waiting_count",
    "treating_count",
    "total_patients",
    "done",
    "available_actions",
    "action_space",
})

_REQUIRED_PATIENT_KEYS = frozenset({
    "id",
    "severity",
    "wait_time",
    "status",
})


def grade_episode(
    env: HospitalEnv,
    agent_fn: Callable[[dict[str, Any]], dict[str, Any]],
    verbose: bool = False,
) -> tuple[float, int, dict[str, Any]]:
    """
    Run one full episode and return a performance report.

    Parameters
    ----------
    env       : a HospitalEnv instance
    agent_fn  : callable(observation) -> action dict
    verbose   : if True, print per-step details

    Returns
    -------
    total_reward : float  — sum of per-step rewards
    steps        : int    — number of steps taken
    report       : dict   — detailed breakdown
    """
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    rewards: list[float] = []
    actions_taken: list[dict[str, Any]] = []
    reward_violations: int = 0
    obs_key_violations: int = 0
    patient_key_violations: int = 0

    while True:
        # ── Validate observation structure ──────────────────────────────
        missing_obs = _REQUIRED_OBS_KEYS - set(obs.keys())
        if missing_obs:
            obs_key_violations += 1
            if verbose:
                print(f"  ⚠ Step {steps}: observation missing keys: {missing_obs}")

        # ── Validate patient sub-dicts ──────────────────────────────────
        for p in obs.get("patients", []):
            missing_p = _REQUIRED_PATIENT_KEYS - set(p.keys())
            if missing_p:
                patient_key_violations += 1
                if verbose:
                    print(f"  ⚠ Step {steps}: patient {p.get('id','?')} "
                          f"missing keys: {missing_p}")

        # ── Agent decides ───────────────────────────────────────────────
        action = agent_fn(obs)
        actions_taken.append(copy.deepcopy(action))

        obs, reward, done, info = env.step(action)

        # ── Validate reward normalization ───────────────────────────────
        if not (0.0 <= reward <= 1.0):
            reward_violations += 1
            if verbose:
                print(f"  ❌ Step {steps}: reward {reward} OUT OF [0, 1]!")

        total_reward += reward
        rewards.append(reward)
        steps += 1

        if verbose:
            action_desc = action.get("type", "?")
            if "patient_id" in action:
                action_desc += f" → patient {action['patient_id']}"
            print(f"  Step {steps:>2}: {action_desc:<35} "
                  f"r={reward:.4f}  treated={obs['treated_count']}/{obs['total_patients']}")

        if done:
            break

    # ── Build report ────────────────────────────────────────────────────────
    treated = obs["treated_count"]
    total = obs["total_patients"]
    efficiency = treated / total if total > 0 else 0.0
    avg_reward = total_reward / steps if steps > 0 else 0.0

    report = {
        "total_reward": round(total_reward, 4),
        "average_reward": round(avg_reward, 4),
        "steps": steps,
        "patients_treated": treated,
        "total_patients": total,
        "treatment_rate": round(efficiency, 4),
        "reward_violations": reward_violations,
        "obs_key_violations": obs_key_violations,
        "patient_key_violations": patient_key_violations,
        "rewards_per_step": rewards,
        "actions_taken": actions_taken,
    }

    return total_reward, steps, report


def validate_env(
    env: HospitalEnv,
    seed: int = 42,
) -> list[str]:
    """
    Run a battery of checks against the env to catch submission issues.

    Returns a list of issue strings. Empty list = all clear ✅.
    """
    issues: list[str] = []

    # ── 1. Reproducibility (FIX #5) ─────────────────────────────────────────
    env_a = HospitalEnv(seed=seed)
    obs_a = env_a.reset()

    env_b = HospitalEnv(seed=seed)
    obs_b = env_b.reset()

    if obs_a["patients"] != obs_b["patients"]:
        issues.append("REPRODUCIBILITY: same seed produces different patients")

    # ── 2. Observation key stability (FIX #6) ───────────────────────────────
    missing = _REQUIRED_OBS_KEYS - set(obs_a.keys())
    if missing:
        issues.append(f"OBS_KEYS: reset() observation missing: {missing}")

    # ── 3. Patient key presence (FIX #2) ────────────────────────────────────
    for p in obs_a.get("patients", []):
        missing_p = _REQUIRED_PATIENT_KEYS - set(p.keys())
        if missing_p:
            issues.append(f"PATIENT_KEYS: patient {p.get('id','?')} missing: {missing_p}")
            break  # one example is enough

    # ── 4. Reward normalization (FIX #3) ────────────────────────────────────
    test_env = HospitalEnv(seed=seed)
    test_obs = test_env.reset()
    waiting = [p for p in test_obs["patients"] if p["status"] == "waiting"]

    if waiting:
        # Test assign_doctor
        _, r, _, _ = test_env.step({"type": "assign_doctor", "patient_id": waiting[0]["id"]})
        if not (0.0 <= r <= 1.0):
            issues.append(f"REWARD_NORM: assign_doctor returned {r}")

    test_env2 = HospitalEnv(seed=seed)
    test_obs2 = test_env2.reset()
    waiting2 = [p for p in test_obs2["patients"] if p["status"] == "waiting"]

    if waiting2 and test_obs2["available_icu_beds"] > 0:
        _, r, _, _ = test_env2.step({"type": "send_to_icu", "patient_id": waiting2[0]["id"]})
        if not (0.0 <= r <= 1.0):
            issues.append(f"REWARD_NORM: send_to_icu returned {r}")

    # Test wait
    _, r, _, _ = test_env.step({"type": "wait"})
    if not (0.0 <= r <= 1.0):
        issues.append(f"REWARD_NORM: wait returned {r}")

    # ── 5. Step safety — invalid actions (FIX #7) ──────────────────────────
    safe_env = HospitalEnv(seed=seed)
    safe_env.reset()

    bad_actions = [
        None,
        42,
        "assign_doctor",
        {},
        {"type": "invalid_action"},
        {"type": "assign_doctor"},                          # missing patient_id
        {"type": "assign_doctor", "patient_id": None},      # None id
        {"type": "assign_doctor", "patient_id": 9999},      # non-existent patient
        {"type": "send_to_icu", "patient_id": "abc"},       # wrong type
    ]

    for bad in bad_actions:
        try:
            obs, r, d, _ = safe_env.step(bad)
            if not (0.0 <= r <= 1.0):
                issues.append(f"STEP_SAFETY: bad action {bad!r} returned reward {r}")
            if not isinstance(obs, dict):
                issues.append(f"STEP_SAFETY: bad action {bad!r} returned non-dict obs")
            if not isinstance(d, bool):
                issues.append(f"STEP_SAFETY: bad action {bad!r} returned non-bool done")
        except Exception as e:
            issues.append(f"STEP_SAFETY: bad action {bad!r} raised {type(e).__name__}: {e}")

    # ── 6. Done after all treated ───────────────────────────────────────────
    quick_env = HospitalEnv(num_patients=1, num_doctors=1, icu_beds=1, max_steps=50, seed=seed)
    qobs = quick_env.reset()
    pid = qobs["patients"][0]["id"]
    _, _, done, _ = quick_env.step({"type": "send_to_icu", "patient_id": pid})
    if not done:
        issues.append("DONE: episode did not end after all patients treated")

    # ── 7. available_actions present and non-empty before done ──────────────
    act_env = HospitalEnv(seed=seed)
    act_obs = act_env.reset()
    avail = act_obs.get("available_actions")
    if avail is None:
        issues.append("AGENT_FRIENDLY: available_actions missing from observation")
    elif len(avail) == 0:
        issues.append("AGENT_FRIENDLY: available_actions empty at episode start")

    # ── 8. Stalemate detection (FIX #4) ────────────────────────────────────
    # Create an impossible scenario: 0 doctors, 0 ICU, patients waiting
    stale_env = HospitalEnv(num_patients=3, num_doctors=0, icu_beds=0, max_steps=100, seed=seed)
    stale_obs = stale_env.reset()
    _, _, stale_done, _ = stale_env.step({"type": "wait"})
    if not stale_done:
        issues.append("STALEMATE: env did not terminate with 0 doctors + 0 icu beds")

    return issues
