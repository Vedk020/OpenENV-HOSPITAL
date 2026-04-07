"""
Grader — evaluates agent performance over a full episode.

Usage:
    from env import HospitalEnv
    from env.grader import grade_episode

    env = HospitalEnv()
    total_reward, steps, report = grade_episode(env, agent_fn)
"""

from typing import Any, Callable
from .hospital_env import HospitalEnv


def grade_episode(
    env: HospitalEnv,
    agent_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[float, int, dict[str, Any]]:
    """
    Run one full episode and return a performance report.

    Parameters
    ----------
    env      : a HospitalEnv instance
    agent_fn : callable(observation) -> action dict

    Returns
    -------
    total_reward : float — sum of rewards over the episode
    steps        : int   — number of steps taken
    report       : dict  — detailed breakdown
    """
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    rewards: list[float] = []

    while True:
        action = agent_fn(obs)
        obs, reward, done = env.step(action)
        total_reward += reward
        rewards.append(reward)
        steps += 1
        if done:
            break

    # Build report
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
        "rewards_per_step": rewards,
    }

    return total_reward, steps, report
