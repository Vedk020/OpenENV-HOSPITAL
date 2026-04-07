"""
app.py — Quick demo of the HospitalEnv simulation.

Run:  python app.py
"""

from env import HospitalEnv
from env.grader import grade_episode


def greedy_agent(obs: dict) -> dict:
    """
    A simple greedy agent that:
    1. Sends severity-5 patients to the ICU if beds are available.
    2. Assigns doctors to the highest-severity waiting patient.
    3. Waits if nothing useful can be done.
    """
    waiting = [
        p for p in obs["patients"] if p["status"] == "waiting"
    ]
    if not waiting:
        return {"type": "wait"}

    # Sort by severity descending, then by wait_time descending
    waiting.sort(key=lambda p: (p["severity"], p["wait_time"]), reverse=True)

    top = waiting[0]

    # Critical → ICU (if beds available)
    if top["severity"] >= 5 and obs["available_icu_beds"] > 0:
        return {"type": "send_to_icu", "patient_id": top["id"]}

    # Assign a doctor to the most critical patient
    if obs["available_doctors"] > 0:
        return {"type": "assign_doctor", "patient_id": top["id"]}

    return {"type": "wait"}


def main() -> None:
    env = HospitalEnv(num_patients=8, num_doctors=3, icu_beds=2, max_steps=20, seed=42)

    print("=" * 60)
    print("  Hospital Environment — Greedy Agent Demo")
    print("=" * 60)

    # Run the episode step-by-step with logging
    obs = env.reset()
    print(f"\n🏥 Initial state: {obs['total_patients']} patients, "
          f"{obs['available_doctors']} doctors, "
          f"{obs['available_icu_beds']} ICU beds\n")

    step = 0
    total_reward = 0.0

    while True:
        action = greedy_agent(obs)
        obs, reward, done = env.step(action)
        total_reward += reward
        step += 1

        action_desc = action["type"]
        if "patient_id" in action:
            action_desc += f" (patient {action['patient_id']})"

        print(f"  Step {step:>2}: {action_desc:<30}  "
              f"reward={reward:.4f}  treated={obs['treated_count']}/{obs['total_patients']}  "
              f"docs={obs['available_doctors']}  icu={obs['available_icu_beds']}")

        if done:
            break

    print(f"\n{'─' * 60}")
    print(f"  Episode finished in {step} steps")
    print(f"  Total reward : {total_reward:.4f}")
    print(f"  Treated      : {obs['treated_count']}/{obs['total_patients']}")
    print(f"{'─' * 60}")

    # Also run the grader for a clean report
    print("\n📊 Grader report (fresh episode):")
    env2 = HospitalEnv(num_patients=10, num_doctors=3, icu_beds=2, max_steps=25, seed=99)
    total_r, steps, report = grade_episode(env2, greedy_agent)
    for k, v in report.items():
        if k != "rewards_per_step":
            print(f"  {k:<20}: {v}")


if __name__ == "__main__":
    main()
