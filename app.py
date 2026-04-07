"""
app.py — Submission-ready demo of the HospitalEnv simulation.

Demonstrates:
  • Agent-friendly action format with patient_id
  • Reward normalization (every step in [0, 1])
  • Reproducible results via seed
  • Stalemate & edge-case handling
  • Full validation suite

Run:  python app.py
"""

from env import HospitalEnv
from env.grader import grade_episode, validate_env


# ─── Greedy Agent ──────────────────────────────────────────────────────────

def greedy_agent(obs: dict) -> dict:
    """
    A deterministic greedy agent that uses available_actions from the
    observation to make decisions — never guesses an invalid action.

    Priority:
      1. Send severity-5 patients to ICU (if bed available)
      2. Assign doctor to highest-severity + longest-waiting patient
      3. Wait
    """
    available = obs.get("available_actions", [])
    if not available:
        return {"type": "wait"}

    waiting = [p for p in obs["patients"] if p["status"] == "waiting"]
    if not waiting:
        return {"type": "wait"}

    # Sort by severity desc → wait_time desc
    waiting.sort(key=lambda p: (p["severity"], p["wait_time"]), reverse=True)
    top = waiting[0]

    # 1) Critical → ICU
    if top["severity"] >= 5:
        icu_actions = [
            a for a in available
            if a["type"] == "send_to_icu" and a.get("patient_id") == top["id"]
        ]
        if icu_actions:
            return icu_actions[0]

    # 2) Assign doctor to the most urgent
    for candidate in waiting:
        doc_actions = [
            a for a in available
            if a["type"] == "assign_doctor" and a.get("patient_id") == candidate["id"]
        ]
        if doc_actions:
            return doc_actions[0]

    # 3) Fallback
    return {"type": "wait"}


# ─── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  🏥  HospitalEnv — Submission-Ready Demo")
    print("=" * 65)

    # ── Phase 1: Validation suite ──────────────────────────────────────────
    print("\n📋 Running validation suite...")
    issues = validate_env(HospitalEnv())
    if issues:
        print(f"  ❌ {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"     • {issue}")
    else:
        print("  ✅ All checks passed — env is submission-ready!")

    # ── Phase 2: Reproducibility check ─────────────────────────────────────
    print("\n🔁 Reproducibility check (seed=42)...")
    env_a = HospitalEnv(seed=42)
    obs_a = env_a.reset()
    env_b = HospitalEnv(seed=42)
    obs_b = env_b.reset()
    match = obs_a["patients"] == obs_b["patients"]
    print(f"  Same seed → same patients: {'✅ YES' if match else '❌ NO'}")

    # ── Phase 3: Step-by-step episode ──────────────────────────────────────
    print("\n🎮 Running greedy agent (8 patients, 3 docs, 2 ICU)...")
    env = HospitalEnv(num_patients=8, num_doctors=3, icu_beds=2, max_steps=20, seed=42)
    obs = env.reset()

    # Show initial patients
    print(f"\n  {'ID':>3}  {'Sev':>3}  {'Wait':>4}  {'Status':<10}")
    print(f"  {'─'*3}  {'─'*3}  {'─'*4}  {'─'*10}")
    for p in obs["patients"]:
        print(f"  {p['id']:>3}  {p['severity']:>3}  {p['wait_time']:>4}  {p['status']:<10}")

    print(f"\n  Available doctors: {obs['available_doctors']}")
    print(f"  Available ICU:    {obs['available_icu_beds']}")
    print(f"  Legal actions:    {len(obs['available_actions'])}")
    print()

    step = 0
    total_reward = 0.0

    while True:
        action = greedy_agent(obs)
        obs, reward, done = env.step(action)
        total_reward += reward
        step += 1

        action_desc = action["type"]
        if "patient_id" in action:
            action_desc += f" → patient {action['patient_id']}"

        status = "🏁" if done else "  "
        print(f"  {status} Step {step:>2}: {action_desc:<32} "
              f"r={reward:.4f}  "
              f"done={obs['treated_count']}/{obs['total_patients']}  "
              f"docs={obs['available_doctors']}  icu={obs['available_icu_beds']}")

        if done:
            break

    print(f"\n{'─' * 65}")
    print(f"  Episode finished in {step} steps")
    print(f"  Total reward     : {total_reward:.4f}")
    print(f"  Average reward   : {total_reward/step:.4f}")
    print(f"  Treatment rate   : {obs['treated_count']}/{obs['total_patients']}")
    print(f"{'─' * 65}")

    # ── Phase 4: Edge case — invalid actions ───────────────────────────────
    print("\n🛡️  Edge case: invalid actions (should NOT crash)...")
    safe_env = HospitalEnv(seed=42)
    safe_env.reset()

    bad_actions = [
        ("None input", None),
        ("number input", 42),
        ("string input", "assign_doctor"),
        ("empty dict", {}),
        ("unknown type", {"type": "fly_helicopter"}),
        ("missing patient_id", {"type": "assign_doctor"}),
        ("null patient_id", {"type": "assign_doctor", "patient_id": None}),
        ("non-existent patient", {"type": "assign_doctor", "patient_id": 9999}),
        ("string patient_id", {"type": "send_to_icu", "patient_id": "abc"}),
    ]

    all_safe = True
    for name, bad_action in bad_actions:
        try:
            obs, r, d = safe_env.step(bad_action)
            ok = 0.0 <= r <= 1.0 and isinstance(obs, dict) and isinstance(d, bool)
            symbol = "✅" if ok else "❌"
            if not ok:
                all_safe = False
            print(f"  {symbol} {name:<30} → r={r:.4f}  done={d}")
        except Exception as e:
            all_safe = False
            print(f"  ❌ {name:<30} → CRASHED: {e}")

    if all_safe:
        print("  ✅ All invalid actions handled gracefully!")

    # ── Phase 5: Stalemate detection ───────────────────────────────────────
    print("\n🔒 Stalemate test (0 doctors, 0 ICU beds)...")
    stale = HospitalEnv(num_patients=3, num_doctors=0, icu_beds=0, max_steps=100, seed=42)
    stale.reset()
    _, _, stale_done = stale.step({"type": "wait"})
    print(f"  Done after 1 step: {'✅ YES (deadlock detected)' if stale_done else '❌ NO (stuck forever!)'}")

    # ── Phase 6: Grader report ─────────────────────────────────────────────
    print("\n📊 Grader report (10 patients, seed=99):")
    env2 = HospitalEnv(num_patients=10, num_doctors=3, icu_beds=2, max_steps=25, seed=99)
    total_r, steps, report = grade_episode(env2, greedy_agent)

    for k, v in report.items():
        if k not in ("rewards_per_step", "actions_taken"):
            print(f"  {k:<25}: {v}")

    # ── Final verdict ──────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    verdict_issues = validate_env(HospitalEnv())
    if not verdict_issues and all_safe:
        print("  🏆  VERDICT: SUBMISSION READY")
    else:
        print("  ⚠️   VERDICT: ISSUES REMAIN — see above")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
