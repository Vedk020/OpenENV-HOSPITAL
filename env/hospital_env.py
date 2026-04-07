"""
HospitalEnv — A hospital simulation environment for reinforcement learning.

Maintains internal state: patients (severity + wait time), doctors, and ICU beds.
Supports actions: assign_doctor, send_to_icu, wait.
Returns (observation, reward, done) tuples from step().
"""

import random
import copy
from typing import Any


class HospitalEnv:
    """
    Simulates a hospital triage environment.

    Internal state
    ──────────────
    • patients       — list[dict] each with 'id', 'severity' (1-5), 'wait_time', 'status'
    • doctors        — int, number of available (idle) doctors
    • icu_beds       — int, number of free ICU beds
    • current_step   — int, step counter
    • max_steps      — int, episode hard-stop
    • treated_count  — int, patients whose status became 'treated' or 'icu'

    Actions (passed to step())
    ──────────────────────────
    dict with:
      'type'       — 'assign_doctor' | 'send_to_icu' | 'wait'
      'patient_id' — int  (required for assign_doctor / send_to_icu)
    """

    # ── Prompt 1 — Class Design ─────────────────────────────────────────────

    # Severity tiers used by the reward function
    SEVERITY_CRITICAL = 5
    SEVERITY_HIGH = 4
    SEVERITY_MEDIUM = 3
    SEVERITY_LOW = 2
    SEVERITY_MINOR = 1

    # Default configuration
    DEFAULT_NUM_PATIENTS = 8
    DEFAULT_NUM_DOCTORS = 3
    DEFAULT_ICU_BEDS = 2
    DEFAULT_MAX_STEPS = 20

    def __init__(
        self,
        num_patients: int | None = None,
        num_doctors: int | None = None,
        icu_beds: int | None = None,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        num_patients : number of patients generated on reset()
        num_doctors  : total available doctors
        icu_beds     : total ICU bed capacity
        max_steps    : maximum steps before forced termination
        seed         : optional RNG seed for reproducibility
        """
        self.num_patients = num_patients or self.DEFAULT_NUM_PATIENTS
        self.num_doctors = num_doctors or self.DEFAULT_NUM_DOCTORS
        self.icu_beds = icu_beds or self.DEFAULT_ICU_BEDS
        self.max_steps = max_steps or self.DEFAULT_MAX_STEPS

        # RNG
        self._rng = random.Random(seed)

        # Internal mutable state (populated by reset())
        self.patients: list[dict[str, Any]] = []
        self.available_doctors: int = 0
        self.available_icu_beds: int = 0
        self.current_step: int = 0
        self.treated_count: int = 0
        self._done: bool = False

        # Track doctors who are busy (will free up after treatment_time ticks)
        self._busy_doctors: list[dict[str, Any]] = []

    # ── Prompt 2 — reset() ──────────────────────────────────────────────────

    def reset(self) -> dict[str, Any]:
        """
        Initialise a new episode.

        Creates *num_patients* patients with random severity (1-5) and
        wait_time (0-3). Resets doctors & ICU beds to full capacity.

        Returns
        -------
        observation : dict  — the initial state snapshot
        """
        self.patients = []
        for i in range(self.num_patients):
            self.patients.append(
                {
                    "id": i,
                    "severity": self._rng.randint(1, 5),
                    "wait_time": self._rng.randint(0, 3),
                    "status": "waiting",        # waiting | treating | treated | icu
                    "treatment_timer": 0,        # countdown while being treated
                }
            )

        self.available_doctors = self.num_doctors
        self.available_icu_beds = self.icu_beds
        self.current_step = 0
        self.treated_count = 0
        self._done = False
        self._busy_doctors = []

        return self.state()

    # ── Prompt 3 — state() ──────────────────────────────────────────────────

    def state(self) -> dict[str, Any]:
        """
        Return a *read-only snapshot* of the current environment.

        Keys
        ----
        patients          — deep-copied list of patient dicts
        available_doctors — int
        available_icu_beds— int
        current_step      — int
        max_steps         — int
        treated_count     — int
        waiting_count     — int (convenience)
        done              — bool
        """
        waiting = [p for p in self.patients if p["status"] == "waiting"]
        treating = [p for p in self.patients if p["status"] == "treating"]

        return {
            "patients": copy.deepcopy(self.patients),
            "available_doctors": self.available_doctors,
            "available_icu_beds": self.available_icu_beds,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "treated_count": self.treated_count,
            "waiting_count": len(waiting),
            "treating_count": len(treating),
            "total_patients": self.num_patients,
            "done": self._done,
        }

    # ── Prompts 4-7 — step(action) ──────────────────────────────────────────

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool]:
        """
        Apply *action* to the environment and advance one time-step.

        Parameters
        ----------
        action : dict
            'type'       — 'assign_doctor' | 'send_to_icu' | 'wait'
            'patient_id' — int  (required for the first two types)

        Returns (Prompt 7)
        -------
        observation : dict   — updated state
        reward      : float  — value in [0, 1]
        done        : bool   — True when the episode is over
        """
        if self._done:
            return self.state(), 0.0, True

        reward = 0.0
        action_type = action.get("type", "wait")
        patient_id = action.get("patient_id")

        # ── Execute the chosen action ───────────────────────────────────────

        if action_type == "assign_doctor":
            reward = self._action_assign_doctor(patient_id)

        elif action_type == "send_to_icu":
            reward = self._action_send_to_icu(patient_id)

        elif action_type == "wait":
            reward = self._action_wait()

        else:
            # Unknown action — small penalty
            reward = 0.05

        # ── Advance simulation clock ────────────────────────────────────────
        self._tick()

        # ── Prompt 6 — Done condition ───────────────────────────────────────
        self.current_step += 1
        all_treated = all(
            p["status"] in ("treated", "icu") for p in self.patients
        )
        if all_treated or self.current_step >= self.max_steps:
            self._done = True

        # ── Prompt 7 — Return format ────────────────────────────────────────
        return self.state(), round(reward, 4), self._done

    # ───────────────────────── private helpers ──────────────────────────────

    def _find_patient(self, patient_id: int | None) -> dict[str, Any] | None:
        """Lookup a patient by id; returns None if not found."""
        if patient_id is None:
            return None
        for p in self.patients:
            if p["id"] == patient_id:
                return p
        return None

    # ── action: assign_doctor ───────────────────────────────────────────────

    def _action_assign_doctor(self, patient_id: int | None) -> float:
        """
        Assign an available doctor to the patient.

        Reward logic (Prompt 5):
        • Higher severity → higher reward  (severity / 5 scaled into [0,1])
        • Bonus for long-waiting patients   (+0.1 per wait unit, capped)
        • Penalty if no doctor available or patient not waiting  → 0.05
        """
        patient = self._find_patient(patient_id)

        # Invalid / impossible actions
        if patient is None or patient["status"] != "waiting":
            return 0.05
        if self.available_doctors <= 0:
            return 0.05

        # Assign the doctor
        self.available_doctors -= 1
        patient["status"] = "treating"
        # Treatment takes (6 - severity) ticks — critical patients are faster
        patient["treatment_timer"] = max(1, 6 - patient["severity"])

        self._busy_doctors.append(
            {"patient_id": patient["id"], "remaining": patient["treatment_timer"]}
        )

        # ── Prompt 5 — Reward calculation ───────────────────────────────────
        severity_score = patient["severity"] / self.SEVERITY_CRITICAL  # 0.2–1.0
        wait_bonus = min(patient["wait_time"] * 0.1, 0.3)             # 0.0–0.3
        reward = min(1.0, severity_score * 0.7 + wait_bonus)          # blend

        return max(0.0, min(1.0, reward))

    # ── action: send_to_icu ─────────────────────────────────────────────────

    def _action_send_to_icu(self, patient_id: int | None) -> float:
        """
        Send a patient directly to the ICU (for critical/high severity).

        Reward logic:
        • severity 5 → 1.0  (perfect decision)
        • severity 4 → 0.7
        • severity ≤ 3 → low reward (wasteful use of ICU)
        • No beds available → 0.05
        """
        patient = self._find_patient(patient_id)

        if patient is None or patient["status"] != "waiting":
            return 0.05
        if self.available_icu_beds <= 0:
            return 0.05

        self.available_icu_beds -= 1
        patient["status"] = "icu"
        self.treated_count += 1

        # Reward: ICU is best reserved for critical patients
        if patient["severity"] == 5:
            return 1.0
        elif patient["severity"] == 4:
            return 0.7
        elif patient["severity"] == 3:
            return 0.4
        else:
            return 0.15  # misuse of ICU for low-severity

    # ── action: wait ────────────────────────────────────────────────────────

    def _action_wait(self) -> float:
        """
        Do nothing — wait times increase for all waiting patients.

        Reward: penalised when critical patients are still waiting.
        """
        critical_waiting = sum(
            1 for p in self.patients
            if p["status"] == "waiting" and p["severity"] >= self.SEVERITY_HIGH
        )
        total_waiting = sum(1 for p in self.patients if p["status"] == "waiting")

        if total_waiting == 0:
            return 0.5  # nothing to do — neutral

        # Heavier penalty the more critical patients are waiting
        penalty = critical_waiting / max(total_waiting, 1)
        return max(0.0, min(1.0, 0.3 - penalty * 0.25))

    # ── simulation tick ─────────────────────────────────────────────────────

    def _tick(self) -> None:
        """
        Advance the simulation by one time unit.

        • Decrement treatment timers for busy doctors.
        • Free doctors whose patients finish treatment.
        • Increment wait_time for all still-waiting patients.
        """
        finished: list[int] = []
        for entry in self._busy_doctors:
            entry["remaining"] -= 1
            if entry["remaining"] <= 0:
                finished.append(entry["patient_id"])

        # Free doctors & mark patients treated
        for pid in finished:
            patient = self._find_patient(pid)
            if patient is not None:
                patient["status"] = "treated"
                self.treated_count += 1
            self.available_doctors += 1

        self._busy_doctors = [e for e in self._busy_doctors if e["remaining"] > 0]

        # Waiting patients get more frustrated
        for p in self.patients:
            if p["status"] == "waiting":
                p["wait_time"] += 1

    # ── utility ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"HospitalEnv(patients={self.num_patients}, doctors={self.num_doctors}, "
            f"icu={self.icu_beds}, step={self.current_step}/{self.max_steps})"
        )
