"""
HospitalEnv — Submission-ready hospital simulation environment for OpenEnv.

Maintains internal state: patients (severity + wait time), doctors, and ICU beds.
Supports actions: assign_doctor, send_to_icu, wait.
Returns (observation, reward, done) tuples from step().

Action format (agent-friendly):
    {"type": "assign_doctor", "patient_id": 2}
    {"type": "send_to_icu",   "patient_id": 0}
    {"type": "wait"}

Observation keys (guaranteed stable across every call):
    patients, available_doctors, available_icu_beds, current_step, max_steps,
    treated_count, waiting_count, treating_count, total_patients, done,
    available_actions, action_space

Reward: always in [0.0, 1.0] — clamped on every code path.
Done:   True when all patients treated, OR max_steps reached, OR stalemate.
"""

import random
import copy
from typing import Any


# ─── Reward clamping utility ────────────────────────────────────────────────

def _clamp_reward(value: float) -> float:
    """Enforce reward ∈ [0.0, 1.0] on every exit path."""
    return round(min(max(float(value), 0.0), 1.0), 4)


# ─── Valid action types (used for validation) ───────────────────────────────

_VALID_ACTION_TYPES = frozenset({"assign_doctor", "send_to_icu", "wait"})

_ACTIONS_REQUIRING_PATIENT = frozenset({"assign_doctor", "send_to_icu"})


class HospitalEnv:
    """
    Simulates a hospital triage environment for reinforcement learning.

    Internal state
    ──────────────
    • patients       — list[dict] each with 'id', 'severity' (1-5),
                       'wait_time', 'status'
    • doctors        — int, number of available (idle) doctors
    • icu_beds       — int, number of free ICU beds
    • current_step   — int, step counter
    • max_steps      — int, episode hard-stop
    • treated_count  — int, patients whose status became 'treated' or 'icu'

    Action format (dict)
    ────────────────────
      'type'       — 'assign_doctor' | 'send_to_icu' | 'wait'
      'patient_id' — int  (REQUIRED for assign_doctor / send_to_icu)
    """

    # ── Severity constants ──────────────────────────────────────────────────

    SEVERITY_CRITICAL = 5
    SEVERITY_HIGH = 4
    SEVERITY_MEDIUM = 3
    SEVERITY_LOW = 2
    SEVERITY_MINOR = 1

    # ── Default configuration ───────────────────────────────────────────────

    DEFAULT_NUM_PATIENTS = 8
    DEFAULT_NUM_DOCTORS = 3
    DEFAULT_ICU_BEDS = 2
    DEFAULT_MAX_STEPS = 20
    DEFAULT_SEED = 42  # ✅ FIX #5: deterministic by default

    # ── Action & observation space descriptors (agent-friendly) ─────────────

    ACTION_SPACE = {
        "type": "dict",
        "keys": {
            "type": {
                "type": "str",
                "values": ["assign_doctor", "send_to_icu", "wait"],
                "required": True,
            },
            "patient_id": {
                "type": "int",
                "required_for": ["assign_doctor", "send_to_icu"],
                "description": "ID of the target patient (from observation.patients[].id)",
            },
        },
    }

    OBSERVATION_SPACE = {
        "patients": "list[dict] — each with keys: id, severity, wait_time, status",
        "available_doctors": "int — idle doctors that can be assigned",
        "available_icu_beds": "int — free ICU beds",
        "current_step": "int — current time step",
        "max_steps": "int — episode length limit",
        "treated_count": "int — patients finished (treated + icu)",
        "waiting_count": "int — patients still waiting",
        "treating_count": "int — patients currently being treated by a doctor",
        "total_patients": "int — total patients in the episode",
        "done": "bool — whether the episode has ended",
        "available_actions": "list[dict] — all legal actions the agent can take right now",
        "action_space": "dict — schema describing the action format",
    }

    # ────────────────────────────────────────────────────────────────────────
    #  __init__
    # ────────────────────────────────────────────────────────────────────────

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
        seed         : RNG seed (default 42 for reproducibility)
        """
        self.num_patients = num_patients if num_patients is not None else self.DEFAULT_NUM_PATIENTS
        self.num_doctors = num_doctors if num_doctors is not None else self.DEFAULT_NUM_DOCTORS
        self.icu_beds = icu_beds if icu_beds is not None else self.DEFAULT_ICU_BEDS
        self.max_steps = max_steps if max_steps is not None else self.DEFAULT_MAX_STEPS

        # ✅ FIX #5: default seed = 42, always reproducible
        self._seed = seed if seed is not None else self.DEFAULT_SEED
        self._rng = random.Random(self._seed)

        # Internal mutable state (populated by reset())
        self.patients: list[dict[str, Any]] = []
        self.available_doctors: int = 0
        self.available_icu_beds: int = 0
        self.current_step: int = 0
        self.treated_count: int = 0
        self._done: bool = False

        # Track doctors who are busy (free up after treatment_time ticks)
        self._busy_doctors: list[dict[str, Any]] = []

    # ────────────────────────────────────────────────────────────────────────
    #  reset()  — Prompt 2
    # ────────────────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """
        Initialise a new episode.

        Creates *num_patients* patients with random severity (1-5) and
        wait_time (0-3). Resets doctors & ICU beds to full capacity.

        Parameters
        ----------
        seed : optional new seed for this episode (overrides constructor seed)

        Returns
        -------
        observation : dict — the initial state snapshot
        """
        # ✅ FIX #5: allow re-seeding per episode for evaluator reproducibility
        if seed is not None:
            self._seed = seed
            self._rng = random.Random(self._seed)

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

    # ────────────────────────────────────────────────────────────────────────
    #  state()  — Prompt 3
    # ────────────────────────────────────────────────────────────────────────

    def state(self) -> dict[str, Any]:
        """
        Return a *read-only snapshot* of the current environment.

        ✅ FIX #2: every patient dict always contains id, severity, wait_time.
        ✅ FIX #6: guaranteed stable keys across every call.
        ✅ Agent-friendly: includes available_actions list.

        Keys (always present)
        ─────────────────────
        patients, available_doctors, available_icu_beds, current_step,
        max_steps, treated_count, waiting_count, treating_count,
        total_patients, done, available_actions, action_space
        """
        waiting = [p for p in self.patients if p["status"] == "waiting"]
        treating = [p for p in self.patients if p["status"] == "treating"]

        return {
            # ✅ FIX #2: deep copy preserves id / severity / wait_time
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
            # ✅ Agent-friendly: tell the agent exactly what it can do
            "available_actions": self._get_available_actions(),
            "action_space": self.ACTION_SPACE,
        }

    # ────────────────────────────────────────────────────────────────────────
    #  available_actions()  — Agent-friendliness
    # ────────────────────────────────────────────────────────────────────────

    def _get_available_actions(self) -> list[dict[str, Any]]:
        """
        Build the full list of legal actions the agent can take right now.

        This makes the env **agent-controllable**: the agent never has to
        guess which actions are valid, it can iterate this list directly.
        """
        if self._done:
            return []

        actions: list[dict[str, Any]] = []

        waiting_patients = [
            p for p in self.patients if p["status"] == "waiting"
        ]

        # assign_doctor actions — one per waiting patient (if doctors free)
        if self.available_doctors > 0:
            for p in waiting_patients:
                actions.append({
                    "type": "assign_doctor",
                    "patient_id": p["id"],
                })

        # send_to_icu actions — one per waiting patient (if ICU beds free)
        if self.available_icu_beds > 0:
            for p in waiting_patients:
                actions.append({
                    "type": "send_to_icu",
                    "patient_id": p["id"],
                })

        # wait is always legal (unless done)
        actions.append({"type": "wait"})

        return actions

    # ────────────────────────────────────────────────────────────────────────
    #  step(action)  — Prompts 4-7
    # ────────────────────────────────────────────────────────────────────────

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Apply *action* to the environment and advance one time-step.

        Parameters
        ----------
        action : dict
            'type'       — 'assign_doctor' | 'send_to_icu' | 'wait'
            'patient_id' — int  (REQUIRED for assign_doctor / send_to_icu)

        Returns  (Prompt 7)
        -------
        observation : dict   — updated state  (stable keys)
        reward      : float  — value in [0.0, 1.0]  (clamped)
        done        : bool   — True when the episode is over
        info        : dict   — auxiliary diagnostics (action_valid, action_type, etc.)

        Safety
        ------
        ✅ FIX #7: invalid/malformed actions never crash — they return
        reward=0.0, done=False (or current done state).
        """
        # ── Already done? Return terminal state. ────────────────────────────
        if self._done:
            return self.state(), 0.0, True, {"action_valid": False, "reason": "episode_already_done"}

        # ── FIX #7: Validate action — never crash ───────────────────────────
        action, valid = self._validate_action(action)
        if not valid:
            # Invalid action: zero reward, no state change, tick the clock
            self._tick()
            self.current_step += 1
            self._check_done()
            return self.state(), 0.0, self._done, {"action_valid": False, "reason": "invalid_action"}

        reward = 0.0
        action_type = action["type"]
        patient_id = action.get("patient_id")

        # ── Execute the chosen action ───────────────────────────────────────

        if action_type == "assign_doctor":
            reward = self._action_assign_doctor(patient_id)

        elif action_type == "send_to_icu":
            reward = self._action_send_to_icu(patient_id)

        elif action_type == "wait":
            reward = self._action_wait()

        # ── Advance simulation clock ────────────────────────────────────────
        self._tick()
        self.current_step += 1

        # ── FIX #4 + #6: Done condition with stalemate detection ────────────
        self._check_done()

        # ── FIX #3: Reward ALWAYS clamped to [0, 1] ─────────────────────────
        info = {
            "action_valid": True,
            "action_type": action_type,
            "treated_count": self.treated_count,
            "waiting_count": len([p for p in self.patients if p["status"] == "waiting"]),
        }
        return self.state(), _clamp_reward(reward), self._done, info

    # ────────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ────────────────────────────────────────────────────────────────────────

    def _validate_action(self, action: Any) -> tuple[dict[str, Any], bool]:
        """
        ✅ FIX #7: Robust action validation.

        Returns (sanitised_action, is_valid).
        Never raises — invalid actions are caught gracefully.
        """
        # Must be a dict
        if not isinstance(action, dict):
            return {"type": "wait"}, False

        action_type = action.get("type")

        # Must have a valid 'type'
        if action_type not in _VALID_ACTION_TYPES:
            return {"type": "wait"}, False

        # Actions that target a patient MUST include patient_id
        if action_type in _ACTIONS_REQUIRING_PATIENT:
            patient_id = action.get("patient_id")

            # patient_id must be an int
            if patient_id is None or not isinstance(patient_id, (int, float)):
                return {"type": "wait"}, False

            patient_id = int(patient_id)

            # patient_id must reference an existing patient
            if not any(p["id"] == patient_id for p in self.patients):
                return {"type": "wait"}, False

            # Return a clean, sanitised action
            return {"type": action_type, "patient_id": patient_id}, True

        # 'wait' — always valid
        return {"type": action_type}, True

    def _check_done(self) -> None:
        """
        ✅ FIX #4 + #6: Comprehensive termination check.

        Done when:
          1. All patients have status 'treated' or 'icu'    (success)
          2. current_step >= max_steps                       (timeout)
          3. STALEMATE: patients still waiting, but no doctors AND
             no ICU beds AND no doctors currently busy        (deadlock)
        """
        all_treated = all(
            p["status"] in ("treated", "icu") for p in self.patients
        )
        if all_treated:
            self._done = True
            return

        if self.current_step >= self.max_steps:
            self._done = True
            return

        # ✅ FIX #4: stalemate detection
        waiting_patients = [p for p in self.patients if p["status"] == "waiting"]
        if waiting_patients:
            no_doctors_free = self.available_doctors <= 0
            no_icu_free = self.available_icu_beds <= 0
            no_doctors_busy = len(self._busy_doctors) == 0

            if no_doctors_free and no_icu_free and no_doctors_busy:
                # Complete deadlock — no way to treat remaining patients
                self._done = True
                return

    def _find_patient(self, patient_id: int | None) -> dict[str, Any] | None:
        """Lookup a patient by id; returns None if not found."""
        if patient_id is None:
            return None
        for p in self.patients:
            if p["id"] == patient_id:
                return p
        return None

    # ────────────────────────────────────────────────────────────────────────
    #  Action handlers
    # ────────────────────────────────────────────────────────────────────────

    def _action_assign_doctor(self, patient_id: int | None) -> float:
        """
        Assign an available doctor to the patient.

        ✅ FIX #1: patient_id is REQUIRED and validated before this is called.

        Reward logic (Prompt 5):
        • severity / 5 → base score  (0.2 – 1.0)
        • +0.1 per unit wait_time    (capped at +0.3)
        • blended & clamped to [0, 1]
        • invalid / impossible → 0.0
        """
        patient = self._find_patient(patient_id)

        # Invalid / impossible — explicit zero (not 0.05, not negative)
        if patient is None or patient["status"] != "waiting":
            return 0.0
        if self.available_doctors <= 0:
            return 0.0

        # Assign the doctor
        self.available_doctors -= 1
        patient["status"] = "treating"
        # Treatment duration: critical patients treated faster
        patient["treatment_timer"] = max(1, 6 - patient["severity"])

        self._busy_doctors.append(
            {"patient_id": patient["id"], "remaining": patient["treatment_timer"]}
        )

        # ── Reward ──────────────────────────────────────────────────────────
        severity_score = patient["severity"] / self.SEVERITY_CRITICAL  # 0.2–1.0
        wait_bonus = min(patient["wait_time"] * 0.1, 0.3)             # 0.0–0.3
        reward = severity_score * 0.7 + wait_bonus

        return _clamp_reward(reward)

    def _action_send_to_icu(self, patient_id: int | None) -> float:
        """
        Send a patient directly to the ICU.

        ✅ FIX #1: patient_id is REQUIRED and validated before this is called.

        Reward:
        • severity 5 → 1.0    (perfect use of ICU)
        • severity 4 → 0.7
        • severity 3 → 0.4
        • severity ≤ 2 → 0.15 (wasteful — low-severity shouldn't use ICU)
        • impossible → 0.0
        """
        patient = self._find_patient(patient_id)

        if patient is None or patient["status"] != "waiting":
            return 0.0
        if self.available_icu_beds <= 0:
            return 0.0

        self.available_icu_beds -= 1
        patient["status"] = "icu"
        self.treated_count += 1

        # Reward: ICU is precious, reserve for critical
        if patient["severity"] == 5:
            return 1.0
        elif patient["severity"] == 4:
            return 0.7
        elif patient["severity"] == 3:
            return 0.4
        else:
            return 0.15

    def _action_wait(self) -> float:
        """
        Do nothing this tick — waiting patients' wait_time increases.

        Reward: penalised when critical patients are still waiting.
        """
        waiting = [p for p in self.patients if p["status"] == "waiting"]
        total_waiting = len(waiting)

        if total_waiting == 0:
            return 0.5  # nothing to do — neutral

        critical_waiting = sum(
            1 for p in waiting if p["severity"] >= self.SEVERITY_HIGH
        )

        # Heavier penalty the more critical patients are waiting
        penalty = critical_waiting / total_waiting
        return _clamp_reward(0.3 - penalty * 0.25)

    # ────────────────────────────────────────────────────────────────────────
    #  Simulation tick
    # ────────────────────────────────────────────────────────────────────────

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

        # Waiting patients accrue wait time
        for p in self.patients:
            if p["status"] == "waiting":
                p["wait_time"] += 1

    # ────────────────────────────────────────────────────────────────────────
    #  Utility
    # ────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"HospitalEnv(patients={self.num_patients}, doctors={self.num_doctors}, "
            f"icu={self.icu_beds}, step={self.current_step}/{self.max_steps}, "
            f"seed={self._seed})"
        )
