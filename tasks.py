"""
tasks.py — Task preset definitions loaded from openenv.yaml.

Provides a clean Python interface to retrieve task configurations
by name (easy, medium, hard) for use in inference and the API.
"""

import yaml
from pathlib import Path
from typing import Any

# ── Load YAML once at import time ──────────────────────────────────────────

_YAML_PATH = Path(__file__).parent / "openenv.yaml"

with open(_YAML_PATH, "r") as f:
    _CONFIG = yaml.safe_load(f)

# ── Public API ─────────────────────────────────────────────────────────────

TASKS: dict[str, dict[str, Any]] = _CONFIG.get("tasks", {})


def get_task(name: str) -> dict[str, Any]:
    """
    Retrieve a task configuration by name.

    Parameters
    ----------
    name : 'easy', 'medium', or 'hard'

    Returns
    -------
    dict with keys: description, num_patients, num_doctors, icu_beds

    Raises
    ------
    KeyError if task name not found.
    """
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {list(TASKS.keys())}")
    return TASKS[name]


def list_tasks() -> list[str]:
    """Return all available task names."""
    return list(TASKS.keys())
