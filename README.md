# OpenEnv HospitalEnv

A reinforcement learning hospital triage simulation environment built for the [OpenEnv](https://openenv.org) challenge.

An RL agent must prioritize patients with varying severity levels using limited doctors and ICU beds. The environment rewards intelligent resource allocation and penalizes neglect of critical patients.

---

## Project Structure

```
openenv-hospital/
├── app.py                 # FastAPI server (/reset, /step, /state)
├── inference.py           # Evaluator-compatible inference loop with strict logs
├── openenv.yaml           # Task definitions + action/observation schemas
├── Dockerfile             # Container setup for HF Spaces / evaluator
├── requirements.txt       # Python dependencies
├── demo.py                # Standalone CLI demo (not used in submission)
├── env/
│   ├── __init__.py        # Package re-export
│   ├── hospital_env.py    # Core simulation environment (HospitalEnv)
│   └── grader.py          # Validation suite + episode grading
├── models.py              # Pydantic request/response models for the API
└── tasks.py               # Task presets loaded from openenv.yaml
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. Test Endpoints

```bash
# Reset the environment
curl -X POST http://localhost:8000/reset

# Take an action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"type": "assign_doctor", "patient_id": 0}}'

# Get current state
curl http://localhost:8000/state
```

### 4. Run Inference (Evaluator Mode)

```bash
python inference.py
```

---

## Docker

```bash
# Build
docker build -t openenv-hospital .

# Run
docker run -p 8000:8000 openenv-hospital
```

---

## API Endpoints

### `POST /reset`

Reinitializes the environment. Optionally accepts task configuration.

**Request body** (optional):
```json
{
  "seed": 42,
  "num_patients": 8,
  "num_doctors": 2,
  "icu_beds": 1
}
```

**Response:**
```json
{
  "observation": { ... },
  "done": false
}
```

### `POST /step`

Advances the simulation by one time-step with the given action.

**Request body:**
```json
{
  "action": {
    "type": "assign_doctor",
    "patient_id": 1
  }
}
```

**Response:**
```json
{
  "observation": { ... },
  "reward": 0.56,
  "done": false
}
```

### `GET /state`

Returns the current observation snapshot without advancing time.

**Response:**
```json
{
  "observation": { ... }
}
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | No | `http://localhost:8000` | Base URL of the running FastAPI server |
| `MODEL_NAME` | No | `GreedyAgent-Triage-v1` | Model identifier for logging |
| `HF_TOKEN` | No | `""` | Hugging Face API token for OpenAI client |

---

## Action Format

```json
{
  "type": "assign_doctor" | "send_to_icu" | "wait",
  "patient_id": <int>    // required for assign_doctor and send_to_icu
}
```

## Observation Keys (12 — always stable)

| Key | Type | Description |
|---|---|---|
| `patients` | `list[dict]` | Each with `id`, `severity`, `wait_time`, `status` |
| `available_doctors` | `int` | Idle doctors available for assignment |
| `available_icu_beds` | `int` | Free ICU beds |
| `current_step` | `int` | Current time step |
| `max_steps` | `int` | Episode length limit |
| `treated_count` | `int` | Patients finished (treated + icu) |
| `waiting_count` | `int` | Patients still waiting |
| `treating_count` | `int` | Patients currently being treated |
| `total_patients` | `int` | Total patients in the episode |
| `done` | `bool` | Whether the episode has ended |
| `available_actions` | `list[dict]` | All legal actions the agent can take |
| `action_space` | `dict` | Schema describing the action format |

---

## Tasks

Defined in `openenv.yaml`:

| Task | Patients | Doctors | ICU Beds | Goal |
|---|---|---|---|---|
| **easy** | 4 | 3 | 2 | Basic prioritization |
| **medium** | 8 | 2 | 1 | Resource trade-offs |
| **hard** | 15 | 2 | 1 | Optimization under pressure |

---

## Reward Design

- Per-step rewards are **always clamped to `[0.0, 1.0]`**.
- `assign_doctor`: severity-weighted score (0.2–1.0) + wait bonus (up to +0.3).
- `send_to_icu`: severity 5 → 1.0, severity 4 → 0.7, severity 3 → 0.4, severity ≤ 2 → 0.15.
- `wait`: penalized proportionally to number of critical patients waiting.
- Invalid actions: reward = 0.0 (never crashes).

---

## Inference Log Format

The evaluator parses these exact log lines:

```
[START] task=medium env=HospitalEnv model=GreedyAgent-Triage-v1
[STEP] step=1 action={"type": "assign_doctor", "patient_id": 4} reward=0.7 done=False
[STEP] step=2 action={"type": "wait"} reward=0.15 done=False
...
[END] success=True steps=12 score=0.82
```

---

## License

MIT
