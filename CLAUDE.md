# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Job Scheduler OpenEnv Environment** — a reinforcement learning training environment where agents learn to optimally schedule jobs across machines. Built on Meta's OpenEnv framework, it provides an HTTP+WebSocket server that can be deployed locally or to Hugging Face Spaces.

**Status**: Incomplete implementation. The environment structure and client/server infrastructure is set up, but the core scheduling logic in `step()` method and reward computation need to be finished.

## Architecture

### High-Level Design

The project implements a **client-server RL environment** following OpenEnv patterns:

```
JobSchedulerEnvEnv (Python Client)
    ↓ (HTTP/WebSocket)
FastAPI Server (server/app.py)
    ↓
JobSchedulerEnvEnvironment (Core Logic)
    ↓
Job & Machine Simulator
```

### Key Components

**Environment State** (`Job_Scheduler_Env/server/Job_Scheduler_Env_environment.py`):
- `Job`: Represents schedulable tasks with duration, deadline, arrival time
- `Machine`: Represents computing resources (3 per episode)
- `JobSchedulerEnvEnvironment`: Manages simulation loop and state transitions
  - `reset(task_level)`: Initializes episode with jobs/machines based on difficulty (level 1 = 3 jobs, level 2 = 5 jobs)
  - `step(action)`: **[INCOMPLETE]** Should process a job-to-machine assignment and advance time
  - Returns observations with `current_time`, `job_info`, `machine_info`, `llm_description`, `reward`, `done`

**Data Models** (`Job_Scheduler_Env/models.py`):
- `JobSchedulerEnvAction`: Agent action (expects job-to-machine tuple as string, e.g., "(job_id, machine_id)")
- `JobSchedulerEnvObservation`: Environment observation with time, jobs, machines, LLM-friendly description

**Client** (`Job_Scheduler_Env/client.py`):
- `JobSchedulerEnvEnv`: Wraps HTTP/WebSocket connection, handles serialization
- Methods: `reset()`, `step(action)`, `close()`
- Supports context manager and Docker startup

### Critical Incomplete Sections

1. **`server/Job_Scheduler_Env_environment.py:step()`** (lines 137-158):
   - Currently has an empty for-loop and returns undefined variables
   - Must: parse action string to (job_id, machine_id), assign job to machine, advance simulation time, check for deadline violations, compute reward

2. **`server/reward.py`**:
   - `compute_reward()` is a stub — define reward signal (e.g., penalties for late jobs, bonus for efficient scheduling)

3. **`Job_Scheduler_Env/client.py:_step_payload()` and `_parse_result()`** (lines 47-84):
   - Currently expect `message` field from echo environment, need updates to map `action` field to actual job scheduling data

## Development

### Setup

```bash
# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

### Common Commands

```bash
# Run the FastAPI server locally (with auto-reload)
uvicorn Job_Scheduler_Env.server.app:app --reload --port 8000

# Or via uv
uv run -m Job_Scheduler_Env.server.app --port 8000

# Test the environment directly (without HTTP)
python Job_Scheduler_Env/server/Job_Scheduler_Env_environment.py

# Build Docker image
docker build -t job-scheduler-env:latest -f Job_Scheduler_Env/server/Dockerfile .

# Deploy to Hugging Face Spaces
cd Job_Scheduler_Env && openenv push
```

### Testing the Environment

The environment can be tested in two ways:

1. **Direct Testing** (no server):
   ```python
   from Job_Scheduler_Env.server.Job_Scheduler_Env_environment import JobSchedulerEnvEnvironment
   env = JobSchedulerEnvEnvironment()
   obs = env.reset(task_level=1)
   obs = env.step(JobSchedulerEnvAction(action="(job_id, machine_id)"))
   ```

2. **Client Testing** (with running server):
   ```python
   from Job_Scheduler_Env import JobSchedulerEnvEnv, JobSchedulerEnvAction
   with JobSchedulerEnvEnv(base_url="http://localhost:8000") as env:
       obs = env.reset()
       obs = env.step(JobSchedulerEnvAction(action="(123, 456)"))
   ```

### Project Structure

```
Job_Scheduler_Env/
├── __init__.py                              # Exports JobSchedulerEnvEnv, Action, Observation
├── client.py                                # EnvClient wrapper for HTTP/WebSocket
├── models.py                                # Pydantic models for Action/Observation
├── openenv.yaml                             # OpenEnv metadata (for HF Spaces deployment)
├── server/
│   ├── app.py                               # FastAPI application
│   ├── Job_Scheduler_Env_environment.py    # Core environment (INCOMPLETE)
│   ├── reward.py                            # Reward computation (STUB)
│   └── Dockerfile                           # Container definition
├── pyproject.toml                           # Dependencies
└── README.md                                # User-facing docs

Root-level test files (old):
├── main.py                                  # Placeholder
├── testing.py                               # Initial Job class prototype
```

### Dependencies

Key packages:
- `openenv-core[core]`: OpenEnv framework for RL environments
- `fastapi`: HTTP server framework
- `uvicorn`: ASGI server
- `pydantic`: Data validation

## Implementation Notes

**Action Format**: Actions are passed as strings representing job-to-machine tuples, e.g., `"(12345, 67890)"`. The `step()` method must parse this format.

**Simulation Time**: `current_time` starts at 0. The `step()` method should increment it (by how much depends on your scheduling model — could be based on job duration or be incremental).

**Reward Signal**: Should penalize jobs that miss deadlines or reward efficient (short makespan) schedules. See `compute_reward()` for where this logic lives.

**Concurrent Sessions**: `max_concurrent_envs=1` in `app.py`. Increase if environment state isolation is guaranteed.

**WebSocket vs HTTP**: Client uses WebSocket for persistent sessions (lower latency). All operations (`reset`, `step`) go through the server via the client.
