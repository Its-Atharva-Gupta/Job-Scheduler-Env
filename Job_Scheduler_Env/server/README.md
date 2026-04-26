---
title: Job Scheduler Env Environment Server
emoji: 🗓️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Job Scheduler Env

A reinforcement learning environment where an agent learns to optimally schedule jobs across machines. Built on Meta's OpenEnv framework, served over HTTP + WebSocket.

## Overview

The agent receives a list of jobs (each with a duration, deadline, and arrival time) and a list of machines, and must assign every job to a machine before deadlines expire.

### Task Levels

| Level | Jobs | Machines |
|-------|------|----------|
| 1     | 3    | 3        |
| 2     | 5    | 4        |
| 3+    | 7    | 5        |

### Action

```python
JobSchedulerEnvAction(action="(job_id, machine_id)")
```

One action assigns one job to one machine. The agent should submit one action per job per episode.

### Observation

| Field | Type | Description |
|-------|------|-------------|
| `current_time` | int | Current simulation time |
| `job_info` | list | All jobs with id, duration, deadline, arrival, status |
| `machine_info` | list | All machines with id, occupied status, running job |
| `llm_description` | str | Natural-language summary of current state |
| `done` | bool | Whether the episode has ended |
| `reward` | float | Reward received this step |

### Reward

| Event | Reward |
|-------|--------|
| Valid action | +1.0 |
| Invalid action | -1.0 |
| Job completed (new this step) | +5.0 |
| Job completed on time | +2.0 bonus |
| Job missed deadline (new this step) | -0.5 |

## Quick Start

```python
from Job_Scheduler_Env import JobSchedulerEnvAction, JobSchedulerEnvEnv

with JobSchedulerEnvEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    obs = result.observation

    print(obs.llm_description)
    # e.g. "Time: 0. 3 jobs pending, 0 done, 0 failed. 3/3 machines free."

    # Assign job 1001 to machine 2001
    result = env.step(JobSchedulerEnvAction(action="(1001, 2001)"))
    print(result.reward)
    print(result.observation.llm_description)
```

## Running Locally

```bash
# Install dependencies
uv sync

# Start the server
uvicorn server.app:app --reload --port 8000
```

## GRPO Training (Kaggle / Colab)

A ready-to-run training notebook is included:

```
train_unsloth_colab.ipynb
```

It trains `Qwen/Qwen2.5-1.5B-Instruct` with Unsloth + GRPO against the live environment server, and includes a base-vs-fine-tuned comparison cell at the end.

## Building the Docker Image

```bash
docker build -t job-scheduler-env:latest -f server/Dockerfile .
```

Connect via Docker:

```python
env = JobSchedulerEnvEnv.from_docker_image("job-scheduler-env:latest")
```

## Deploying to Hugging Face Spaces

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Push to a specific repo
openenv push --repo-id my-org/job-scheduler-env

# Push as private
openenv push --private
```

After deployment the space exposes:
- `/web` — Interactive UI
- `/docs` — OpenAPI/Swagger docs
- `/health` — Health check
- `/ws` — WebSocket endpoint

## Project Structure

```
Job_Scheduler_Env/
├── __init__.py                          # Exports JobSchedulerEnvEnv, Action, Observation
├── client.py                            # HTTP/WebSocket client
├── models.py                            # Action and Observation pydantic models
├── openenv.yaml                         # OpenEnv manifest
├── pyproject.toml                       # Dependencies
├── train_unsloth_colab.ipynb            # GRPO training notebook
└── server/
    ├── app.py                           # FastAPI application
    ├── Job_Scheduler_Env_environment.py # Core environment logic
    ├── reward.py                        # Reward computation
    └── Dockerfile                       # Container image
```
