# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Job Scheduler Env Environment.

This module creates an HTTP server that exposes the JobSchedulerEnvEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from pathlib import Path

from fastapi.responses import HTMLResponse, RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import JobSchedulerEnvAction, JobSchedulerEnvObservation
    from .Job_Scheduler_Env_environment import JobSchedulerEnvEnvironment
except ModuleNotFoundError:
    from models import JobSchedulerEnvAction, JobSchedulerEnvObservation
    from server.Job_Scheduler_Env_environment import JobSchedulerEnvEnvironment


app = create_app(
    JobSchedulerEnvEnvironment,
    JobSchedulerEnvAction,
    JobSchedulerEnvObservation,
    env_name="Job_Scheduler_Env",
    max_concurrent_envs=1,
)

_UI_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    if _UI_PATH.exists():
        return _UI_PATH.read_text()
    return "<html><body><h2>UI not found — use <a href='/docs'>/docs</a> for API access.</h2></body></html>"


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui")


@app.get("/web", response_class=RedirectResponse, include_in_schema=False)
async def web():
    return RedirectResponse(url="/ui")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m Job_Scheduler_Env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn Job_Scheduler_Env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
