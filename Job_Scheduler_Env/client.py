# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Job Scheduler Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import JobSchedulerEnvAction, JobSchedulerEnvObservation


class JobSchedulerEnvEnv(
    EnvClient[JobSchedulerEnvAction, JobSchedulerEnvObservation, State]
):
    """
    Client for the Job Scheduler Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with JobSchedulerEnvEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(JobSchedulerEnvAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = JobSchedulerEnvEnv.from_docker_image("Job_Scheduler_Env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(JobSchedulerEnvAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: JobSchedulerEnvAction) -> Dict:
        """
        Convert JobSchedulerEnvAction to JSON payload for step message.

        Args:
            action: JobSchedulerEnvAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action": action.action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[JobSchedulerEnvObservation]:
        """
        Parse server response into StepResult[JobSchedulerEnvObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with JobSchedulerEnvObservation
        """
        obs_data = payload.get("observation", {})
        observation = JobSchedulerEnvObservation(
            current_time=obs_data.get("current_time", 0),
            job_info=obs_data.get("job_info", []),
            machine_info=obs_data.get("machine_info", []),
            llm_description=obs_data.get("llm_description", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
