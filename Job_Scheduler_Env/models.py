# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Job Scheduler Env Environment.

The Job_Scheduler_Env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class JobSchedulerEnvAction(Action):
    """Action for the Job Scheduler Env environment - just a message to echo."""

    action: str = Field(..., description="What action to take. for example if you want to schedule the job with the id 123 to the machine with id 345, send (123, 345)")


class JobSchedulerEnvObservation(Observation):
    """Observation from the Job Scheduler Env environment - the echoed message."""

    current_time: int = Field(default=0, description="The current simulation time")
    job_info: list = Field(..., description="All the info about a job")
    machine_info: list = Field(default_factory=list, description="machine info")
    llm_description: str =Field(..., description="A text based description of the current situation for llms")
    done: bool = Field(..., description="Whether the episode has terminated")
    reward: float = Field(default=0, description="Reward recieved by agent")

