# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Job Scheduler Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import random
import time
from uuid import uuid4

try:
    from .reward import compute_reward
except ImportError:
    from reward import compute_reward
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import JobSchedulerEnvAction, JobSchedulerEnvObservation
except ImportError:
    from models import JobSchedulerEnvAction, JobSchedulerEnvObservation

class Job:
    _counter = 0

    def __init__(self, current_time):
        # Use short, simple IDs to prevent model hallucination
        Job._counter += 1
        self.id = 1000 + Job._counter  # e.g., 1001, 1002, 1003
        self.duration = random.randint(1, 10)
        self.deadline = current_time + random.randint(5, 25)
        self.arrival = current_time + random.randint(0, 5)
        self.is_happening: bool = False
        self.done: bool = False
    

def job_as_json(jobs: list[Job]):
        data = []
        for job in jobs:
            data.append({
        'id' : job.id,
        'duration': job.duration,
        'deadline': job.deadline,
        'arrival': job.arrival,
        'is_happening': job.is_happening,
        'done': job.done
        })
        return data 

class Machine:
    _counter = 0

    def __init__(self, current_time):
        # Use short, simple IDs to prevent model hallucination
        Machine._counter += 1
        self.id = 2000 + Machine._counter  # e.g., 2001, 2002, 2003
        self.occupied = False
        self.become_free_time = 0
        self.job_running: Job | None = None

   
def Machine_as_json(machines:list[Machine]):
        data = []
        for machine in machines:
            data.append({
            'id' : machine.id,
            'occupied' : machine.occupied,
            'become_free_time' : machine.become_free_time,
            'job_running' : machine.job_running.id if machine.job_running else None
            })
        return data
    
class JobSchedulerEnvEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = JobSchedulerEnvEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Job Scheduler Env environment ready!"
        >>>
        >>> obs = env.step(JobSchedulerEnvAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the Job_Scheduler_Env environment."""
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self, seed=None, episode_id=None, **kwargs) -> JobSchedulerEnvObservation:
        """
        Reset the environment.

        Returns:
            JobSchedulerEnvObservation with a ready message
        """
        # Reset ID counters for new episode (IDs will be 1001-1003, 2001-2003)
        Job._counter = 0
        Machine._counter = 0

        self.current_time = 0
        task_level = kwargs.get("task_level", 1)

        if task_level == 1:
            self.num_jobs = 3
            self.num_machines = 3
        elif task_level == 2:
            self.num_jobs = 5
            self.num_machines = 4
        else:
            self.num_jobs = 7
            self.num_machines = 5

        self.jobs : list[Job]= []

        for i in range(self.num_jobs):
            new_job = Job(current_time=self.current_time)
            self.jobs.append(new_job)

        self.machines : list[Machine] = []

        for i in range(self.num_machines):
            self.machines.append(Machine(current_time=self.current_time))

        new_episode_id = episode_id if episode_id else str(uuid4())
        self._state = State(episode_id=new_episode_id, step_count=0)
        self._reset_count += 1
        self.available_jobs = []
        for job in self.jobs:
            self.available_jobs.append(job.id)
        return JobSchedulerEnvObservation(
            done=False,
            reward=0,
            current_time=self.current_time,
            job_info=job_as_json(self.jobs),
            machine_info=Machine_as_json(self.machines),
            llm_description=self._build_description()
        )

    def step(self, action: JobSchedulerEnvAction, **kwargs) -> JobSchedulerEnvObservation:
        """
        Execute a step in the environment by processing a job-to-machine assignment.

        Args:
            action: JobSchedulerEnvAction with action string like "(job_id, machine_id)"

        Returns:
            JobSchedulerEnvObservation with updated state and reward
        """
        self._state.step_count += 1

        # Parse action
        try:
            parts = action.action.strip("() ").split(",")
            job_id = int(parts[0].strip())
            machine_id = int(parts[1].strip())
            action_valid = True
        except (ValueError, IndexError, AttributeError):
            action_valid = False

        # Advance time to next event (earliest machine-free or job-arrival)
        next_events = [m.become_free_time for m in self.machines if m.occupied]
        next_events += [j.arrival for j in self.jobs if not j.is_happening and not j.done]
        if next_events:
            self.current_time = min(next_events)

        # Release completed jobs from machines — track which ones are newly done
        already_done = {j.id for j in self.jobs if j.done}
        for machine in self.machines:
            if machine.occupied and machine.become_free_time <= self.current_time:
                if machine.job_running:
                    machine.job_running.is_happening = False
                    machine.job_running.done = True
                machine.occupied = False
                machine.job_running = None
        newly_done = [j for j in self.jobs if j.done and j.id not in already_done]

        # Process assignment if valid
        if action_valid:
            job = next((j for j in self.jobs if j.id == job_id), None)
            machine = next((m for m in self.machines if m.id == machine_id), None)

            if job and machine and not machine.occupied and not job.done and not job.is_happening and job.arrival <= self.current_time:
                machine.occupied = True
                machine.become_free_time = self.current_time + job.duration
                machine.job_running = job
                job.is_happening = True
            else:
                action_valid = False

        missed = [j for j in self.jobs if not j.done and not j.is_happening and self.current_time > j.deadline]
        reward = compute_reward(newly_done, missed, self.current_time, action_valid)

        # Check done
        done = all(j.done or self.current_time > j.deadline for j in self.jobs)

        return JobSchedulerEnvObservation(
            current_time=self.current_time,
            job_info=job_as_json(self.jobs),
            machine_info=Machine_as_json(self.machines),
            llm_description=self._build_description(),
            done=done,
            reward=reward,
        )

    def _build_description(self) -> str:
        """Build an LLM-friendly description of the current state."""
        pending = [j for j in self.jobs if not j.done and self.current_time <= j.deadline]
        done = [j for j in self.jobs if j.done]
        failed = [j for j in self.jobs if self.current_time > j.deadline and not j.done]
        free_machines = [m for m in self.machines if not m.occupied]
        return (
            f"Time: {self.current_time}. "
            f"{len(pending)} jobs pending, {len(done)} done, {len(failed)} failed. "
            f"{len(free_machines)}/{len(self.machines)} machines free."
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

