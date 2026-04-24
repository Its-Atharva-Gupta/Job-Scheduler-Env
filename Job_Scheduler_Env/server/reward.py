"""Reward computation for the Job Scheduler environment."""


def compute_reward(jobs: list, current_time: int, action_valid: bool) -> float:
    """
    Compute reward based on job scheduling performance.

    Args:
        jobs: List of Job objects in the environment
        current_time: Current simulation time
        action_valid: Whether the action was valid

    Returns:
        Reward signal as a float
    """
    reward = 0.0

    if not action_valid:
        reward -= 0.5

    for job in jobs:
        if job.done:
            slack = job.deadline - current_time
            reward += 1.0 + max(0, slack) * 0.1
        elif current_time > job.deadline and not job.is_happening:
            reward -= 2.0

    return reward
