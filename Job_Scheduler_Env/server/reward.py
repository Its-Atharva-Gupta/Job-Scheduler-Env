"""Reward computation for the Job Scheduler environment."""


def compute_reward(newly_done: list, missed: list, current_time: int, action_valid: bool) -> float:
    """
    Compute reward based on state transitions this step only.

    Args:
        newly_done:   Jobs that completed (done=True) for the first time this step.
        missed:       Jobs whose deadline has passed and are not running or done.
        current_time: Current simulation time.
        action_valid: Whether the action taken was valid.

    Returns:
        Reward signal as a float.
    """
    reward = 0.0

    if action_valid:
        reward += 1.0
    else:
        reward -= 1.0

    for job in newly_done:
        reward += 5.0
        if current_time <= job.deadline:
            reward += 2.0

    for job in missed:
        reward -= 0.5

    return reward
