"""Reward computation for the Job Scheduler environment."""


def compute_reward(jobs: list, current_time: int, action_valid: bool) -> float:
    """
    Compute reward based on job scheduling performance.

    Reward structure:
    - Valid action: +1.0 (encourage taking actions)
    - Invalid action: -1.0 (discourage invalid moves)
    - Completed job: +5.0 (primary goal)
    - Job on time: +0.5 (bonus for completing before deadline)
    - Job missed deadline: -2.0 (penalize missed deadlines)

    Args:
        jobs: List of Job objects in the environment
        current_time: Current simulation time
        action_valid: Whether the action was valid

    Returns:
        Reward signal as a float
    """
    reward = 0.0

    # Base reward for valid/invalid action
    if action_valid:
        reward += 1.0  # Encourage taking valid actions
    else:
        reward -= 1.0  # Penalize invalid actions

    # Job-based rewards
    for job in jobs:
        if job.done:
            # Reward for completing a job
            reward += 5.0
            # Bonus if completed before deadline
            if current_time <= job.deadline:
                reward += 2.0
        elif current_time > job.deadline and not job.is_happening:
            # Penalize jobs that miss deadline
            reward -= 0.5

    return reward
