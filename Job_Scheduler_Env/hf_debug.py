#!/usr/bin/env python3
"""
Debug script to test environment on HuggingFace Spaces.
This connects to the online deployed environment instead of local.
"""

from Job_Scheduler_Env import JobSchedulerEnvEnv, JobSchedulerEnvAction


def parse_action(text: str) -> str:
    """Extract (job_id, machine_id) from model output."""
    import re
    match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
    if match:
        return f"({match.group(1)}, {match.group(2)})"
    return None


async def test_environment():
    """Test if HF Spaces environment works and gives non-constant rewards."""
    print("=" * 60)
    print("Testing Job Scheduler Environment on HF Spaces")
    print("=" * 60)

    # Connect to HF Spaces environment
    env = await JobSchedulerEnvEnv.from_env("Atharva1232/Job_Scheduler_Env")
    async with env:

        # Test 1: Reset
        print("\n1. Testing reset()...")
        try:
            result = await env.reset()
            obs = result.observation
            print(f"   ✓ Reset successful")
            print(f"   - Current time: {obs.current_time}")
            print(f"   - Jobs: {len(obs.job_info)}")
            print(f"   - Machines: {len(obs.machine_info)}")
            print(f"   - Description: {obs.llm_description[:100]}")
        except Exception as e:
            print(f"   ✗ Reset failed: {e}")
            return

        # Test 2: Parse action from description
        print("\n2. Testing action extraction...")
        job_ids = [j["id"] for j in obs.job_info]
        machine_ids = [m["id"] for m in obs.machine_info]

        if job_ids and machine_ids:
            test_action_str = f"({job_ids[0]}, {machine_ids[0]})"
            print(f"   Test action: {test_action_str}")
        else:
            print(f"   ✗ No jobs or machines available")
            return

        # Test 3: Step with valid action
        print("\n3. Testing step() with valid action...")
        try:
            action = JobSchedulerEnvAction(action=test_action_str)
            result = await env.step(action)
            obs = result.observation
            reward = result.reward
            done = result.done

            print(f"   ✓ Step successful")
            print(f"   - Reward: {reward}")
            print(f"   - Done: {done}")
            print(f"   - New description: {obs.llm_description[:100]}")
        except Exception as e:
            print(f"   ✗ Step failed: {e}")
            return

        # Test 4: Step with invalid action
        print("\n4. Testing step() with invalid action...")
        try:
            action = JobSchedulerEnvAction(action="(99999, 99999)")
            result = await env.step(action)
            reward = result.reward
            print(f"   ✓ Invalid action handled")
            print(f"   - Reward for invalid action: {reward}")
        except Exception as e:
            print(f"   ✗ Invalid action caused error: {e}")

        # Test 5: Multiple episodes
        print("\n5. Testing multiple episodes...")
        rewards_list = []
        for ep in range(3):
            result = await env.reset()
            obs = result.observation
            ep_reward = 0

            for step in range(5):
                jobs = obs.job_info
                machines = obs.machine_info

                if jobs and machines:
                    job_id = jobs[0]["id"]
                    machine_id = machines[0]["id"]
                    action_str = f"({job_id}, {machine_id})"
                    action = JobSchedulerEnvAction(action=action_str)
                    result = await env.step(action)
                    obs = result.observation
                    ep_reward += float(result.reward or 0.0)

                    if result.done:
                        break

            rewards_list.append(ep_reward)
            print(f"   Episode {ep + 1}: reward={ep_reward:.2f}")

        avg_reward = sum(rewards_list) / len(rewards_list) if rewards_list else 0
        reward_std = (sum((r - avg_reward) ** 2 for r in rewards_list) / len(rewards_list)) ** 0.5
        print(f"   Average: {avg_reward:.2f}, Std: {reward_std:.2f}")

        if reward_std < 0.1:
            print(f"\n   ⚠️  WARNING: Reward variation is very low (std={reward_std:.4f})")
            print(f"      This will prevent the model from learning!")

    print("\n" + "=" * 60)
    print("Diagnosis Complete")
    print("=" * 60)


if __name__ == "__main__":
    print("Testing Job Scheduler Environment on HF Spaces...")
    print("Make sure the space is deployed and running.\n")

    import asyncio
    asyncio.run(test_environment())
