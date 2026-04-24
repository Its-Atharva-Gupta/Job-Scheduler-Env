#!/usr/bin/env python3
"""
GRPO Training Script — Job Scheduler Env

Trains a language model (Qwen2.5) to learn optimal job scheduling policies
using Group Relative Policy Optimization (GRPO) from TRL.

The agent receives scheduling state as text and generates job-to-machine assignments.
Each episode is a multi-step scheduling task where the agent must assign all jobs
before deadlines.

Setup (requires running server in separate terminal):

  # Terminal 1: Start OpenEnv server
  uvicorn Job_Scheduler_Env.server.app:app --reload --port 8000

  # Terminal 2: Run training
  python train.py --model-id Qwen/Qwen2.5-0.5B --env-url http://localhost:8000 --dataset-size 20
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import re
from datetime import datetime
from pathlib import Path

# Silence TRL experimental warning for rollout_func
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

from client import JobSchedulerEnvEnv
from models import JobSchedulerEnvAction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# System prompt: tells the model how to behave
SYSTEM_PROMPT = """You are an intelligent job scheduler. Your task is to assign jobs to machines efficiently.

Output ONE job-to-machine assignment per turn in the format: (job_id, machine_id)

IMPORTANT:
- Only assign jobs that have arrived (arrival_time <= current_time)
- Only assign jobs that haven't been scheduled yet (done=false, is_happening=false)
- Only assign to machines that are free (occupied=false)
- Consider job deadlines to avoid late completions
- No explanations or markdown, just the action tuple."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GRPO training for Job Scheduler")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B", help="Model to fine-tune")
    parser.add_argument("--env-url", default="http://localhost:8000", help="OpenEnv server URL")
    parser.add_argument("--dataset-size", type=int, default=20, help="Number of training episodes")
    parser.add_argument("--max-turns", type=int, default=10, help="Max steps per episode")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max tokens per generation")
    parser.add_argument("--num-generations", type=int, default=4, help="G for GRPO")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--logging-steps", type=int, default=1, help="Logging frequency")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--reward-log", default="reward_log.csv", help="CSV path for rewards")
    return parser.parse_args()


def format_observation(obs) -> str:
    """Format observation into agent-readable text."""
    current_time = getattr(obs, "current_time", 0)
    job_info = getattr(obs, "job_info", [])
    machine_info = getattr(obs, "machine_info", [])
    llm_description = getattr(obs, "llm_description", "")

    # Concise state representation
    pending_jobs = [j for j in job_info if not j["done"] and not j["is_happening"]]
    available_machines = [m for m in machine_info if not m["occupied"]]

    text = f"""{llm_description}

AVAILABLE JOBS: {len(pending_jobs)}
{[f"Job {j['id']}: deadline={j['deadline']}, duration={j['duration']}, arrival={j['arrival']}" for j in pending_jobs[:3]]}

FREE MACHINES: {len(available_machines)}/{len(machine_info)}
{[f"Machine {m['id']}" for m in available_machines]}

Assign the next job to a free machine. Output: (job_id, machine_id)"""
    return text


def parse_action(text: str) -> str:
    """Extract job-to-machine action from model response."""
    # Look for pattern like (123, 456) in the response
    match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
    if match:
        return f"({match.group(1)}, {match.group(2)})"
    return None


def apply_chat_template(tokenizer, messages):
    """Apply chat template with fallback."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


async def rollout_once(
    trainer: GRPOTrainer,
    env: JobSchedulerEnvEnv,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    max_turns: int,
) -> dict[str, list]:
    """
    Run one full job scheduling episode asynchronously.

    The agent receives the state, generates actions, and receives rewards.
    Tokens accumulate across turns so GRPO trains on the full episode sequence.
    """
    result = await env.reset()
    observation = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []

    episode_history: list[dict] = []
    MAX_TOTAL_TOKENS = 2048

    for turn in range(max_turns):
        if result.done:
            break

        if len(completion_ids) >= MAX_TOTAL_TOKENS:
            break

        # Build prompt with context
        history_text = ""
        if episode_history:
            history_text = "PREVIOUS ACTIONS:\n"
            for entry in episode_history[-2:]:  # last 2 actions for context
                history_text += f"- {entry['action']}: {entry['feedback']}\n"
            history_text += "\n---\n\n"

        obs_text = format_observation(observation)
        user_prompt = history_text + f"CURRENT STATE:\n{obs_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = apply_chat_template(tokenizer, messages)

        # Generate action with vLLM via TRL
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # Parse action from model output
        action_str = parse_action(completion_text)
        if not action_str:
            step_rewards.append(-1.0)
            episode_history.append({
                "action": completion_text[:50],
                "feedback": "Invalid action format",
            })
            continue

        try:
            # Execute action in environment
            action = JobSchedulerEnvAction(action=action_str)
            result = await env.step(action)
            observation = result.observation
            reward = float(result.reward or 0.0)
            step_rewards.append(reward)

            episode_history.append({
                "action": action_str,
                "feedback": observation.llm_description[:100],
            })

            if result.done:
                break
        except Exception as e:
            logger.warning(f"Step error: {e}")
            step_rewards.append(-0.5)
            episode_history.append({
                "action": action_str,
                "feedback": f"Error: {str(e)[:50]}",
            })
            break

    # Compute episode reward
    total_reward = sum(step_rewards) if step_rewards else -1.0

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "total_reward": total_reward,
    }


# Reward functions (TRL convention)
def reward_total(completions: list[str], **kwargs) -> list[float]:
    """Return total episode reward for each completion."""
    rewards = kwargs.get("total_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]


def main() -> None:
    """Main training loop."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Job Scheduler — GRPO Training (OpenEnv + TRL)")
    logger.info("=" * 60)
    logger.info(f"Agent model:    {args.model_id}")
    logger.info(f"Env URL:        {args.env_url}")
    logger.info(f"Episodes:       {args.dataset_size}")
    logger.info(f"Generations/G:  {args.num_generations}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Connect to environment
    env = JobSchedulerEnvEnv(base_url=args.env_url)

    # Dataset (each entry triggers one episode)
    dataset = Dataset.from_dict({"prompt": ["Schedule jobs optimally"] * args.dataset_size})

    # Output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name_clean = args.model_id.replace("/", "-")
    default_output_dir = Path("outputs") / f"job-scheduler-grpo-{model_name_clean}-{timestamp}"
    output_dir = Path(args.output_dir or default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GRPO config
    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=1,
        generation_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=5,
        temperature=args.temperature,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        loss_type="dapo",
        mask_truncated_completions=True,
        beta=0.01,
    )

    # CSV logging
    reward_log_path = output_dir / args.reward_log
    episode_counter = [0]
    all_rewards = []

    with open(reward_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "timestamp"])

    def log_episode(total_r: float):
        episode_counter[0] += 1
        all_rewards.append(total_r)
        with open(reward_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode_counter[0], total_r, datetime.now().isoformat()])

        n = len(all_rewards)
        mean_all = sum(all_rewards) / n
        last_5 = all_rewards[-5:]
        mean_5 = sum(last_5) / len(last_5)

        logger.info(
            f"Episode {episode_counter[0]}: reward={total_r:.2f} | "
            f"mean={mean_all:.2f}, mean(5)={mean_5:.2f}"
        )

    # Rollout function (called by GRPO trainer each step)
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        episode_prompt_ids: list[list[int]] = []
        episode_completion_ids: list[list[int]] = []
        episode_logprobs: list[list[float]] = []
        total_rewards: list[float] = []

        for prompt_text in prompts:
            # Run async rollout_once in sync context
            episode = asyncio.run(
                rollout_once(
                    trainer=trainer,
                    env=env,
                    tokenizer=tokenizer,
                    system_prompt=SYSTEM_PROMPT,
                    max_turns=args.max_turns,
                )
            )
            episode_prompt_ids.append(episode["prompt_ids"])
            episode_completion_ids.append(episode["completion_ids"])
            episode_logprobs.append(episode["logprobs"])
            total_rewards.append(episode["total_reward"])
            log_episode(episode["total_reward"])

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "total_reward": total_rewards,
        }

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Trainer
    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[reward_total],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting GRPO training...")
    try:
        trainer.train()
    finally:
        env.close()

    # Save
    trainer.save_model(str(output_dir))
    logger.info(f"Model saved to {output_dir}")
    logger.info(f"Reward log: {reward_log_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
