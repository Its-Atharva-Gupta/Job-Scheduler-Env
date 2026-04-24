#!/usr/bin/env python3
"""
GRPO Training Script — Job Scheduler Env

Follows the OpenEnv + TRL pattern from tran_example.py.
Uses GRPOTrainer with vLLM for efficient generation and training.

Setup (2 terminals):

  # Terminal 1: Start OpenEnv server
  uvicorn Job_Scheduler_Env.server.app:app --reload --port 8000

  # Terminal 2: Run training (small test)
  python Job_Scheduler_Env/train.py --dataset-size 5 --max-steps 20

  # Or full training
  python Job_Scheduler_Env/train.py --dataset-size 50 --max-steps 500
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

from client import JobSchedulerEnvEnv
from models import JobSchedulerEnvAction

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


SYSTEM_PROMPT = """You are a job scheduler. Analyze the scheduling state and output the next action.

Output ONLY: (job_id, machine_id)

Constraints:
- Only assign arrived jobs (arrival_time <= current_time)
- Only assign unscheduled jobs (done=false, is_happening=false)
- Only assign to free machines (occupied=false)
- Consider deadlines"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for Job Scheduler")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B", help="Model to fine-tune")
    parser.add_argument("--env-url", default="http://localhost:8000", help="OpenEnv server URL")
    parser.add_argument("--dataset-size", type=int, default=20, help="Number of episodes")
    parser.add_argument("--max-turns", type=int, default=5, help="Max steps per episode")
    parser.add_argument("--max-completion-length", type=int, default=20, help="Max tokens per generation")
    parser.add_argument("--context-length", type=int, default=2048, help="Model context window length")
    parser.add_argument("--num-generations", type=int, default=2, help="G for GRPO")
    parser.add_argument("--learning-rate", type=float, default=2e-6, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=100, help="Max training steps")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--reward-log", default="reward_log.csv", help="CSV path for rewards")
    return parser.parse_args()


def format_observation(obs) -> str:
    """Format observation into compact agent-readable text."""
    job_info = obs.job_info
    machine_info = obs.machine_info
    llm_description = obs.llm_description

    pending = [j for j in job_info if not j["done"] and not j["is_happening"]]
    free_machines = len([m for m in machine_info if not m["occupied"]])

    return f"{llm_description}\nPending: {len(pending)} | Free: {free_machines}/{len(machine_info)}\nAction:"


def parse_action(text: str) -> str:
    """Extract (job_id, machine_id) from model output."""
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
    episode_id: int = 0,
) -> dict[str, list]:
    """
    Run one full job scheduling episode asynchronously.

    Token accumulation across turns for episode-level GRPO training.
    """
    logger.info(f"=== EPISODE {episode_id} START ===")
    result = await env.reset()
    observation = result.observation
    logger.info(f"Environment reset. Initial observation: current_time={observation.current_time}, num_jobs={len(observation.job_info)}, num_machines={len(observation.machine_info)}")

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []

    episode_history: list[dict] = []
    MAX_TOTAL_TOKENS = 512  # Cap to prevent OOM
    turn = 0

    for turn_idx in range(max_turns):
        turn = turn_idx + 1
        if result.done:
            logger.info(f"Episode done at turn {turn}")
            break

        if len(completion_ids) >= MAX_TOTAL_TOKENS:
            logger.info(f"Token limit reached at turn {turn}")
            break

        # Build prompt with minimal history
        history_text = ""
        if episode_history:
            history_text = "PREV: " + " | ".join(
                f"{h['action']}" for h in episode_history[-2:]
            ) + "\n"

        obs_text = format_observation(observation)
        user_prompt = history_text + obs_text
        logger.debug(f"TURN {turn} PROMPT:\n{user_prompt}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = apply_chat_template(tokenizer, messages)

        logger.info(f"TURN {turn}: Generating completion (prompt_len={len(prompt_text.split())} words)")

        # Generate with vLLM via TRL with strict length limit
        # max_completion_length is configured in GRPOConfig
        rollout_outputs = generate_rollout_completions(
            trainer,
            [prompt_text],
        )[0]

        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        logger.info(f"TURN {turn}: Raw completion: '{completion_text}'")
        logger.info(f"TURN {turn}: Completion tokens: {len(rollout_outputs['completion_ids'])}, logprobs: {rollout_outputs.get('logprobs', [])}")

        # Parse action
        action_str = parse_action(completion_text)
        logger.info(f"TURN {turn}: Parsed action: {action_str}")

        if not action_str:
            logger.warning(f"TURN {turn}: Failed to parse action from '{completion_text}'")
            step_rewards.append(-1.0)
            episode_history.append({
                "action": completion_text[:30],
                "reward": -1.0,
            })
            continue

        try:
            # Execute action
            logger.info(f"TURN {turn}: Executing action {action_str}")
            action = JobSchedulerEnvAction(action=action_str)
            result = await env.step(action)
            observation = result.observation
            reward = float(result.reward or 0.0)
            step_rewards.append(reward)

            logger.info(f"TURN {turn}: Step result - reward={reward}, done={result.done}, current_time={observation.current_time}")
            logger.info(f"TURN {turn}: Observation - {len([j for j in observation.job_info if not j['done']])} pending jobs, {len([m for m in observation.machine_info if not m['occupied']])} free machines")

            episode_history.append({
                "action": action_str,
                "reward": reward,
            })

            if result.done:
                logger.info(f"TURN {turn}: Episode done signal received")
                break
        except Exception as e:
            logger.error(f"TURN {turn}: Step error: {e}", exc_info=True)
            step_rewards.append(-0.5)
            episode_history.append({
                "action": action_str,
                "reward": -0.5,
            })
            break

    total_reward = sum(step_rewards) if step_rewards else -1.0
    logger.info(f"=== EPISODE {episode_id} END ===")
    logger.info(f"Total turns: {turn}, Total reward: {total_reward}, Step rewards: {step_rewards}")
    logger.info(f"Total prompt_ids: {len(prompt_ids)}, completion_ids: {len(completion_ids)}, logprobs: {len(logprobs)}")

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "total_reward": total_reward,
    }


# Reward functions
def reward_total(completions: list[str], **kwargs) -> list[float]:
    """Return total episode reward."""
    rewards = kwargs.get("total_reward", [])
    result = [float(r) for r in rewards] if rewards else [0.0 for _ in completions]
    logger.info(f"reward_total called: {len(completions)} completions, rewards={result}")
    return result


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Job Scheduler — GRPO Training (OpenEnv + TRL)")
    logger.info("=" * 60)
    logger.info(f"Agent model:           {args.model_id}")
    logger.info(f"Episodes:              {args.dataset_size}")
    logger.info(f"Max completion length: {args.max_completion_length}")
    logger.info(f"Max steps:             {args.max_steps}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create environment (async client)
    env = JobSchedulerEnvEnv(base_url=args.env_url)

    # Dataset
    dataset_prompt = "Schedule jobs optimally"
    dataset = Dataset.from_dict({"prompt": [dataset_prompt] * args.dataset_size})

    # Output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name_clean = args.model_id.replace("/", "-")
    output_dir = Path(args.output_dir or f"outputs/job-scheduler-{model_name_clean}-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # GRPO config with learning-friendly hyperparameters
    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        generation_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        # Context and length settings
        max_length=args.context_length,  # Total context window
        # Prevent policy divergence
        temperature=0.5,  # Lower = closer to base model
        lr_scheduler_type="cosine",
        warmup_steps=2,
        max_grad_norm=0.5,  # Tighter gradient clipping
        beta=0.1,  # Higher KL penalty to prevent divergence
        # Logging
        logging_steps=1,
        save_steps=10,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
        mean = sum(all_rewards) / n
        last_5 = all_rewards[-5:]
        mean_5 = sum(last_5) / len(last_5)

        logger.info(f"Episode {episode_counter[0]}: reward={total_r:.2f} | mean={mean:.2f}, mean(5)={mean_5:.2f}")

    # Create persistent event loop for async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Rollout function
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        logger.info(f"Rollout batch: {len(prompts)} episodes")
        episode_prompt_ids: list[list[int]] = []
        episode_completion_ids: list[list[int]] = []
        episode_logprobs: list[list[float]] = []
        total_rewards: list[float] = []

        for ep_idx, prompt_text in enumerate(prompts):
            # Run async rollout using persistent event loop
            logger.info(f"Starting episode {ep_idx} in batch")
            episode = loop.run_until_complete(
                rollout_once(
                    trainer=trainer,
                    env=env,
                    tokenizer=tokenizer,
                    system_prompt=SYSTEM_PROMPT,
                    max_turns=args.max_turns,
                    episode_id=ep_idx,
                )
            )
            episode_prompt_ids.append(episode["prompt_ids"])
            episode_completion_ids.append(episode["completion_ids"])
            episode_logprobs.append(episode["logprobs"])
            total_rewards.append(episode["total_reward"])
            log_episode(episode["total_reward"])

        logger.info(f"Rollout batch complete. Rewards: {total_rewards}")
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
        lora_dropout=0.05,
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
    logger.info(f"Training config:")
    logger.info(f"  - Model: {args.model_id}")
    logger.info(f"  - Context length: {args.context_length}")
    logger.info(f"  - Dataset size: {args.dataset_size}")
    logger.info(f"  - Max steps: {args.max_steps}")
    logger.info(f"  - Max turns per episode: {args.max_turns}")
    logger.info(f"  - Max completion length: {args.max_completion_length}")
    logger.info(f"  - Num generations: {args.num_generations}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Output dir: {output_dir}")
    logger.info("⚠️  First, run: python train_debug.py")
    logger.info("    Verify: (1) rewards vary, (2) actions parse, (3) no errors")
    try:
        trainer.train()
    finally:
        try:
            loop.run_until_complete(env.close())
        except Exception:
            try:
                env.close()
            except Exception:
                pass
        loop.close()

    # Save
    trainer.save_model(str(output_dir))
    logger.info(f"Model saved to {output_dir}")
    logger.info(f"Reward log: {reward_log_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
