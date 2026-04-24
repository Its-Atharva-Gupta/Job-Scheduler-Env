#!/usr/bin/env python3
"""
GRPO Training Script with Unsloth — Job Scheduler Env
Simple: just generate (job_id, machine_id) tuples.
"""

import asyncio
import logging
import re
from datasets import Dataset
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# === UNSLOTH SETUP ===
from unsloth import FastLanguageModel

max_seq_length = 768
lora_rank = 4

logger.info("Loading Qwen2.5 1.5B with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B",
    load_in_4bit=True,
    max_seq_length=max_seq_length,
    offload_embedding=True,
)

logger.info("Applying LoRA patches...")
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# === ENVIRONMENT ===
from client import JobSchedulerEnvEnv
from models import JobSchedulerEnvAction

# === PROMPT ===
prompt = """Schedule the next job to a machine.
Output ONLY: (job_id, machine_id)
Example: (0, 1)
YOu are provided a number of Jobs and 3 Machines. Your task is to schedule the jobs to minimize Machine idle time, and finish tasks before deadline. Only output (Job_id, Machine_id) NOTHING ELSE. Just (Job_id, MAchine_id). For example if you wish to set  
""".strip()

# === REWARD FUNCTION ===
def get_reward(completions, **kwargs):
    """Simple reward: return collected rewards from episodes."""
    total_rewards = kwargs.get("total_reward", [])
    if total_rewards:
        return [float(r) for r in total_rewards]
    return [0.0] * len(completions)

# === TRAINING ===
def main():
    logger.info("=" * 70)
    logger.info("Job Scheduler — GRPO Training")
    logger.info("=" * 70)

    dataset = Dataset.from_list([
        {"prompt": [{"role": "user", "content": prompt}], "answer": 0}
        for _ in range(1)  # Just 1 episode for testing
    ])

    max_prompt_length = len(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True
    )) + 1
    max_completion_length = max_seq_length - max_prompt_length

    logger.info(f"Max prompt length: {max_prompt_length}")
    logger.info(f"Max completion length: {max_completion_length}")

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=2e-4,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=1,  # Just 1 step for testing
        save_steps=20,
        report_to="none",
        output_dir="outputs_scheduler",
    )

    # Custom rollout function that runs episodes
    env = JobSchedulerEnvEnv(base_url="http://localhost:8000")

    def rollout_func(prompts, trainer):
        """Rollout: run episodes and collect rewards."""
        total_rewards = []

        for idx, _ in enumerate(prompts):
            asyncio.run(_run_episode(env, total_rewards))

        return {
            "prompt_ids": [[] for _ in prompts],
            "completion_ids": [[] for _ in prompts],
            "logprobs": [[] for _ in prompts],
            "total_reward": total_rewards,
        }

    async def _run_episode(env, total_rewards):
        """Run one episode."""
        logger.info("\n" + "=" * 70)
        logger.info("EPISODE START")
        logger.info("=" * 70)

        result = await env.reset()
        obs = result.observation
        episode_reward = 0.0
        steps = 0

        logger.info(f"Initial state: time={obs.current_time}, jobs={len(obs.job_info)}, machines={len(obs.machine_info)}")
        logger.info(f"Jobs: {obs.job_info}")
        logger.info(f"Machines: {obs.machine_info}")

        for turn in range(5):
            if result.done:
                logger.info("Environment signaled done")
                break

            # Format state for model
            pending = len([j for j in obs.job_info if not j["done"] and not j["is_happening"]])
            free = len([m for m in obs.machine_info if not m["occupied"]])
            state_text = f"Time: {obs.current_time}, Pending: {pending}, Free: {free}/{len(obs.machine_info)}"

            logger.info(f"\n--- TURN {turn + 1} ---")
            logger.info(f"State: {state_text}")

            # Generate action
            full_prompt = f"{prompt}\n\n{state_text}\n\nAction:"
            logger.info(f"Prompt:\n{full_prompt}")

            inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=10, temperature=1.0, do_sample=True)
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

            logger.info(f"Raw completion: '{completion}'")

            # Parse tuple
            match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', completion)
            if not match:
                logger.warning(f"❌ Failed to parse action from: {completion}")
                episode_reward -= 1.0
                continue

            action_str = f"({match.group(1)}, {match.group(2)})"
            logger.info(f"✓ Parsed action: {action_str}")

            try:
                result = await env.step(JobSchedulerEnvAction(action=action_str))
                obs = result.observation
                step_reward = float(result.reward or 0.0)
                episode_reward += step_reward
                logger.info(f"Step reward: {step_reward}, Total: {episode_reward:.2f}, Done: {result.done}")
                steps += 1
            except Exception as e:
                logger.error(f"❌ Error executing action: {e}", exc_info=True)
                episode_reward -= 0.5

        logger.info(f"\n" + "=" * 70)
        logger.info(f"EPISODE END: total_reward={episode_reward:.2f}, steps={steps}")
        logger.info("=" * 70 + "\n")
        total_rewards.append(episode_reward)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[get_reward],
        args=training_args,
        train_dataset=dataset,
        rollout_func=rollout_func,
    )

    logger.info("Starting training...")
    trainer.train()

    output_dir = Path("outputs_scheduler/final_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained_merged(str(output_dir), tokenizer, save_method="merged_16bit")
    logger.info("Done!")

if __name__ == "__main__":
    main()
