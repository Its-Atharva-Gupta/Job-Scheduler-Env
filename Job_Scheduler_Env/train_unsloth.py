#!/usr/bin/env python3
"""
GRPO Training Script with Unsloth — Job Scheduler Env

Uses Unsloth for fast job scheduling strategy learning.
"""

import asyncio
import logging
import re
from datasets import Dataset
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# === UNSLOTH SETUP ===
from unsloth import FastLanguageModel

max_seq_length = 2048
lora_rank = 8

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

# === ENVIRONMENT SETUP ===
from client import JobSchedulerEnvEnv
from models import JobSchedulerEnvAction

env = JobSchedulerEnvEnv(base_url="http://localhost:8000")

# === TRAINING PROMPT ===
PROMPT = """Analyze the job scheduling state and output the next action.
Current time, pending jobs, and available machines are provided.
Output ONLY: (job_id, machine_id)

Example:
(0, 1)
""".strip()

# === UTILITY FUNCTIONS ===
def parse_action(text: str) -> str:
    """Extract (job_id, machine_id) from model output."""
    match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
    if match:
        return f"({match.group(1)}, {match.group(2)})"
    return None

def format_observation(obs) -> str:
    """Format observation for the model."""
    pending = [j for j in obs.job_info if not j["done"] and not j["is_happening"]]
    free_machines = len([m for m in obs.machine_info if not m["occupied"]])
    return f"Time: {obs.current_time}, Pending jobs: {len(pending)}, Free machines: {free_machines}/{len(obs.machine_info)}"

async def evaluate_strategy(max_turns: int = 5):
    """Run one episode and collect rewards."""
    result = await env.reset()
    observation = result.observation

    step_rewards = []

    for turn in range(max_turns):
        if result.done:
            break

        # Format prompt for this step
        obs_text = format_observation(observation)
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": obs_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate action
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=1.0,
            do_sample=True,
        )
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract action
        action_str = parse_action(completion)
        logger.debug(f"Turn {turn + 1}: Generated '{completion}' -> Action: {action_str}")

        if not action_str:
            step_rewards.append(-1.0)
            continue

        try:
            # Execute action
            action = JobSchedulerEnvAction(action=action_str)
            result = await env.step(action)
            observation = result.observation
            reward = float(result.reward or 0.0)
            step_rewards.append(reward)
            logger.debug(f"Turn {turn + 1}: Reward: {reward}, Done: {result.done}")

            if result.done:
                break
        except Exception as e:
            logger.debug(f"Turn {turn + 1}: Error: {e}")
            step_rewards.append(-0.5)

    total_reward = sum(step_rewards) if step_rewards else -1.0
    return total_reward, step_rewards

# === REWARD FUNCTION ===
def compute_reward(completions, **kwargs):
    """Reward based on total episode reward."""
    total_rewards = kwargs.get("total_reward", [])
    if total_rewards:
        return [float(r) for r in total_rewards]
    return [0.0] * len(completions)

# === TRAINING SETUP ===
def main():
    logger.info("=" * 70)
    logger.info("Job Scheduler — GRPO Training with Unsloth")
    logger.info("=" * 70)

    # Create dataset
    dataset = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": PROMPT}],
            "answer": 0,
        } for _ in range(50)
    ])

    # Calculate prompt length
    sample_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Time: 0, Pending jobs: 3, Free machines: 2/3"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    max_prompt_length = len(sample_prompt.split()) + 50
    max_completion_length = 20  # Just need "(X, Y)"

    logger.info(f"Max prompt length: {max_prompt_length}")
    logger.info(f"Max completion length: {max_completion_length}")

    # GRPO Training config
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
        max_steps=50,
        save_steps=10,
        report_to="none",
        output_dir="outputs_scheduler",
    )

    logger.info("Creating trainer...")

    def rollout_func(prompts, trainer):
        """Rollout function that evaluates strategies."""
        total_rewards = []
        prompt_ids_list = []
        completion_ids_list = []
        logprobs_list = []

        for idx, prompt in enumerate(prompts):
            logger.info(f"Episode {idx}")
            total_reward, step_rewards = asyncio.run(evaluate_strategy(max_turns=5))
            total_rewards.append(total_reward)
            logger.info(f"Episode {idx}: Total reward = {total_reward:.2f}, Steps = {len(step_rewards)}")

            # Return minimal token data
            prompt_ids_list.append([])
            completion_ids_list.append([])
            logprobs_list.append([])

        return {
            "prompt_ids": prompt_ids_list,
            "completion_ids": completion_ids_list,
            "logprobs": logprobs_list,
            "total_reward": total_rewards,
        }

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[compute_reward],
        args=training_args,
        train_dataset=dataset,
        rollout_func=rollout_func,
    )

    logger.info("Starting training...")
    try:
        trainer.train()
    finally:
        asyncio.run(env.close())

    # Save model
    output_dir = Path("outputs_scheduler/final_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained_merged(str(output_dir), tokenizer, save_method="merged_16bit")
    logger.info("Done!")

if __name__ == "__main__":
    main()
