#!/usr/bin/env python3
"""
GRPO Training Script with Unsloth — Job Scheduler Env
(FIXED: system role + prompt structure + debug visibility)
"""

import logging
import re
import subprocess
import sys
import time
import urllib.request
from datasets import Dataset
from pathlib import Path
import torch 
import json

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_debug.log", mode="w"),
    ],
)

# Silence noisy libs
for _noisy in [
    "unsloth","unsloth_zoo","transformers","datasets","trl","peft","accelerate","torch"
]:
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# === MODEL ===
from unsloth import FastLanguageModel

max_seq_length = 768
lora_rank = 4

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    load_in_4bit=True,
    max_seq_length=max_seq_length,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# === ENV ===
from client import JobSchedulerEnvEnv
from models import JobSchedulerEnvAction

env = JobSchedulerEnvEnv(base_url="http://localhost:8000").sync()

# === PROMPT ===
def _build_prompt(obs) -> str:
    return (
        "Assign each job to a machine.\n\n"
        f"Jobs JSON:\n{json.dumps(obs.job_info)}\n\n"
        f"Machines JSON:\n{json.dumps(obs.machine_info)}\n\n"
        "Output one assignment per job in format:\n(job_id, machine_id)"
    )

def _sample_prompt() -> str:
    result = env.reset()
    return _build_prompt(result.observation)

# === ENV EXECUTION ===
def _run_episode(action_strs: list[str]) -> float:
    result = env.reset()
    total_reward = 0.0

    for action_str in action_strs:
        if result.done:
            break
        try:
            result = env.step(JobSchedulerEnvAction(action=action_str))
            total_reward += float(result.reward or 0.0)
        except Exception:
            total_reward -= 0.5

    return total_reward

# === REWARD ===
def env_reward(completions, **kwargs):
    scores = []

    for completion in completions:
        text = completion[0]["content"]

        matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)

        if not matches:
            scores.append(-2.0)
            continue

        actions = [f"({a},{b})" for a,b in matches]

        try:
            reward = _run_episode(actions)
            scores.append(reward)
        except Exception:
            scores.append(-1.0)

    return scores

# === SERVER ===
def _start_server():
    root = Path(__file__).resolve().parent.parent

    proc = subprocess.Popen(
        ["uv","run","uvicorn","Job_Scheduler_Env.server.app:app","--port","8000"],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(30):
        try:
            urllib.request.urlopen("http://localhost:8000/docs")
            return proc
        except:
            time.sleep(1)

    proc.kill()
    raise RuntimeError("Server failed")

# === MAIN ===
def main():
    server = _start_server()

    prompts = [_sample_prompt() for _ in range(100)]

    dataset = Dataset.from_list([
        {
            "prompt": [
                {
                    "role": "system",
                    "content": "You are a job scheduling AI. Always assign ALL jobs to machines using given data."
                },
                {
                    "role": "user",
                    "content": p
                }
            ],
            "answer": 0
        }
        for p in prompts
    ])

    # 🔍 DEBUG: see what model actually sees
    print("\n===== DEBUG CHAT TEMPLATE =====\n")
    print(tokenizer.apply_chat_template(
        dataset[0]["prompt"],
        tokenize=False,
        add_generation_prompt=True
    ))
    print("\n===============================\n")

    token_lens = [
        len(tokenizer.apply_chat_template(
            d["prompt"],
            tokenize=True,
            add_generation_prompt=True
        ))
        for d in dataset
    ]

    max_prompt_length = max(token_lens) + 10
    max_completion_length = max_seq_length - max_prompt_length

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        num_generations=2,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=100,
        logging_steps=1,
        report_to="none",
        output_dir="outputs_scheduler",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[env_reward],
        args=training_args,
        train_dataset=dataset,
    )

    try:
        trainer.train()
        model.save_pretrained_merged(
            "outputs_scheduler/final_model",
            tokenizer,
            save_method="merged_16bit"
        )
    finally:
        env.close()
        server.terminate()
        server.wait(timeout=10)

if __name__ == "__main__":
    main()