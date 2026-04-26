---
title: "job-scheduler-env — teaching an LLM to schedule jobs before the deadline kills it"
thumbnail: docs/blog/hero.png
authors: 
  - user: Atharva1232
---

# job-scheduler-env — teaching an LLM to schedule jobs before the deadline kills it

**TL;DR**

- A live RL environment where an agent assigns jobs to machines under time pressure — built on Meta's OpenEnv framework, served over HTTP + WebSocket.
- A **transition-based reward signal** that fires exactly once per event, eliminating the double-counting bug that plagues naive step-reward implementations.
- **Three task levels, three different scaling challenges**: Level 1 is a warm-up, Level 2 adds pressure, Level 3 is a combinatorial test.
- End-to-end **GRPO training** on `Qwen/Qwen2.5-1.5B-Instruct` (Unsloth + TRL), with a base-vs-fine-tuned comparison cell built into the notebook.
- A **custom interactive UI** on HF Spaces so anyone can play the scheduling game themselves — no API key, no setup.

---

---

## Architecture at a Glance

![System Architecture](docs/arch_overview.png)

*Figure 1 — End-to-end system architecture. The GRPO trainer (top) drives an LLM agent client that communicates over WebSocket/HTTP with the FastAPI environment server. The server wraps the core `JobSchedulerEnvEnvironment` which manages simulation state and delegates reward computation to `reward.py`.*

---

## Why job scheduling?

Job scheduling is a deceptively hard domain for LLMs. The surface looks easy: here are some jobs, here are some machines, assign them. A general-purpose model will handle the first few assignments confidently. Then arrivals start overlapping, deadlines start expiring, and machines start filling up — and the model starts hallucinating job IDs, assigning to occupied machines, and missing deadlines it should have seen coming.

The failure mode is invisible in a chat window and legible in a reward curve. That's the point.

We built job-scheduler-env to make that failure mode measurable — and to give GRPO a clean signal to train against. The environment runs as an HTTP + WebSocket server compatible with any `openenv-core` client, so any RL stack can drive it.

---

## The environment

![Core Data Models](docs/data_models.png)

*Figure 2 — UML class diagram of the core data models. `JobSchedulerEnvEnvironment` aggregates `Job` and `Machine` objects, accepts a `JobSchedulerEnvAction`, and returns a `JobSchedulerEnvObservation` on every step.*

Each episode initialises a set of jobs and machines. A **Job** has:
- `id` — a short integer (1001, 1002, …) to limit hallucination surface
- `duration` — how long it runs once assigned
- `arrival` — the earliest time it can be assigned
- `deadline` — the time by which it must finish or it's a miss

A **Machine** has an `id`, an `occupied` flag, a `become_free_time`, and a pointer to the job currently running on it.

The agent receives the full job and machine state as JSON on every step and must emit one assignment per action:

```
(job_id, machine_id)
```

The environment advances `current_time` to the next event (earliest machine-free or job-arrival), releases completed jobs, validates the assignment, and returns the new observation with reward and done flag.

### Episode Lifecycle

![Episode Lifecycle Flowchart](docs/episode_flow.png)

*Figure 3 — Step-by-step episode flow. The LLM generates an `(job_id, machine_id)` action each turn; the environment validates it, advances simulation time, releases completed jobs, computes a transition-based reward, and checks the terminal condition.*

### Three task levels

| Level | Jobs | Machines | Challenge |
|-------|------|----------|-----------|
| 1 | 3 | 3 | Warm-up — simple 1-to-1 assignment |
| 2 | 5 | 4 | Queuing — one machine must run two jobs |
| 3 | 7 | 5 | Combinatorial — deadlines start conflicting |

Level 3 is where base models fall apart: they run out of context budget tracking the queue, confuse job IDs they've already assigned, and emit assignments for done jobs. That's the signal GRPO has to clean up.

---

## The reward function — why transition-based matters

![Reward Signal Design](docs/reward_signal.png)

*Figure 4 — Reward signal breakdown and a sample episode timeline showing exactly when each component fires. Every reward event triggers exactly once per state transition — never per accumulated history.*


The naive reward implementation iterates over all jobs on every step and adds `+5.0` for every `job.done == True`. That means if Job A finishes at step 1, it pays out `+5.0` again at step 2, step 3, and every step after. A 3-job episode accumulates `+15` in phantom reward before the episode ends, dwarfing the signal from the actual assignments. GRPO trains against noise.

The fixed version snapshots `already_done` IDs before releasing machines, diffs to get `newly_done`, and fires each reward component exactly once per state transition:

```python
already_done = {j.id for j in self.jobs if j.done}

# ... release machines ...

newly_done = [j for j in self.jobs if j.done and j.id not in already_done]
missed     = [j for j in self.jobs if not j.done and not j.is_happening
                                   and self.current_time > j.deadline]

reward = compute_reward(newly_done, missed, self.current_time, action_valid)
```

```python
# reward.py — fires once per transition, not once per step
def compute_reward(newly_done, missed, current_time, action_valid):
    reward = 0.0
    reward += 1.0 if action_valid else -1.0      # action quality
    for job in newly_done:
        reward += 5.0                             # completion
        if current_time <= job.deadline:
            reward += 2.0                         # on-time bonus
    for job in missed:
        reward -= 0.5                             # deadline miss (fires once)
    return reward
```

The on-time bonus is the shaped incentive: it rewards the agent for assigning high-priority jobs early, not just for eventually assigning everything.

---

## What an episode actually looks like

```
[reset]  task_level=1
         Jobs:     1001 (dur=3, arrival=0, deadline=8)
                   1002 (dur=5, arrival=1, deadline=12)
                   1003 (dur=2, arrival=0, deadline=6)
         Machines: 2001 (free), 2002 (free), 2003 (free)
         Time: 0

[step 1] action:  (1003, 2001)   ← assign shortest-deadline job first
         reward:  +1.0  (valid assignment)
         obs:     Machine 2001 busy until t=2. 1003 running.

[step 2] action:  (1001, 2002)
         reward:  +1.0  (valid)
         obs:     Time advances to 2. 1003 finishes. Machine 2001 free.
                  newly_done=[1003], current_time=2 ≤ deadline=6
                  reward += 5.0 + 2.0 = +7.0  (completion + on-time)

[step 3] action:  (1002, 2001)
         reward:  +1.0  (valid)
         obs:     Time=3. 1001 finishes (dur=3). Machine 2002 free.
                  newly_done=[1001], current_time=3 ≤ deadline=8
                  reward += 5.0 + 2.0 = +7.0

[step 4] action:  (done — all jobs assigned, env advances time)
         obs:     Time=7. 1002 finishes. done=True.
                  newly_done=[1002], current_time=7 ≤ deadline=12
                  reward += 5.0 + 2.0 = +7.0

episode total:  +1+8 + +1+8 + +1+8 = 27.0   all on time, all valid
```

A base model on Level 3 typically outputs valid JSON but assigns jobs out of arrival order, lands the wrong job on a busy machine (getting `-1.0` for the invalid action), and misses 1–2 deadlines. The reward curve shows the difference immediately.

---

## Training setup — GRPO on Qwen2.5-1.5B

The full pipeline lives in `train_unsloth_colab.ipynb`. A live run of the notebook (with full output and reward logs) is available on Kaggle: https://www.kaggle.com/code/atharvagupta123/notebook24895e5d56

Key configuration:

```python
# Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    load_in_4bit=True,
    max_seq_length=768,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, ...)

# Dataset — 500 prompts, mixed across all three task levels, shuffled
# 200 × level 1 + 200 × level 2 + 100 × level 3

# GRPO config
GRPOConfig(
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    num_generations=4,       # 4 rollouts per prompt — enough contrast, fits T4 VRAM
    max_steps=500,
)
```

**Why `num_generations=4` and not 2?** GRPO's gradient is an advantage estimate computed by ranking K rollouts within the same group. With K=2, one rollout is always "better" and one always "worse" regardless of how similar they are — the variance of the estimate is high and the signal is noisy. K=4 gives the optimizer enough contrast to learn from without exhausting T4 VRAM.

**Why `learning_rate=5e-5` and not `2e-4`?** GRPO is sensitive to learning rate — at `2e-4` the policy update overshoots on the first few steps and destabilises training before the reward signal has stabilised.

The reward function is the live environment itself, not a proxy: every rollout calls `/reset` and `/step` on the running FastAPI server, so the model is always graded against the real simulator.

---

## Evaluation — base model vs fine-tuned

The notebook's final cell loads a fresh copy of the base model and runs both models for 20 episodes on Level 1 prompts:

```
=============================================
                       Base    Fine-tuned
=============================================
Mean reward           3.900         ?
Median reward         3.000         ?
Std dev               4.712         ?
Min reward           -1.000         ?
Max reward           17.000         ?
=============================================
```

*(Results update on each training run — re-run the comparison cell after training to populate.)*

The base model already scores positive on Level 1 because Qwen2.5-1.5B-Instruct is instruction-tuned and can parse the JSON format. The fine-tuned model should show:
- Higher mean on Level 2 and Level 3, where deadline pressure requires reasoning about arrival order
- Lower variance — fewer episodes where the model misses all deadlines
- Higher rate of on-time completions (the `+2.0` bonus events)

The honest caveat from the sre-gym team applies here too: **500 GRPO steps on a 1.5B model with K=4 rollouts is a starting point, not a ceiling**. The environment, the reward signal, and the training loop are all ready to scale — more steps, larger model, K=8 rollouts. The numbers will follow.

---

## Try it

The environment is live. Pick a task level, assign jobs to machines, and watch the reward accumulate (or not).

<iframe
  src="https://Atharva1232-Job_Scheduler_Env.hf.space/ui"
  frameborder="0"
  width="100%"
  height="700">
</iframe>

---

## The claim

job-scheduler-env is a minimal but non-trivial RL testbed that exposes exactly the class of failure modes — hallucinated IDs, deadline blindness, greedy assignment without lookahead — that general-purpose LLMs exhibit on combinatorial planning tasks. The reward function fires on state transitions, not on accumulated history, so the gradient signal is clean. The three task levels provide a natural curriculum. The environment server is OpenEnv-compatible, so any GRPO trainer can drive it without modification.

Built for the OpenEnv hackathon — by atharva. Apache 2.0.
