---
title: "sre-gym — three tiers of SRE incident-response, one rubric that won't let you fake it"
thumbnail: docs/blog/hero_three_tiers.png
authors:
  - user: Madhav189
  - user: dakshdoesdev
---

# sre-gym — three tiers of SRE incident-response, one rubric that won't let you fake it

**TL;DR**

- A live RL environment for SRE agents: 12 incident templates × 6 procgen variants = **72 deterministic scenarios**.
- A **5-component reward rubric** that sums to exactly 1.0, with a heuristic ceiling pinned to `[0.65, 0.80]` and a scripted-expert floor at `≥0.90` — both enforced by CI invariants on every commit.
- **Three tiers, three different bottlenecks**: Triage escalates compute, Strategy escalates horizon, Operations escalates realism.
- **Coliseum** — a parallel-rollout pool server that turns the env into a lease-based HTTP service so any GRPO trainer can drive it.
- An end-to-end SFT → GRPO pipeline trained on Qwen2.5-7B (Unsloth + TRL). Numbers below; honest framing included.

## Why this matters (read this first)

Calibrated incident-response is the capability gap. Every general-purpose LLM is bad at it: they hallucinate confident root causes, over-trust the loudest signal, skip verification, and declare incidents resolved before checking anything. Those failure modes are invisible in chat demos and catastrophic in production. **sre-gym makes them legible enough to measure, then small enough to fix** — and exposes the env via the OpenEnv contract so any RL stack can train against it.

We treat incident-response as a small **world-modelling** problem: the agent has to maintain a hidden-state estimate of which service is actually broken, update it from noisy observations, and commit to irreversible actions under uncertainty. The 5-component rubric grades the *mechanical signature* of that loop — evidence first, hypothesis with calibrated confidence, remediation, verification, only then resolution — instead of rewarding output that merely looks right.

![Three tiers — Triage escalates compute, Strategy escalates horizon, Operations escalates realism](docs/blog/hero_three_tiers.png)

## The hook

Most SRE-agent demos on the timeline show the same scene: a tidy chat window, a hand-curated incident, a model that names the right service on the first try. Nothing trains. Nothing breaks. Nothing learns.

Most SRE-agent demos are prompts dressed up as products. We built the other half.

```bash
# 5-minute local demo — no API keys, no server, no GPU
pip install -e .
ollama pull llama3.2
python -m sre_gym.local triage worker_deploy_cascade
```

The CLI drives `UnifiedIncidentEnvironment` directly and prints per-tick reward, the 5-component score breakdown, and a final summary. Same code path as the HF Space at `https://huggingface.co/spaces/Madhav189/sre-env` — just without the Gradio UI in front of it.

## Three tiers, three bottlenecks

Each tier hardens a *different* dimension. Not difficulty bands — different bottlenecks of the SRE training loop, because SRE is not one-dimensional.

- **Triage** — bounded by **compute**. Student / Kaggle persona, $30 of HF credits or one Colab A100. Pre-digested observations, dense reward shaping, 8K context, 11-action space, 8–13 ticks per episode.
- **Strategy** — bounded by **horizon**. Seed-stage startup persona, 1–2 A100 days. Multi-incident chains with persistent state across episodes; unresolved alerts and pending deploys carry forward as horizon noise.
- **Operations** — bounded by **realism**. Enterprise SRE platform persona, multi-day distributed training. A 22-node service-graph state-machine simulator with 11 chaos patterns pinned to real production post-mortems (Fly.io gossip-protocol deadlock, Cloudflare config rollout, Stripe retry storm, etc.).

Triage is the only tier we trained against, so it gets the most concrete defense. Its world is a deliberately small 4-service topology plus an 11-service noise-decoy pool that surfaces in alerts but never in queries:

![4-service topology + 11 noise decoys](docs/blog/topology_4service.png)

The 12 base templates each pin a different *cognitive failure mode*: `worker_deploy_cascade` teaches deploy-history reasoning; `db_config_rollout` teaches config-vs-code disambiguation with a worker-deploy decoy; `gateway_auth_rollout` teaches the wrong-loud-service trap; `cache_stale_state` teaches the metrics-look-good-but-customers-don't trap. Twelve templates × five procgen variants (jittered metrics, deploy timestamps, decoy rotation) = 72 deterministic-but-distinct scenarios. One variant per template is held out as the eval split.

## The reward rubric

The agent has eleven Pydantic-validated actions: `query_logs`, `query_metrics`, `query_dependencies`, `query_deploys`, `rollback_deploy`, `restart_service`, `isolate_service`, `run_check`, `submit_hypothesis`, `escalate`, `declare_resolved`. A successful episode is `gather evidence → submit_hypothesis → rollback_deploy → restart_service → both run_check pass → declare_resolved`. Wrong rollback target, premature restart, or premature resolution all return negative reward and a typed `failure_type`.

Reward is a 5-component composite, weights baked into `unified_incident_env/server/grader.py`:

```
final_reward = 0.45·outcome
             + 0.20·action_validity
             + 0.15·anticheat
             + 0.10·format
             + 0.10·efficiency

  outcome          root-cause action correct + recovery confirmed
  action_validity  fraction of step() actions that are well-formed Pydantic
  anticheat        declare_resolved blocked unless ≥1 query already ran
  format           submit_hypothesis was called before declare_resolved
  efficiency       exp(-current_tick / optimal_ticks_for_template)
```

![Triage rubric — 5 components, sums to exactly 1.0](docs/blog/rubric_donut.png)

Each component defends against a specific cheat:

| Cheat strategy | Blocked by |
|---|---|
| `declare_resolved` before any query | `anticheat` (0.15) |
| Skip `submit_hypothesis` to save a tick | `format` (0.10) |
| Spam hypotheses to fish for partial credit | hypothesis idempotence + `action_validity` |
| Send malformed actions | `action_validity` (0.20) |
| Resolve before checks pass | `outcome` (0.45) + `premature_resolution` step penalty |

The cleverest piece is the **calibration term inside `submit_hypothesis`**. Its sub-reward is:

```
hypothesis_reward = 0.04·cause_match
                  + 0.03·service_match
                  + 0.03·action_quality
                  + 0.02·calibration
```

The calibration term reads `confidence` from the structured hypothesis payload (a float ∈ [0,1] the model emits as part of the action) and awards `+1.0` for confident-correct, `+0.5` for hedged-correct, `-0.2` for hedged-wrong, `-1.0` for confident-wrong. **A model that bluffs high confidence on a wrong root cause is worse than one that hedges.** That's the whole pitch in one term: we're training calibrated confidence, not just accuracy.

Per-tick rewards are shaped via a potential function (`-step_cost + Δincident_health_potential + bonus - penalty`). Potential-based shaping (Ng et al. 1999) gives GRPO dense intermediate signal without changing the optimal policy.

## What an episode actually looks like

```
[tick 0] obs:    INCIDENT_SUMMARY: gateway returning 503s, error_rate spiked 15min ago.
                  ALERTS: api-gateway p99=12s, worker error_rate=2%, stripe-webhook noisy.
                  DEPLOYS: worker @ -12min (worker@2026.04.23-bad), gateway @ -3h.
        action:  {"action_type":"query_deploys","service":"worker"}
        reward:  +0.05  (right service queried — affected service bonus)

[tick 1] obs:    DEPLOYS[worker]: 2026.04.23-bad rolled out 12 minutes ago.
                  Recent commits include schema migration + retry handler refactor.
        action:  {"action_type":"submit_hypothesis","hypothesis":{
                  "root_cause":"bad_worker_deploy","affected_services":["worker"],
                  "confidence":0.85,"recommended_next_action":"rollback_deploy"}}
        reward:  +0.12  (correct root_cause, confidence=0.85 → calibrated bonus)

[tick 2] action: {"action_type":"rollback_deploy","service":"worker"}
        reward:  +0.20  (correct rollback target — outcome credit landing)

[tick 3] action: {"action_type":"restart_service","service":"database"}
        reward:  +0.05

[tick 4] action: {"action_type":"run_check","check_name":"database_recovery"}
        reward:  +0.05  (check passes)

[tick 5] action: {"action_type":"run_check","check_name":"end_to_end"}
        reward:  +0.05  (check passes)

[tick 6] action: {"action_type":"declare_resolved"}
        reward:  +0.43  (anticheat satisfied + outcome confirmed → terminal bonus)

final_score: 0.94    incident_resolved: true   ticks: 7
breakdown:   outcome=0.45 valid=0.20 anti=0.15 fmt=0.10 eff=0.04
```

The above is the scripted-optimal solve on `worker_deploy_cascade`. Fail any step and the rubric collects: wrong rollback target zeros `outcome`; calling `declare_resolved` before any query loses the full 0.15 `anticheat` slice; emitting a malformed JSON action chips at `action_validity` for the rest of the episode.

## Coliseum — parallel-rollout pool server

The env is a synchronous Python object. GRPO wants K parallel rollouts per scenario. Coliseum is the bridge: a FastAPI pool server that wraps the env in a lease-based HTTP contract:

```
allocate(task_key)                    -> {ok: true, lease_id}
reset(lease_id, task_meta, run_ctx)   -> {ok: true, observation}
exec_tool(lease_id, tool_call)        -> {ok: true, observation}
evaluate(lease_id)                    -> {ok: true, score}
close(lease_id)                       -> {ok: true}
```

8-way concurrent rollouts on a single process via per-lease `asyncio.Lock`; a background reaper evicts idle leases after `COLISEUM_LEASE_TTL_S` so a crashed worker doesn't leak env instances. Point any GRPO trainer at `COLISEUM_BASE_URL` and it runs.

## The training pipeline

The environment is the project. Training scripts orbit around it. We ran a real end-to-end Triage run on the day of the deadline. **GRPO** (group relative policy optimization) is a critic-free RL method that estimates advantages by comparing K rollouts inside the same group — instead of training a learned value function, it ranks the rollouts in a batch and pushes the policy toward the better ones. Lower memory than PPO, fewer moving parts, well-suited to small-data RL.

Pipeline lives in [`notebooks/01_triage_train_grpo_qwen25_7b.ipynb`](notebooks/01_triage_train_grpo_qwen25_7b.ipynb):

1. **SFT cold-start** — Unsloth + Qwen2.5-7B-Instruct (4-bit) + LoRA r=32. The student imitates a 120-episode trajectory corpus harvested by running teacher models (Claude Opus, Llama-3.3-70B via Groq, scripted-optimal) against the env itself. Every trajectory was graded by the same 5-component rubric the trained model is graded by — so the SFT signal and the GRPO reward signal are by-construction aligned. 50 steps, eval perplexity 1.755, lands in the healthy `[1.5, 3.0]` band.
2. **GRPO online** — TRL's `GRPOTrainer`, 40 steps × K=2 rollouts per scenario. Reward is the same composite plus a scenario-aware first-action bonus (`+0.40` correct rollback target, `+0.30` matches `truth.best_next_action`, `+0.15` queries an affected service, `-0.30` premature `declare_resolved` or wrong rollback target).
3. **Held-out eval** — the `__p05` procgen variant of every template (12 scenarios) × 3 seeds = 36 episodes per policy.

### What the run produced

| policy | mean | median | p25 | p75 | resolved_rate |
|---|---|---|---|---|---|
| random | 0.342 | 0.378 | 0.340 | 0.380 | 0/36 |
| **qwen25-7b-sft-only** | **0.379** | 0.380 | 0.378 | 0.380 | 0/36 |
| **qwen25-7b-grpo** | **0.379** | 0.380 | 0.378 | 0.380 | 0/36 |
| heuristic (queries + correct hypothesis, no remediation) | 0.704 | 0.705 | 0.703 | 0.705 | 0/36 |
| scripted-optimal | 0.938 | 0.939 | 0.937 | 0.940 | 36/36 |

![Held-out eval — 12 scenarios × 3 seeds × 5 policies](docs/blog/baselines_bar.png)

The honest reading: **SFT lifted the model 11% above random (0.342 → 0.379). GRPO did not move it further in the time we had** — the K=2 rollouts on a 7B model with a 40-step budget weren't enough to overcome the heuristic plateau at 0.704. The model produces schema-valid JSON 100% of the time (the corpus's `action_type` key sticks) but it does not yet learn the multi-step plan: query → hypothesise → rollback → verify → resolve.

That gap is the env doing its job. The rubric refuses to give partial credit for "looks right" output. A model that emits a polished hypothesis but never calls `rollback_deploy` is rewarded for zero remediation, and the 0.45 `outcome` slice stays empty. **The numbers don't lie about the work the model didn't do** — and that's the load-bearing engineering claim of the whole project.

The training-time weak spot is data scale. The 120-episode corpus + 50-step SFT + 40-step GRPO is what fits inside a hackathon weekend on one A100. Scaling either side (more teacher trajectories, more GRPO steps, K=4 rollouts) is the obvious next move; the env, the rubric, and the Coliseum pool server are ready for it.

## Operations — 22 nodes, 11 chaos patterns, grounded in real incidents

Every chaos pattern is pinned to a real production post-mortem from the last 18 months:

![11 chaos patterns, grounded in real production incidents](docs/blog/chaos_timeline.png)

Reading post-mortems, not blog posts. Fly.io's gossip-protocol deadlock from October 2024 (cluster nodes stop hearing each other, the consensus layer wedges, no service is "down" but everything stalls). Stripe's classic 2022 retry storm (a downstream timeout amplifies into thundering-herd and self-DDoSes the dependency graph). Cloudflare's November 2025 config rollout (a global config push lands a regression that takes down the data plane for a region). Each pattern declares targets, an `inject` knob (`cors_origin_allow_all_inverted_to_block_all`, `signature_validation_off_by_one`, `rls_policy_typo`), a canonical recovery sequence, and a `grader_focus`. Composability is enforced — the `composition_safety` block whitelists safe pairs and caps stacked patterns at three so the simulator can't generate genuinely unrecoverable states.

## Two paths

![Path A — Verified Runbook Skill (zero training) vs Path B — GRPO-Trained Adapter](docs/blog/two_paths.png)

**Path A** packages the env as a Claude Code skill. The agent reads scenario evidence, writes a verified runbook on a clean solve, and reads the runbook on the next attempt. No training, no GPU. Twelve runbook drafts ship in `skill/verified-runbooks/`.

**Path B** is the trained adapter. Same env, different compute envelope. Path A is what an agent ships *today*; Path B is what raises the floor on the templates it sees over and over. The env grades both paths the same way.

## Try it

The Triage env is live. Pick a scenario, pick a model provider, watch each tick stream the action, env response, reward delta, and rubric breakdown.

<iframe src="https://Madhav189-sre-env.hf.space" frameborder="0" width="100%" height="800"></iframe>

For the per-tier deep dives: [`docs/TRIAGE_TIER.md`](docs/TRIAGE_TIER.md) · [`docs/STRATEGY_TIER.md`](docs/STRATEGY_TIER.md) · [`docs/OPERATIONS_TIER.md`](docs/OPERATIONS_TIER.md). For the rubric defense: [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md). For the architectural narrative: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For the operator guide: [`execution.md`](execution.md). The training notebook lives at [`notebooks/01_triage_train_grpo_qwen25_7b.ipynb`](notebooks/01_triage_train_grpo_qwen25_7b.ipynb).

## The claim

sre-gym is the first SRE training environment that grades calibrated confidence as a first-class signal. The rubric tells you exactly where your model is bluffing — to two decimal places, on every commit, with a CI invariant that fails the build if the heuristic ceiling drifts out of band. Train against it and the hidden-state estimate inside your model gets sharper episode by episode. Skip the rubric and your agent stays a chat-window demo.

Built for the OpenEnv-class hackathon, India 2026 — by the Madhav-GPT / dakshdoesdev team. Apache 2.0.
