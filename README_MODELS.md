# G1 Robot — Model Definitions, Metrics &amp; Launch Commands

Three trained tasks, each documented with the same metric set: episode reward, episode length, success rate / task error, fall count, key reward components (reward vs. penalty), and training wall time.

---

## 1. Throw as far as possible (no target)

**What it is:** the robot throws a ball with maximum forward distance, in a straight line, without falling. No target — pure distance + stability.

**Model:** `policies/g1_free_throw_ppo_distance_only/best_model.zip`
**Environment:** `envs/g1_free_throw_env.py` → `G1FreeThrowEnv`

### How to launch

```bash
source .venv/bin/activate
python scripts/play_ppo_pitcher.py --model policies/g1_free_throw_ppo_distance_only/best_model.zip
```

To retrain from scratch:
```bash
python scripts/train_ppo_pitcher.py --timesteps 500000 --n-envs 4 --output policies/g1_free_throw_ppo_distance_only
```

### Metrics (100 episodes, deterministic policy, seeds 0–99)

| Metric | Value |
|---|---|
| Episode reward | mean 22.63, std 8.00 (min 9.02, max 31.37) |
| Episode length | 43.0 steps (~0.86s) |
| Success rate / task error | *no target* — task error reinterpreted as: distance 1.98m ± 0.73m, lateral drift 0.02m ± 0.27m |
| Fall count | 0 / 100 (0%) |
| Training wall time | 1390s (~23 min), 500,000 steps, ~365–390 fps |

### Key reward components (avg / episode)

| Component | Type | Avg/ep |
|---|---|---|
| Alive bonus | **Reward** | +2.15 |
| Distance bonus (forward, capped, straight-line bonus) | **Reward** | +21.23 |
| Side penalty (lateral deviation from straight line) | **Penalty** | −0.75 |
| Short-throw penalty (backward or &lt;0.1m throws) | **Penalty** | 0.00 (inactive) |
| Fall penalty | **Penalty** | 0.00 (inactive) |

---

## 2. Throw at a fixed target ("dartboard")

**What it is:** same throwing mechanics, but the objective changes from "maximize distance" to "hit the center of a physical target" — a tilted, 5-ring dartboard the ball sticks to on contact, placed 2.8m out at ~chest height.

**Model:** `policies/g1_dartboard_ppo/best_model.zip`
**Environment:** same `G1FreeThrowEnv`, with `TARGET_POS=(2.8, 0.0, 0.6)`, `SUCCESS_RADIUS=0.28`
**Warm-started from:** the distance-only model above (not trained from scratch)

### How to launch

```bash
source .venv/bin/activate
python scripts/play_ppo_pitcher.py --model policies/g1_dartboard_ppo/best_model.zip
```

To retrain (warm-started):
```bash
python scripts/train_ppo_pitcher.py --timesteps 500000 --n-envs 4 \
    --init policies/g1_free_throw_ppo_distance_only/best_model.zip \
    --output policies/g1_dartboard_ppo
```

### Metrics (20 episodes, deterministic policy, seeds 0–19)

| Metric | Value |
|---|---|
| Episode reward | mean 26.20, std 13.53 (min 4.81, max 38.84) |
| Episode length | 46.5 steps (~0.93s) |
| **Success rate** (bullseye, within 0.28m of center) | **60%** (12/20) |
| **Task error** (landing_error, distance from ball to target center) | mean 0.44m, std 0.49m |
| Fall count | 0 / 20 (0%) |
| Training wall time | 1390s (~23.2 min), 500,000 steps, ~365 fps |

### Key reward components (avg / episode)

| Component | Type | Avg/ep |
|---|---|---|
| Alive bonus | **Reward** | +2.32 |
| Accuracy bonus (exponential, peaks at target center) | **Reward** | +9.32 |
| Bullseye bonus (flat, only within success radius) | **Reward** | +12.00 |
| Miss penalty (linear, scales with distance from center — gives a "get closer" gradient even on bad misses) | **Penalty** | −1.33 |
| Fall penalty | **Penalty** | 0.00 (inactive) |

---

## 3. Walk 0.5m, then throw at the target

**What it is:** two separately-trained policies chained sequentially in one MuJoCo scene — a walking policy moves the robot ~0.5m forward and settles, then a throwing policy (fine-tuned specifically for this handoff) takes over and throws at the same dartboard target, now placed 2.8m from the *original* start point to keep the effective throw distance the same after the walk.

**Models:**
- Walk: `policies/g1_walk_ppo_v2/best_model_frozen_test.zip` (frozen snapshot — never overwritten by later walk experiments)
- Throw: `policies/g1_dartboard_postwalk_ppo/best_model.zip` (fine-tuned from the dartboard model above, specifically on randomized post-walk starting states)

**Environments:** `envs/g1_walk_env.py` → `G1WalkEnv` (walk phase), then `envs/g1_free_throw_env.py` → `G1FreeThrowEnv` (throw phase). State (full qpos/qvel) is copied directly from the walk env's final frame into the throw env, since both share the identical MuJoCo scene.

### How to launch

```bash
source .venv/bin/activate
python scripts/play_walk_then_throw.py
```

Runs continuously in a loop (one walk-then-throw cycle per iteration) using a fixed seed (`walk_env.reset(seed=247)`) for a consistent, reproducible best-case demo (0.06m landing error). To sample different walk conditions, edit the seed near the top of the script's main loop.

To retrain the throw side for this handoff (walk side stays frozen):
```bash
python scripts/train_ppo_dartboard_postwalk.py --timesteps 300000 --n-envs 4 \
    --init policies/g1_dartboard_ppo/best_model.zip
```

### Metrics (30 episodes, varied walk seeds 0–29, deterministic policy)

| Metric | Value |
|---|---|
| Episode reward | *(reward accrues across two separate envs; not directly comparable to tasks 1–2 — see task error instead)* |
| Episode length | 202.1 steps combined (~4.04s: walk + throw) |
| **Success rate** (bullseye) | **20%** (6/30) — *best single seed (247): 0.06m, effectively a bullseye* |
| **Task error** (landing_error) | mean 0.863m, std 0.494m, median 0.903m |
| Fall count | walk: 0/30 (0%) · throw: 0/30 (0%) |
| Training wall time (fine-tuning only; walk side reused) | 2321s (~38.7 min), 300,000 steps — slower per-step than tasks 1–2 because every reset also runs a full walk rollout first |

### Key reward components

Identical reward function to Task 2 (the fine-tuning only changes *what state `reset()` starts from*, not the reward itself):

| Component | Type |
|---|---|
| Alive bonus | **Reward** |
| Accuracy bonus (exponential near center) | **Reward** |
| Bullseye bonus (flat, in-radius) | **Reward** |
| Miss penalty (linear, distance-scaled) | **Penalty** |
| Fall penalty | **Penalty** |

### Why success rate is lower than Task 2 alone

The throw policy now has to release accurately from whatever pose/velocity the walk left the robot in — a much wider range of starting conditions than "always start from a clean standing pose." Fine-tuning on 300k randomized post-walk states roughly doubled accuracy over the un-tuned dartboard model in this setting (10% → 20% success, mean error ~1.2m → ~0.86m), and eliminated visible arm oscillation during the follow-through, but has not yet closed the full gap to Task 2's standalone 60%. The current script sidesteps this by using a specific known-good seed (247) for a reliable demo; the underlying policy's general-case accuracy is the 20%/0.86m figures above.

---

## Reward/penalty legend (applies to all three throwing setups)

- **Reward** = the model is trying to *earn* this by acting well.
- **Penalty** = subtracted from the total; the model is trying to *avoid* this.
- A component reading **0.00 (inactive)** means it exists in the code but wasn't triggered in this particular evaluation (e.g. no falls occurred, so `FALL_PENALTY` never fired) — it is still part of the trained objective, just not observed in that sample.

*Generated 2026-07-26 from live evaluation of the saved models (fresh rollouts, not cached numbers) plus training log files.*
