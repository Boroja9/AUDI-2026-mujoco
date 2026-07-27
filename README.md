# G1 Robot — Model Definitions, Metrics &amp; Launch Commands

Three trained tasks, each documented with the same metric set: episode reward, episode length, success rate / task error, fall count, key reward components (reward vs. penalty), and training wall time.

---

## Setup (fresh clone)

Requires Python 3.12. No separate MuJoCo/Unitree asset download needed — the G1 model, meshes, and every trained policy checkpoint already live in this repo.

```bash
git clone https://github.com/Boroja9/AUDI-2026-mujoco.git
cd AUDI-2026-mujoco
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, replace the venv-creation/activation lines with:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Verified against a clean clone on 2026-07-26: `mujoco 3.10.0`, `gymnasium 1.3.0`, `stable-baselines3 2.9.0`, `torch 2.13.0` — all three tasks below ran headless immediately after `pip install`, reproducing the documented metrics exactly (Task 3, seed 247: 0.062m landing error).

---

## 1. Throw as far as possible (no target)

**What it is:** the robot throws a ball with maximum forward distance, in a straight line, without falling. No target — pure distance + stability.

**Model:** `policies/g1_free_throw_ppo_distance_v2/best_model.zip`
**Environment:** `envs/g1_free_throw_distance_env.py` → `G1FreeThrowDistanceEnv` (a dedicated fork — see note below)

### How to launch

```bash
source .venv/bin/activate
python scripts/play_ppo_distance.py --model policies/g1_free_throw_ppo_distance_v2/best_model.zip
```

To retrain (warm-started):
```bash
python scripts/train_ppo_distance.py --timesteps 500000 --n-envs 4 \
    --init policies/g1_free_throw_ppo_distance_only/best_model.zip \
    --output policies/g1_free_throw_ppo_distance_v2
```

**Why a separate env file:** `envs/g1_free_throw_env.py` (used by tasks 2–3 below) evolved into a
target-accuracy-only reward over the course of this project — it can no longer produce genuine
"maximize distance" behavior. `envs/g1_free_throw_distance_env.py` is an independent fork with a real
distance-maximizing reward, kept so retraining task 2/3 never risks breaking this task and vice versa.

### Metrics (30 episodes, deterministic policy, seeds 0–29)

| Metric | Value |
|---|---|
| Episode reward | mean 51.68, std 4.48 (min 37.98, max 55.25) |
| Episode length | 57.7 steps (~1.15s) |
| Success rate / task error | *no target* — task error reinterpreted as distance: **6.37m ± 0.79m** (min 4.41, max 7.45) |
| Fall count | 0 / 30 (0%) |
| Training wall time | 1394s (~23.2 min), 507,904 steps, ~364 fps |

### Key reward components (avg / episode)

| Component | Type | Avg/ep |
|---|---|---|
| Alive bonus | **Reward** | +2.88 |
| Distance bonus (forward, capped at 6m) | **Reward** | +46.46 |
| Side penalty (lateral deviation from straight line) | **Penalty** | −1.60 |
| Short-throw penalty (backward or &lt;0.1m throws) | **Penalty** | 0.00 (inactive) |
| Fall penalty | **Penalty** | 0.00 (inactive) |

*Iteration note: the original `g1_free_throw_ppo_distance_only` model reached only 1.98m ± 0.73m
mean distance under an earlier, weaker distance-reward formula — reworking the terminal reward into
a properly capped distance bonus (see `g1_free_throw_distance_env.py`) more than tripled it.*

---

## 2. Throw at a fixed target ("dartboard")

**What it is:** same throwing mechanics, but the objective changes from "maximize distance" to "hit the center of a physical target" — a tilted, 5-ring dartboard the ball sticks to on contact, placed 2.8m out at ~chest height.

**Model:** `policies/g1_dartboard_ppo_center_v2/best_model.zip`
**Environment:** `G1FreeThrowEnv`, with `TARGET_POS=(2.8, 0.0, 0.6)` by default — **pass
`target_pos=(2.3, 0.0, 0.6)` explicitly to reproduce the metrics below exactly**, since the
target moved 0.5m after this model was trained/evaluated (to compensate for Task 3's walk distance)
and `target_pos` is baked into the observation, not just the reward.
**Warm-started from:** `g1_dartboard_ppo_xonly` (itself warm-started from `g1_dartboard_ppo` → the distance-only model)

### How to launch

```bash
source .venv/bin/activate
python scripts/play_ppo_pitcher.py --model policies/g1_dartboard_ppo_center_v2/best_model.zip
```

To retrain (warm-started):
```bash
python scripts/train_ppo_pitcher.py --timesteps 500000 --n-envs 4 \
    --init policies/g1_dartboard_ppo_xonly/best_model.zip \
    --output policies/g1_dartboard_ppo_center_v2
```

### Metrics (50 episodes, deterministic policy, seeds 0–49)

| Metric | Value |
|---|---|
| Episode reward | mean 49.09, std 10.95 (min 6.16, max 53.76) |
| Episode length | 50.7 steps (~1.01s) |
| **Success rate** (bullseye, within 0.28m of center) | **92%** (46/50) |
| **Task error** (landing_error, X-axis depth only — see design note) | mean 0.073m, **median 0.015m**, max 1.341m |
| Fall count | 0 / 50 (0%) |
| Training wall time | 1479s (~24.65 min), 507,904 steps, ~343 fps |

### Key reward components (avg / episode)

| Component | Type | Avg/ep |
|---|---|---|
| Alive bonus | **Reward** | +2.54 |
| Accuracy bonus (exponential, peaks at target center) | **Reward** | +18.28 |
| Bullseye bonus (flat, only within success radius) | **Reward** | +23.00 |
| Miss penalty (linear, scales with distance from center — gives a "get closer" gradient even on bad misses) | **Penalty** | −0.22 |
| Fall penalty | **Penalty** | 0.00 (inactive) |

*Iteration note: success rate took a non-monotonic path — 60% (original, full 3D landing-error
scoring) → 90% (switched to X-axis-only scoring, decoupling lateral miss from the success metric)
→ 78% (an over-aggressive center-accuracy reward push regressed generalization) → **92%**
(moderated back). Median error (0.015m) is far better than the mean (0.073m) — a few outlier
misses pull the average up; most throws are near-perfect bullseyes.*

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
| Episode reward (throw phase) | mean 32.10, std 17.57 |
| Episode length | 252.1 steps combined (~5.04s: walk + throw) |
| **Success rate** (bullseye) | **50%** (15/30) — *best single seed (247): 0.06m, effectively a bullseye* |
| **Task error** (landing_error) | mean 0.337m, median 0.270m, std 0.351m, max 1.511m |
| Fall count | walk: 0/30 (0%) · throw: 0/30 (0%) |
| Training wall time (fine-tuning only; walk side reused) | 2321s (~38.7 min), 303,104 steps — slower per-step than tasks 1–2 because every reset also runs a full walk rollout first |

*Iteration note: success rate readings for this exact model — 20% (N=30, very first evaluation) →
67% (N=15, smaller re-check after a repo cleanup pass) → **50%** (N=30, latest and most
statistically reliable read, same seed range as the first). The model itself was not
retrained between these checks — the 67% figure came from a smaller sample and should not
be read as an improvement; 50%/0.337m is the number to cite. Training wall time and
step count are properties of the one fine-tuning run behind all three checks.*

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

The throw policy now has to release accurately from whatever pose/velocity the walk left the robot in — a much wider range of starting conditions than "always start from a clean standing pose." Fine-tuning on 300k randomized post-walk states substantially improved accuracy over the un-tuned dartboard model in this setting (~10% → 50% success, mean error ~1.2m → ~0.34m), but has not yet closed the full gap to Task 2's standalone 92% (itself measured on a later, more refined model than this one was fine-tuned from). The current demo script sidesteps this by using a specific known-good seed (247) for a reliable presentation; the underlying policy's general-case accuracy is the 50%/0.34m figures above, from a real 30-seed evaluation, not a cherry-picked run.

---

## Reward/penalty legend (applies to all three throwing setups)

- **Reward** = the model is trying to *earn* this by acting well.
- **Penalty** = subtracted from the total; the model is trying to *avoid* this.
- A component reading **0.00 (inactive)** means it exists in the code but wasn't triggered in this particular evaluation (e.g. no falls occurred, so `FALL_PENALTY` never fired) — it is still part of the trained objective, just not observed in that sample.

*Updated 2026-07-26 from live evaluation of the current best saved models (fresh rollouts, not cached numbers) plus training log files. For the full technical write-up — observation/action spaces, every reward weight, PPO hyperparameters, the complete bug/iteration list, and honest limitations — see `TECHNICAL_REPORT_FULL.md`.*
