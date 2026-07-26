# G1 Dartboard Throw — Target Accuracy Model

Throwing policy for the Unitree G1 humanoid trained to hit the **center of a fixed target** ("dartboard"), not just throw far. Warm-started from the pure-distance model (`g1_free_throw_ppo_distance_only`) and further trained to trade raw distance for precision.

## How to run it

```bash
source .venv/bin/activate
python scripts/play_ppo_pitcher.py --model policies/g1_dartboard_ppo/best_model.zip
```

Opens a MuJoCo viewer showing the tilted dartboard target (~15° from vertical, "almost upright," on a stand) about 2.3m in front of the robot. The ball physically sticks (via a dynamically-activated weld constraint) wherever it hits.

## Basic training details

| Setting | Value |
|---|---|
| Algorithm | PPO (`stable_baselines3`), `MlpPolicy` |
| Warm start | `policies/g1_free_throw_ppo_distance_only/best_model.zip` (not trained from scratch) |
| Training steps | 500,000 |
| Parallel envs | 4 |
| Learning rate | 3e-4 |
| n_steps / batch_size / n_epochs | 2048 / 256 / 10 |
| gamma / gae_lambda | 0.99 / 0.95 |
| ent_coef | 0.01 |
| Target position | (2.3, 0.0, 0.6) m — roughly chest height, where the ball's natural flight arc passes |
| Target size (bullseye radius) | 0.28 m |
| Target tilt | ~15° from vertical (nearly upright, "on a stand") |
| Script | `scripts/train_ppo_pitcher.py --init <distance_model> --output policies/g1_dartboard_ppo` |

## Metrics (20-episode evaluation, deterministic policy, seeds 0–19)

| Metric | Value |
|---|---|
| **Episode reward** | mean=26.20, std=13.53 (min=4.81, max=38.84) |
| **Episode length** | 46.5 steps (~0.93s) |
| **Success rate** (bullseye hit, within 0.28m of center) | **60%** (12/20) |
| **Task error** (landing_error — distance from ball to target center) | mean=0.44m, std=0.49m |
| **Fall count** | 0/20 (0%) |
| **Training wall time** | 1390s (~23.2 min), 500k steps, ~365 fps |

### Key reward components (avg per episode, N=20)

| Component | Type | Avg/ep |
|---|---|---|
| Alive bonus | **Reward** | +2.32 |
| Accuracy bonus (exponential, peaks at target center) | **Reward** | +9.32 |
| Bullseye bonus (flat, only within success radius) | **Reward** | +12.00 |
| Miss penalty (linear, scales with distance from center) | **Penalty** | -1.33 |
| Fall penalty | **Penalty** | 0.00 (inactive — robot never fell) |

The reward has two layers by design: a **linear miss penalty** gives a consistent "get closer" gradient even on a bad throw (far misses aren't just "no bonus," they cost something proportional to how far off they were), while the **exponential accuracy bonus** and flat **bullseye bonus** reward fine precision once the throw is already in the right neighborhood. This two-layer design was added after an earlier version (exponential-only) got stuck with ~1.87m average error, since a purely exponential reward gives almost the same near-zero signal for "very far" and "somewhat far" misses.

## Design notes

- **Controlled DOFs:** same as the distance model — 3 right-arm joints (residual correction on a scripted RAISE→COCK→WHIP→RECOVER baseline), waist_yaw (corrects sideways drift), and release timing (can release any time after 0.4s).
- **Sticking mechanic:** the target is a real, collidable, tilted disc. On ball-target contact, a MuJoCo weld constraint is activated at runtime with the ball's current relative pose, so it stays exactly where it hit rather than bouncing off or rolling away.
- **Visual:** the target looks like an actual dartboard — 5 concentric rings (black outer, white, red, white, gold bullseye).

## Where it fits in the project

Built on top of `g1_free_throw_ppo_distance_only` (see that model's own README). Full iteration history — including two physics bugs found while building the sticking mechanic (`eq_data` indexing, and a `prev_action`/observation-shape mismatch after adding waist control) — is in `README_TRAINING_LOG.md` at the project root.
