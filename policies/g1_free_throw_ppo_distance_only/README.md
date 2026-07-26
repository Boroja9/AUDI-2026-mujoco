# G1 Free Throw — Distance Model

Best pure-distance throwing policy for the Unitree G1 humanoid. Trained to throw a ball as far as possible in a straight line, while staying on its feet — no target, no accuracy objective, just distance and stability.

## How to run it

```bash
source .venv/bin/activate
python scripts/play_ppo_pitcher.py --model policies/g1_free_throw_ppo_distance_only/best_model.zip
```

Opens a MuJoCo viewer window and repeats the throw every episode (deterministic policy, fixed seed).

## What it does

- **Environment:** `envs/g1_free_throw_env.py` (`G1FreeThrowEnv`), residual RL on top of a scripted baseline throw (RAISE → COCK → WHIP → RECOVER arm sequence).
- **Controlled DOFs:** 3 right-arm joints (shoulder pitch/roll, elbow) as a residual correction, plus full control over waist_yaw (compensates the sideways drift caused by the arm's swing reaction torque) and the release-timing decision (can release the ball any time after 0.4s into the motion).
- **Smoothing:** a low-pass filter on the residual action keeps the arm motion smooth (structural fix, not reward-learned) — no visible jitter.
- **Ratchet:** during the whip phase, the active joints can only move forward toward the throw pose, never back toward the cocked position — prevents flailing/re-winding mid-swing.
- **Reward:** rewards forward ball distance (capped at a useful max) and straight-line accuracy (penalizes sideways landing), with a real penalty (not just zero) for weak/backward throws — this specifically closes an exploit where the policy could release the ball too early to avoid the fall-risk of a real swing.

## Metrics (20-episode evaluation, deterministic policy, seeds 0–19)

Computed using this model's **own original reward formula** (distance + straightness, not the later dartboard-accuracy reward the codebase now defaults to — see note below).

| Metric | Value |
|---|---|
| **Episode reward** | mean=22.63, std=8.00 (min=9.02, max=31.37) |
| **Episode length** | 43.0 steps (~0.86s) |
| **Task error** (no fixed target — measured as throw distance / straightness instead) | distance: mean=2.11m, std=0.60m (min=0.87m, max=3.33m); lateral offset: mean\|y\|=0.25m, std=0.31m |
| **Fall count** | 0/20 (0%) |
| **Training wall time** | ~1390s (~23 min), 500k steps, ~365-390 fps |

There is no "success rate" for this model in the strict sense — it has no target to hit, only distance to maximize. The closest equivalents are throw distance and lateral straightness, both reported above.

### Key reward components (avg per episode, N=20)

| Component | Type | Avg/ep |
|---|---|---|
| Alive bonus | **Reward** | +2.15 |
| Distance bonus | **Reward** | +21.23 |
| Side penalty (lateral deviation from straight line) | **Penalty** | -0.75 |
| Short-throw penalty (backward or <0.1m throws) | **Penalty** | 0.00 (inactive — no throw was that bad) |
| Fall penalty | **Penalty** | 0.00 (inactive — robot never fell) |

The two "Reward" rows are what the policy is trying to maximize (survive + throw far); the three "Penalty" rows subtract from that whenever the throw goes sideways, comes up short/backward, or the robot falls — all three sat at (near-)zero in this evaluation, meaning the model isn't currently paying any of those costs.

*Note on reward numbers:* the codebase's environment file (`envs/g1_free_throw_env.py`) was later rewritten for the dartboard-accuracy task, so simply re-running this old model through the *current* file's reward function would score it against a target it was never trained to aim at, producing a misleading (much lower) number. The reward above was reconstructed from the original distance-reward formula this model actually trained under (`W_DISTANCE=12.0`, `W_SIDE=3.0`, `MAX_USEFUL_X=2.5`, `MIN_USEFUL_DIST=0.1`, `W_SHORT_PENALTY=15.0`), applied to fresh rollouts of the model.

## Where it fits in the project

This is the model used as the **warm-start baseline** for the dartboard/accuracy task (`policies/g1_dartboard_ppo`) — that model started from this one's weights and was then trained further to aim at a specific target instead of just maximizing distance. See `README_TRAINING_LOG.md` in the project root for the full iteration history (bugs found, fixes applied) that led to this model.
