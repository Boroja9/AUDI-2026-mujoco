# G1 Robot Training Log — Full Session Report

This document covers the full training session for the Unitree G1 humanoid robot: free throwing, dartboard-accuracy throwing, and walking. It records every iteration, the bugs found along the way, the fixes applied, and final metrics for each task.

All environments live in `envs/`, training scripts in `scripts/`, trained models in `policies/`. Only the current/best checkpoint for each task lives at the top level of `policies/` — every superseded iteration named below (e.g. `g1_free_throw_ppo_no_recovery`, `g1_free_throw_ppo_jittery`, `g1_walk_ppo_v2_higherlift`) has been moved to `policies/archive/<name>/` to keep the top level readable; the folder names themselves are unchanged. Old env-file snapshots from before each risky edit likewise live under `envs/backups/`.

---

## Task 1: Free Throw (maximize distance)

**Environment:** `envs/g1_free_throw_env.py` (`G1FreeThrowEnv`), built on top of `envs/g1_fixed_body_throw_env.py`.
**Training script:** `scripts/train_ppo_pitcher.py` (PPO, 500k steps per run, 4 parallel envs).
**Approach:** Residual RL — a scripted baseline throw (RAISE → COCK → WHIP → RECOVER arm poses) provides the "skeleton" motion; the RL policy learns a small correction on top, plus full control over waist rotation and release timing.

### Iteration history

| # | Model folder | Change made | Best reward | Issue found |
|---|---|---|---|---|
| 1 | `g1_free_throw_ppo_no_recovery` | Initial fix of pre-existing bugs (syntax error, playback reset bug, scene XML ball-height mismatch) | 11.12 | Arm stayed frozen in throw pose after release |
| 2 | `g1_free_throw_ppo_jittery` | Added RECOVER phase (arm returns to neutral after release) | 12.72 | Arm shook/jittered visibly |
| 3 | `g1_free_throw_ppo_action_rate_penalty` | Added reward penalty for large frame-to-frame action changes | 9.37 | Penalty fought the task itself — weaker throws, jitter not fixed (reward-shaping unreliable) |
| 4 | `g1_free_throw_ppo_no_waist` | Replaced reward penalty with a **low-pass filter** on the residual action (structural fix, not learned) | 10.93 | Jitter fixed. But ball consistently curved sideways (systematic bias, same direction every time) — diagnosed as unbalanced reaction torque from the arm swing, with no DOF to correct it |
| 5 | `g1_free_throw_ppo_waist_no_release_control` | Added **waist_yaw** as a controllable joint so RL can counter the sideways drift | 13.97 | Confirmed waist has real authority to fix sideways drift (verified via controlled test: waist=+1 flips the sign of the sideways error) |
| 6 | *(crashed, no dir)* | Gave RL control over release timing (min hold 0.4s) | — | **Bug:** base class `prev_action` sizing didn't account for the new extra actuator → observation shape mismatch, crashed on `vec_env` reset. Fixed in `g1_fixed_body_throw_env.py`. |
| 7 | `g1_free_throw_ppo_ratchet_bug` | Added a "ratchet": during the whip window, joints can only move forward toward the throw pose, never back toward cock (prevents flailing) | 9.30 | **Bug:** ratchet's reference point was a fixed idealized pose, not the actual arm position when the whip phase began → visible snap/stutter at the phase transition |
| 8 | `g1_free_throw_ppo_early_release_exploit` | Fixed ratchet to snapshot the *actual* position on first entry to the whip window (no more snap) | 9.14 | **Reward exploit:** RL learned to release the ball at the earliest allowed instant (0.4s, before any real throwing motion), since a weak/backward toss risked nothing (no fall) while a real throw risked the fall penalty. Ball frequently landed *behind* the robot. |
| 9 | `g1_free_throw_ppo_best_13_84` | Raised minimum release time to 1.00s (must finish cocking first); also increased distance reward | 13.97 | Better, but user requested a middle-ground release time |
| 10 | `g1_free_throw_ppo_distance_only` | `MIN_RELEASE_TIME=0.4` restored per user request, but paired with an explicit backward/short-throw penalty (`W_SHORT_PENALTY`) instead of a hard time gate | **31.49** | This is the best pure-distance model. No more backward throws; consistent, far, stable throws. |

### Final metrics — `g1_free_throw_ppo_distance_only` (best distance model)
*(Predates the dartboard task — this model maximizes distance, not target accuracy.)*

- **Best eval reward:** 31.49 @ 440,000 steps (last eval: 30.95)
- **Training wall time:** ~1390s (~23 min) per 500k-step run (typical across all runs in this family, ~365-390 fps)

---

## Task 2: Dartboard (hit a specific target, not just "far")

**Change of objective:** replaced "maximize forward distance" with "hit the center of a target." Target is a physical, collidable, tilted disc (~35° lean, later widened and made more upright per request) positioned in the ball's flight path; a MuJoCo weld constraint activates on contact so the ball **sticks** where it hits (see "Bugs found" below for the two physics bugs this required fixing).

**Model:** `policies/g1_dartboard_ppo`, warm-started from `g1_free_throw_ppo_distance_only` (not trained from scratch).

### Reward iterations
1. **First attempt:** pure exponential reward `W_ACCURACY * exp(-K*error)`. Result: after ~75% of training, mean landing error was still ~1.87m (target success radius 0.28m) — essentially never hit. Diagnosis: the exponential reward gives almost the same (near-zero) signal for "very far" and "somewhat far" misses, so the policy gets no gradient telling it which direction to improve from a bad throw.
2. **Fix:** added a **linear miss penalty** (`W_MISS_PENALTY * error`) on top of the exponential bonus — gives a consistent "get closer" gradient at any distance, while the exponential term still rewards fine precision once close.

### Final metrics — `g1_dartboard_ppo`
*(20-episode evaluation, deterministic policy, seeds 0-19)*

| Metric | Value |
|---|---|
| Episode reward | mean=26.20, std=13.53 (min=4.81, max=38.84) |
| Episode length | 46.5 steps (~0.93s) |
| **Success rate** (bullseye hit, radius 0.28m) | **60%** (12/20) |
| **Task error** (landing_error) | mean=0.44m, std=0.49m |
| Fall count | 0/20 (0%) |
| Training wall time | 1390s (~23.2 min), 500k steps, ~365 fps |
| Best eval reward (training log) | 31.56 @ 480,000 steps |

**Key reward components** (avg per episode, N=20):

| Component | Avg/ep |
|---|---|
| Alive bonus | +2.32 |
| Accuracy bonus (exponential, peaks at center) | +9.32 |
| Bullseye bonus (flat, only if within success radius) | +12.00 |
| Miss penalty (linear, always active) | -1.33 |
| Fall penalty | 0.00 |

---

## Bugs found and fixed (cross-cutting, affect the base environment)

1. **`envs/g1_free_throw_env.py` syntax error** — pre-existing indentation bug made the whole env unimportable. Fixed at the very start of the session.
2. **`scripts/play_ppo_pitcher.py` reset bug** — called `env.reset()` at the top of every loop iteration, destroying every episode after one step. Fixed.
3. **`assets/unitree_g1/scene_throw.xml` ball-height mismatch** — an uncommitted local edit had moved the ball's spawn height by +1m relative to the weld target, causing a physics shock on every reset. Reverted to the committed value.
4. **`eq_data` indexing bug** (dartboard stick mechanic) — MuJoCo's weld-constraint data layout is `[anchor(3), relpos(3), relquat(4), torquescale(1)]` (11 floats), not `relpos(3)+relquat(4)` at offset 0 as first assumed. Writing to the wrong offset caused the "stuck" ball to drift/fly off (up to 1m) instead of staying put. Fixed by writing to indices `[3:10]`.
5. **Base-class `prev_action` sizing bug** — after adding the waist as an "extra actuator", the base class's `prev_action` array and the declared `observation_space` disagreed by one element, crashing `SubprocVecEnv`/`DummyVecEnv` on reset. Fixed by having the subclass probe the true observation length instead of computing it analytically.

---

## Task 3: Walking (0.5m forward, stay upright, no leg-crossing, settle at the end)

**Environment:** `envs/g1_walk_env.py` (`G1WalkEnv`) — a separate, standalone task, not combined with throwing. Full leg control (12 joints), scripted approach was attempted first and abandoned (see below).

### Why RL instead of a scripted gait
A hand-scripted 2-step gait (lift foot, shift weight, swing, plant) was attempted first, iterating on sign conventions and weight-shift magnitude via direct physics probing. It reliably fell over as soon as true single-leg support began — open-loop scripted trajectories cannot hold balance without real-time feedback control, which is a much larger undertaking (attempted once with a simple ankle PD feedback loop; made things worse due to gain/sign tuning being non-trivial). Given the difficulty, the task was handed to RL instead, which can discover its own balance strategy through trial and reward.

### Reward design
- `W_FORWARD` on **net displacement per step** (not instantaneous velocity — see exploit below)
- `W_UPRIGHT`, `W_HEIGHT`, `W_SIDE`: stability terms
- `W_HIP_LATERAL`: penalizes extreme hip_roll/hip_yaw deviation (prevents leg-crossing)
- `W_ASYMMETRY`: penalizes left/right leg deviating very differently from each other (encourages a natural, balanced gait)
- `SUCCESS_BONUS`, `FALL_PENALTY`
- **Settle check:** reaching 0.5m only counts as success if the robot then holds a low tilt and low speed for `SETTLE_STEPS` (15) consecutive steps — prevents "reached the distance mid-fall" from counting as success
- Low-pass filter on the leg action (same smoothing trick used for the arm)

### Bug found: instantaneous-velocity reward exploit
First full training run (1.5M steps) reached eval rewards of 400+ and *looked* like strong progress, but a direct evaluation showed **100% fall rate** and **0% success**. Root cause: `W_FORWARD` rewarded the pelvis's instantaneous physical velocity, not actual progress. The policy learned to produce a violent jerk/spin (reported by the user as "spinning 360° in place") that reads an enormous instantaneous velocity (up to 8.77 m/s) while the pelvis was actually net moving *backward*, then falls. Confirmed via step-by-step logging:

```
step=20  fwd_vel=7.05  fwd=-0.09  tilt=0.41
step=27  fwd_vel=8.77  fwd=-0.23  tilt=0.66
step=28  fwd_vel=8.64  fwd=-0.25  tilt=0.72  → FELL
```

**Fix:** replaced `W_FORWARD * instantaneous_velocity` with `W_FORWARD * (fwd - prev_fwd) / control_dt` — real net displacement per control step, which cannot be gamed by a brief spin/jerk since it reflects actual position change, not a momentary velocity reading.

### Status: training in progress (restarted after the fix)
*(Metrics below are from the first ~60k/1.5M steps post-fix — will change significantly as training progresses.)*

| Metric | Value (early, ~4% into training) |
|---|---|
| Episode reward | ~123-160 (declining from an initial spike, normal early PPO noise) |
| Training wall time so far | ~130-170s at ~420-430 fps |
| Success rate / Fall count | not yet re-evaluated post-fix |

---

## Summary of trained models on disk (`policies/`)

| Folder | Task | Best reward | Notes |
|---|---|---|---|
| `g1_free_throw_ppo_no_recovery` | distance | 11.12 | early iteration |
| `g1_free_throw_ppo_jittery` | distance | 12.72 | jittery arm |
| `g1_free_throw_ppo_action_rate_penalty` | distance | 9.37 | reward-shaping backfired |
| `g1_free_throw_ppo_no_waist` | distance | 10.93 | sideways bias, no waist correction |
| `g1_free_throw_ppo_waist_no_release_control` | distance | 13.97 | waist added |
| `g1_free_throw_ppo_ratchet_bug` | distance | 9.30 | ratchet snap bug |
| `g1_free_throw_ppo_early_release_exploit` | distance | 9.14 | early-release exploit |
| `g1_free_throw_ppo_best_13_84` | distance | 13.97 | 1.0s min release |
| **`g1_free_throw_ppo_distance_only`** | distance | **31.49** | **best distance model** |
| **`g1_dartboard_ppo`** | accuracy (dartboard) | 31.56 (60% bullseye rate) | **best accuracy model**, warm-started from the above |
| `g1_walk_ppo_v2` | walk 0.5m | *(training, post-exploit-fix)* | in progress |

*Report generated 2026-07-25 from live evaluation of the saved models plus training log files. Numbers for `g1_walk_ppo_v2` will be updated once training completes.*
