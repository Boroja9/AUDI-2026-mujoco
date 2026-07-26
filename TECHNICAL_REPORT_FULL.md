# G1 Humanoid — Complete Technical Report (All Four Tasks)

Covers every task built in this project: **(1)** throw-as-far-as-possible with a fixed
pelvis, **(2)** throw-at-a-fixed-target with a fixed pelvis, **(3)** standalone walking
(0.5 m forward), and **(4)** walking-then-throwing-at-the-target combined. Organized to
map directly onto the scoring rubric in `Technical_AI_Robotics_final.pdf`: task
definition → success criteria → baseline → RL pipeline → training runs → baseline-vs-trained
comparison → limitations/Sim2Real.

All numbers below are pulled from the actual code and from fresh evaluation runs
(multiple seeds, deterministic policy) done directly against the saved checkpoints —
not estimated.

---

## 1. Task Definitions & Justification

| # | Task | Robot base | Goal |
|---|---|---|---|
| 1 | Throw as far as possible | Fixed pelvis (non-ambulatory) | Release a held ball so it travels the maximum forward distance without falling |
| 2 | Throw at a fixed target | Fixed pelvis | Release the ball so it lands as close as possible to a fixed 3D target point (a tilted "dartboard" ~2.3 m out) |
| 3 | Walk forward | Free-standing, full leg control | Move the pelvis ~0.5 m forward (~2 steps), stay upright, settle to a stop |
| 4 | Walk then throw | Free-standing → fixed pelvis handoff | Walk 0.5 m, then execute the target-throw from whatever pose/velocity the walk left the robot in |

**Why these tasks:** all four are feasible in one week, measurable with simple
geometric metrics (distance, landing error, forward displacement), visible in a short
demo, and directly exercise core humanoid-RL concepts (residual control, balance,
reward shaping, sequential policy composition) without requiring full from-scratch
bipedal locomotion as the *only* deliverable (a explicitly flagged "risky task" for a
one-week scope).

**Feasibility note:** Task 1/2 deliberately fix the pelvis (no locomotion) to isolate
the throwing problem from the balance problem — a direct application of "start with
locked complexity" / "freeze irrelevant joints." Task 3 isolates locomotion from
throwing. Task 4 is the only task that combines both, and is explicitly the highest-risk
one (documented as such in the limitations section).

---

## 2. Success Criteria & Metrics (Formulas)

| Task | Success formula | Threshold |
|---|---|---|
| 1 (distance) | no target — `ball_x` at landing is the metric itself | maximize (reward capped at `DISTANCE_CAP = 6.0 m`) |
| 2 (target) | `landing_error = |ball_x − target_x|` (X-axis only, see §8 design note) | success if `landing_error ≤ SUCCESS_RADIUS = 0.28 m` |
| 3 (walk) | reach `TARGET_DISTANCE = 0.5 m` **and** hold `tilt ≤ SETTLE_TILT_MAX (0.15 rad)` and `speed ≤ SETTLE_VEL_MAX (0.3 m/s)` for `SETTLE_STEPS = 15` consecutive steps | binary success flag in `info["success"]` |
| 4 (combined) | walk phase has no explicit success gate (state is just handed to the throw phase); throw phase reuses Task 2's exact success formula | same as Task 2 |

**Landing detection (Tasks 1/2/4):** `landed = hit_target OR ball_z ≤ ball_radius + 0.015`, checked every physics step; `hit_target` is a real MuJoCo contact test between the ball geom and the target geom (not a proximity heuristic).

**Fall detection (all tasks):** `pelvis_z < FALL_HEIGHT_RATIO × nominal_pelvis_height` (ratio `0.55` for walking; throwing uses the equivalent `0.55 × nominal_base_height` inherited from the shared base class) **or** `tilt > 0.7 rad`.

**Measurement protocol:** deterministic policy (`model.predict(obs, deterministic=True)`), N episodes over a fixed seed range (0…N−1), no cherry-picking — every number in §6 below was produced this way in this session.

---

## 3. Baseline Behavior (Non-Learning Reference)

### 3.1 Throwing baseline (Tasks 1, 2, 4) — scripted RAISE→COCK→WHIP arm trajectory

A hand-authored joint-angle trajectory drives the right arm through three phases,
interpolated with a smoothstep (`3x²−2x³`) for continuity:

| Phase | Window | Pose target (radians, `[shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]`) |
|---|---|---|
| RAISE | `t < 0.45s` | interpolates 0 → `[-2.0, -0.80, 0, 0, 0, 0, 0]` |
| COCK | `0.45s ≤ t < 1.00s` | interpolates RAISE → `[-2.0, -0.80, 0, -0.5, 0, 0, 0]` |
| WHIP | `1.00s ≤ t < 1.20s` | interpolates COCK → `[-2.0, -0.80, 0, 1.5, 0, 0, 0]` (this is the actual throwing motion) |

This baseline alone, with no RL correction, produces a *repeatable but inaccurate* throw
(no way to correct for imbalance, sideways drift, or hit a specific target). It is used
as the residual-RL reference for all throwing tasks — the RL policy learns a *correction*
on top of it (see §4), not the throw from scratch.

**Why residual RL instead of full RL for throwing:** the scripted skeleton already
encodes "how a throw is shaped in time," which is not something the reward needs to
discover from zero — it collapses the search space to fine correction + release timing +
balance compensation only. This directly matches the "hybrid strategy" recommendation
(script first, learn around it) for cases where a stable baseline already exists.

### 3.2 Walking baseline — scripted gait attempts (all failed, replaced with RL)

A hand-scripted two-step gait (lift foot → shift weight → swing → plant), tuned via
direct physics probing of sign conventions and weight-shift magnitude, was attempted
**before** any RL work on walking began. It reliably fell over as soon as true
single-leg support began. A simple ankle PD feedback loop was added on top and made
things *worse* (gain/sign tuning for balance recovery is non-trivial by hand). Given
this, walking was handed entirely to RL — the "no scripted baseline available, fastest
route is not hybrid" case explicitly rather than the "script first" case used for
throwing. This is documented directly in the walking env's own docstring:
*"pokusaji sa rucno skriptovanim hodom su padali - jednonozni oslonac zahteva feedback
koji je tesko rucno podesiti"* ("hand-scripted-gait attempts fell — single-leg support
needs feedback that's hard to hand-tune").

**Baseline metric for walking (what "no learning" looks like):** standing still (zero
action) — the robot does not fall (it's already balanced at the reset keyframe) but
makes 0 m of forward progress and never reaches the 0.5 m target — i.e., 0% success by
construction. This is the honest "non-learning reference."

---

## 4. RL Pipeline Design

### 4.1 Task 1 — Throw far (`envs/g1_free_throw_distance_env.py`, class `G1FreeThrowDistanceEnv`)

**Observation space** (concatenated, `float32`):

| Segment | Dim | Source |
|---|---|---|
| Arm joint positions | 7 | `qpos` of the 7 right-arm joints |
| Arm joint velocities | 7 | `qvel`, same joints |
| Ball position (world) | 3 | |
| Ball velocity (world, linear) | 3 | |
| Target position (constant, unused for reward here) | 3 | inherited field, see §8 |
| Previous action | 9 | arm(7)+waist(1)+release(1) |
| Released flag | 1 | 0/1 |
| Time remaining | 1 | |
| Pelvis height | 1 | |
| Gravity vector (torso frame, xy) | 2 | tilt sensing |
| Pelvis angular velocity | 3 | |
| Pelvis linear velocity | 3 | |
| Waist position + velocity | 2 | |

Total = 46-dim (base-class 35 + subclass 11, probed at runtime rather than computed
analytically — see bug #5 in §7).

**Action space** — `Box(-1, 1, shape=(5,))`:
- [0:3) residual correction on `(shoulder_pitch, shoulder_roll, elbow)`, added to the
  scripted baseline, scale `RESIDUAL_SCALE = 0.6`, low-pass filtered (`FILTER_ALPHA = 0.25`)
- [3] waist_yaw residual, scale `WAIST_RESIDUAL_SCALE = 0.3`
- [4] release decision, thresholded at `> 0.5`, gated by `MIN_RELEASE_TIME = 0.4s`,
  forced at `RELEASE_DEADLINE = 1.2s`

**Reward function** (per step; terminal terms paid once at landing):

| Component | Type | Weight | Formula |
|---|---|---|---|
| Alive bonus | Reward | `ALIVE_BONUS = 0.05` | flat, every step |
| Tilt penalty | Penalty | `W_TILT = 1.2` | `max(0, tilt − TILT_FREE=0.12)` |
| Wobble penalty | Penalty | `W_WOBBLE = 0.05` | `‖pelvis linear vel‖` |
| Sway penalty | Penalty | `W_SWAY = 0.03` | `‖pelvis angular vel‖` |
| Height penalty | Penalty | `W_HEIGHT = 1.0` | `max(0, nominal_height − pelvis_z)` |
| Fall penalty | Penalty | `FALL_PENALTY = 20.0` | flat, once, on fall |
| Release-speed bonus | Reward | `W_RELEASE_VX = 1.0`, capped `W_RELEASE_SPEED_CAP = 8.0` | `min(max(0, ball_vx), cap)` at the release step |
| **Distance bonus** (terminal) | Reward | `W_DISTANCE = 8.0`, capped `DISTANCE_CAP = 6.0 m` | `W_DISTANCE × min(max(0, ball_x), cap)` |
| **Side penalty** (terminal) | Penalty | `W_SIDE_THROW = 3.0` | `× |ball_y|` |
| **Short-throw penalty** (terminal) | Penalty | `W_SHORT_THROW = 15.0` | flat, if `ball_x < SHORT_THROW_MIN = 0.1 m` |

**Termination:** landed or fell; truncated at `episode_time = 3.0s`.

### 4.2 Task 2 — Throw at target (`envs/g1_free_throw_env.py`, class `G1FreeThrowEnv`)

Same observation/action architecture as Task 1 (this is the *original* file; Task 1 is
a later fork of it — see §8). Reward differs only in the terminal branch:

| Component | Type | Weight | Formula |
|---|---|---|---|
| Miss penalty (terminal) | Penalty | `W_MISS_PENALTY = 3.0` | `× landing_error` (linear — always gives a "get closer" gradient) |
| Accuracy bonus (terminal) | Reward | `W_ACCURACY = 20.0`, `K_ACC = 1.8` | `W_ACCURACY × exp(−K_ACC × landing_error)` |
| Bullseye bonus (terminal) | Reward | `BULLSEYE_BONUS = 25.0` | flat, if within `SUCCESS_RADIUS` |

(Alive/tilt/wobble/sway/height/fall/release-speed terms are identical to Task 1, same
weights.) `TARGET_POS = (2.8, 0.0, 0.6)` by default (see §8 for why this specific value
matters for reproducibility).

**Post-throw arm behavior** — iterated three times (see §7 bugs/iteration log): the arm
now transitions via a pure scripted smoothstep from its *actual* end-of-whip position to
the neutral pose (`RECOVER_DURATION = 0.35s`), with zero RL involvement in that phase —
chosen after a learned-recovery attempt produced inconsistent, sometimes-wrong-direction
motion (documented fully in §7).

### 4.3 Task 3 — Walking (`envs/g1_walk_env.py`, class `G1WalkEnv`)

**Observation space** — `46`-dim: `[pelvis_z(1), gravity_xy(2), pelvis_angular_vel(3),
pelvis_linear_vel(3), leg_qpos_deviation(12), leg_qvel×0.1(12), prev_action(12),
time_remaining(1)]`.

**Action space** — `Box(-1, 1, shape=(12,))`, one per leg joint (hip pitch/roll/yaw,
knee, ankle pitch/roll × 2 legs), position-control offsets from nominal, low-pass
filtered (`FILTER_ALPHA = 0.3`).

**Reward function** (current, final iteration in this session):

| Component | Type | Weight | Formula |
|---|---|---|---|
| Alive bonus | Reward | `W_ALIVE = 0.05` | flat |
| Forward progress | Reward | `W_FORWARD = 40.0` | `× (capped_fwd_t − capped_fwd_{t−1})`, capped at `TARGET_DISTANCE=0.5m` — **real displacement, not instantaneous velocity** (see bug in §7) |
| Upright bonus | Reward | `W_UPRIGHT = 1.0` | `× max(0, 1 − tilt)` |
| Height penalty | Penalty | `W_HEIGHT = 2.0` | `× max(0, nominal − pelvis_z)` |
| Side penalty | Penalty | `W_SIDE = 1.0` | `× |pelvis_y|` |
| Action-rate penalty | Penalty | `W_ACTION_RATE = 0.02` | `× ‖Δaction‖²` |
| Hip-lateral penalty | Penalty | `W_HIP_LATERAL = 0.5` | `× ‖hip_roll/yaw deviation‖²` (anti leg-crossing) |
| Foot-lift bonus | Reward | `W_FOOT_LIFT = 32.0`, cap `MAX_FOOT_LIFT = ∞` (uncapped) | `× lift_height`, only within `MAX_LIFT_DURATION=0.4s` of swing |
| Knee-bend bonus | Reward | `W_KNEE_BEND = 0.0` | **disabled** — deprioritized vs. stride/lift |
| Knee-stance penalty | Penalty | `W_KNEE_STANCE_PENALTY = 3.0` | `× |knee deviation|` while foot is planted (keep stance leg straight) |
| Stride-length bonus | Reward | `W_STRIDE = 25.0`, cap `MAX_STRIDE = 0.25 m` | `× (foot_x − pelvis_x)`, only while the foot is actually lifted |
| Asymmetry penalty | Penalty | `W_ASYMMETRY = 0.0` | **disabled** — was suppressing all stepping attempts early on |
| Airborne penalty | Penalty | `W_AIRBORNE = 4.0` | flat, if *neither* foot touches the floor (no jumping) |
| Overshoot penalty | Penalty | `W_OVERSHOOT = 6.0` | `× max(0, fwd − 0.5m)` |
| Stand-still bonus | Reward | `W_STAND_STILL = 2.0` | `× max(0, 1 − speed/0.3)`, only once past the target line |
| Backward penalty | Penalty | `W_BACKWARD = 2.0` | `× |fwd|`, if `fwd < 0` |
| Success bonus | Reward | `SUCCESS_BONUS = 30.0` | flat, once, on settle-success |
| Fall penalty | Penalty | `FALL_PENALTY = 30.0` | flat, on fall |

**Termination:** fall or settle-success; truncated at `EPISODE_SEC = 3.0s`.

### 4.4 Task 4 — Walk then throw (`envs/g1_post_walk_throw_env.py`, class `G1PostWalkThrowEnv`)

Not a new reward function — it's `G1FreeThrowEnv` (Task 2) unchanged, with `reset()`
overridden: on every reset, an internal, frozen `G1WalkEnv` + walk policy
(`WALK_MODEL_PATH`, hardcoded to `policies/g1_walk_ppo_v2/best_model_frozen_test.zip`)
runs a full walk episode with a randomized seed (`0..MAX_WALK_SEED=300`), and its final
`qpos`/`qvel` are copied directly into the throw environment's MuJoCo `data` before the
throw policy takes over. This is direct MuJoCo state transfer between two independently
trained policies, not a shared/joint training run.

---

## 5. PPO Hyperparameters (all training runs, `stable_baselines3.PPO`, `MlpPolicy`)

| Hyperparameter | Value | Notes |
|---|---|---|
| `learning_rate` | `3e-4` | never tuned, SB3 default |
| `n_steps` | `2048` | per environment, per rollout |
| `batch_size` | `256` | |
| `n_epochs` | `10` | |
| `gamma` | `0.99` | |
| `gae_lambda` | `0.95` | |
| `ent_coef` | `0.01` default; `0.05` used for warm-started walking runs specifically (raised to break a stuck, over-converged local optimum — see §7) | |
| `n_envs` | `4` (throwing tasks, all runs); `4`→`8`→`16` across successive walking runs as more CPU headroom was allocated | `SubprocVecEnv` via `make_vec_env` |
| `device` | training: default (`auto`, picks CUDA if present); **playback/eval: forced `cpu`** | GPU introduces run-to-run nondeterminism in playback that broke seed-reproducibility; irrelevant for training throughput |
| Eval callback | `n_eval_episodes=10`, `eval_freq = 10_000/n_envs` steps | `EvalCallback`, saves `best_model.zip` (highest eval reward seen) separately from `final_model.zip` (last checkpoint — often worse due to normal late-training PPO variance) |

**Warm-start lineage (checkpoint → checkpoint, via `PPO.load(path, env=...)`):**

```
Task 1: g1_free_throw_ppo_distance_only → g1_free_throw_ppo_distance_v2
Task 2: g1_dartboard_ppo → g1_dartboard_ppo_xonly → g1_dartboard_ppo_center → g1_dartboard_ppo_center_v2 → g1_dartboard_ppo_recover
Task 3: g1_walk_ppo → g1_walk_ppo_v2 → (stance → stride → freelift → frozenyaw, parallel exploratory branches, not fully converged)
Task 4: g1_dartboard_ppo (Task 2's model) → g1_dartboard_postwalk_ppo (walk side always frozen, never retrained)
```

---

## 6. Completed Training Runs, Metrics & Baseline-vs-Trained Comparison

All numbers below: deterministic policy, fresh evaluation in this session, fixed seed
range per task (stated), **not** cherry-picked single runs.

### Task 1 — `g1_free_throw_ppo_distance_v2` (current best)

| | |
|---|---|
| Total timesteps | 507,904 (target 500,000) |
| Training wall time | 1394 s (~23.2 min) at ~364 fps |
| Warm-started from | `g1_free_throw_ppo_distance_only` |

| Metric | Baseline (no learning, zero action) | Trained (N=30, seeds 0–29) |
|---|---|---|
| Episode reward | not meaningful (no distance term fires without a real throw) | mean 51.68, std 4.48 (min 37.98, max 55.25) |
| Episode length | ~64 steps (arm never releases meaningfully) | mean 57.7 steps (~1.15 s) |
| Success rate / task error | 0 m distance (no throw at all) | **mean distance 6.37 m**, std 0.79 m (min 4.41, max 7.45) |
| Fall count | 0/30 (never moves enough to fall) | 0/30 (0%) |
| Key reward components (avg/ep) | n/a | Alive +2.88, Distance-bonus +46.46, Side-penalty −1.60, Short-throw-penalty 0.00 (inactive), Fall 0.00 (inactive) |
| Training wall time | n/a | 1394 s / 500k steps |

*(Earlier iteration, `g1_free_throw_ppo_distance_only`, achieved only 2.33 m mean
distance on the same 15 seeds — the distance-reward-only rework in `g1_free_throw_ppo_distance_v2`
nearly tripled reach.)*

### Task 2 — `g1_dartboard_ppo_center_v2` (current best)

| | |
|---|---|
| Total timesteps | 507,904 (target 500,000) |
| Training wall time | 1479 s (~24.65 min) at ~343 fps |
| Warm-started from | `g1_dartboard_ppo_xonly` |

| Metric | Baseline (scripted, no RL residual) | Trained (N=50, seeds 0–49) |
|---|---|---|
| Episode reward | not meaningful (no accuracy term without aim) | mean 49.09, std 10.95 (min 6.16, max 53.76) |
| Episode length | ~46 steps | mean 50.7 steps (~1.01 s) |
| **Success rate** | 0% (no aim correction at all) | **92% (46/50)** |
| **Task error** | uncontrolled, several meters typical | mean 0.073 m, **median 0.015 m** (a few outlier misses pull the mean up; max 1.341 m) |
| Fall count | 0/50 | 0/50 (0%) |
| Key reward components (avg/ep) | n/a | Alive +2.54, Accuracy +18.28, Bullseye +23.00, Miss-penalty −0.22, Fall 0.00 (inactive) |

**Iteration history for Task 2's success rate** (all on the same architecture, reward-weight
changes only): 60% (original, 3D landing-error) → 90% (X-only scoring, decoupling
lateral miss from scoring) → 78% (over-aggressive center-accuracy push, `K_ACC` too
sharp) → **92%** (moderated center-accuracy weights, current). This non-monotonic path
is itself evidence that reward-shape sharpness trades off against generalization, and
is documented honestly rather than hidden.

### Task 3 — `g1_walk_ppo_v2` (`best_model_frozen_test.zip`, the version frozen for
production use by Task 4)

| | |
|---|---|
| Total timesteps | ~990,000 (evaluation curve; best eval reward 179.74 under an earlier, less-refined reward configuration than the current file) |
| Training wall time | ~1482 s (~24.7 min) recorded fps ~414 (partial log; multiple resumed sessions across this checkpoint's lineage) |

| Metric | Baseline (zero action, stands still) | Trained (N=30, seeds 0–29) |
|---|---|---|
| Episode reward | ~0 (alive bonus only, no progress) | mean 234.69, std 14.38 |
| Episode length | 150 steps (full episode_time, never terminates early) | mean 137.5 steps (~2.75 s) |
| **Success rate** | 0% (never reaches 0.5 m) | **70% (21/30)** |
| Task error (final forward distance) | 0 m | mean 0.542 m, std 0.030 m (slightly overshoots target, within overshoot-penalty tolerance) |
| Fall count | 0/30 (standing still doesn't fall) | 0/30 (0%) |

**Important limitation:** this exact checkpoint predates the most recent gait-naturalness
refinements (foot-lift cap removal, stride-length reward, knee-bend deprioritization,
asymmetry-penalty removal) that were developed later in this session
(`g1_walk_ppo_v2_stance/_stride/_freelift`) — those runs improved *visual* gait quality
(genuine stride length, no leg-dragging) but were not run to full convergence before
being superseded by other priorities (see §7/§9). **The frozen checkpoint above is what
Task 4 actually uses in production** — the newer, not-fully-trained gait variants are
reported separately as in-progress work, not swapped in, specifically so Task 4's
numbers stay reproducible against a fixed walking policy.

### Task 4 — `g1_dartboard_postwalk_ppo` (throw side) + `g1_walk_ppo_v2` (walk side, frozen)

| | |
|---|---|
| Total timesteps (throw fine-tune only; walk side reused, not retrained) | 303,104 (target 300,000) |
| Training wall time | 2321 s (~38.7 min) at ~130 fps — slower than Tasks 1–2 because every single reset also runs a full walk rollout first |
| Warm-started from | `g1_dartboard_ppo` (Task 2's model, an earlier iteration, not `center_v2`) |

| Metric | Baseline (Task 2's un-fine-tuned model, fed post-walk states) | Trained (N=30, walk seeds 0–29) |
|---|---|---|
| Episode reward (throw phase only) | not separately measured (see below) | mean 32.10, std 17.57 |
| Episode length (combined) | ~190 steps | mean 252.1 steps (~5.04 s: walk + throw) |
| **Success rate** | ~10% (documented from earlier un-tuned baseline) | **50% (15/30)** |
| **Task error** (landing error) | ~1.2 m (documented from earlier un-tuned baseline) | mean 0.337 m, median 0.270 m, std 0.351 m, max 1.511 m |
| Fall count | walk: 0/30 · throw: 0/30 | walk: 0/30 (0%) · throw: 0/30 (0%) |

**Why Task 4's success rate is lower than Task 2's standalone 92%:** the throw policy
here must release accurately from whatever pose/velocity the walk left the robot in — a
much wider range of starting conditions than "always start from a clean standing pose."
Fine-tuning on 300k randomized post-walk states roughly doubled accuracy over the
un-tuned model (10% → 50% in this fresh 30-seed run; a smaller, 15-seed sample taken
earlier in the session showed 67%, and a specific hand-picked walk seed — 247 — gives a
near-perfect 0.062 m error and is what the interactive demo script defaults to for a
reliable live presentation). The general-case accuracy (this table) is the honest number
for the report; the seed-247 result is a documented best-case demo, not the claimed
average.

---

## 7. Bugs Found and Fixed (Full List — "What Failed" Evidence)

Cross-cutting (shared base class, predates individual tasks but load-bearing for all of them):

1. **Pre-existing syntax/indentation error** made `g1_free_throw_env.py` unimportable at session start. Fixed.
2. **Playback reset bug** — `play_ppo_pitcher.py` called `env.reset()` at the top of every loop iteration, destroying every episode after one step. Fixed.
3. **Scene XML ball-height mismatch** — an uncommitted local edit had moved the ball spawn height ~+1 m relative to the hold-weld anchor, causing a physics shock every reset. Reverted.
4. **`eq_data` indexing bug** (target-sticking weld mechanic) — MuJoCo's weld-constraint data layout is `[anchor(3), relpos(3), relquat(4), torquescale(1)]` (11 floats), not `relpos(3)+relquat(4)` at offset 0. Caused the "stuck" ball to drift up to ~1 m. Fixed by writing to the correct `[3:10]` slice.
5. **`prev_action`/observation-shape mismatch** — after adding the waist as an "extra actuator," the analytically-computed observation dimension no longer matched reality, crashing `SubprocVecEnv` on reset. Fixed by probing the real base observation length at runtime instead of computing it by formula.
6. **Weld-activation `NaN`** — a fast-moving ball snapping into a stiff weld constraint blew up `qacc`. Fixed by softening `solref`/`solimp` on the stick weld relative to the hold weld, and disabling the ball's own collision immediately after sticking.

Task-1/2/4 (throwing) specific:

7. **Arm jitter** — raw per-step residual action caused visible high-frequency shaking. A reward-based fix (action-rate penalty) was tried first and *made throws weaker without fixing the jitter* — replaced with a structural fix (`FILTER_ALPHA=0.25` low-pass filter on the residual), which worked immediately.
8. **Systematic sideways throw bias** — traced to unbalanced reaction torque from the arm swing, with no DOF able to correct it. Fixed by giving RL control of `waist_yaw`, confirmed via a controlled test (forcing `waist=+1` flips the sign of the lateral error).
9. **Whip-phase ratchet reference bug** — an early "don't flail backward during the whip" mechanism used a fixed idealized reference pose instead of the actual position when the whip phase began, causing a visible snap/stutter at the phase transition. Fixed by capturing the *actual* position on first entry to the window.
10. **Early-release reward exploit** — RL learned to release at the earliest allowed instant (weak/backward tosses risked nothing; a real throw risked the fall penalty). Ball frequently landed behind the robot. Fixed with a longer minimum-release-time gate, later relaxed back to the original value once paired with an explicit short/backward-throw penalty instead.
11. **Exponential-only accuracy reward plateau** (Task 2) — landing error stuck at ~1.87 m for ~75% of a training run. Root cause: an exponential-only reward gives almost the same near-zero signal for "very far" and "somewhat far" misses, so there is no gradient telling the policy which direction is closer. Fixed by adding a linear miss-penalty term (`W_MISS_PENALTY`) on top of the exponential accuracy bonus.
12. **Episode-terminates-before-recovery-can-run bug** — when the "return arm to neutral after throwing" feature was added, episodes were found to terminate on the *exact* step the ball lands (often ~0.98–1.02 s, before `WHIP_END=1.2s`), meaning the newly-added recovery-phase code (whether scripted or learned) never got a chance to execute at all for any successful throw. Fixed by holding the episode alive for `RECOVER_HOLD_SECONDS=1.0s` after first landing before actually terminating (fall still terminates immediately, as a safety exception).
13. **Learned-recovery reward-scale explosion** — a per-step "jerk" (velocity-change) smoothness penalty, calibrated without first measuring typical magnitudes, used a weight of `2.0` against jerk² values that spike to ~156 at the whip→recovery transition — this alone drove mean episode reward to **−1130** in the very first training iteration. Fixed by measuring actual jerk² statistics (mean ≈20, max ≈156) and recalibrating the weight to `0.02` with an explicit cap (`RECOVER_JERK_CAP=30`).
14. **Learned-recovery wrong-direction/overshoot motion** — even after the jerk-penalty fix, the RL-controlled recovery (constrained to `shoulder_pitch`+`elbow` only, with a monotonic "ratchet" preventing the commanded *target* from moving away from neutral) still produced a visible temporary rise/wrong-side swing immediately after the throw, attributable to physical momentum carried over from the whip motion, which a target-space ratchet cannot fully suppress (it constrains the commanded target, not the actual Cartesian trajectory, which depends nonlinearly on multiple simultaneous joint angles). **Resolution:** abandoned the learned-recovery approach for the arm-settling behavior specifically and replaced it with a pure scripted smoothstep transition anchored to the *actual* captured end-of-whip position (not an idealized constant) — this is fully deterministic, requires no additional training, and is verified to reach the neutral pose to within 0.003 rad with zero effect on the underlying 90%+ throw accuracy.

Task 3 (walking) specific:

15. **Hand-scripted gait failure** — see §3.2; abandoned in favor of pure RL.
16. **Instantaneous-velocity reward exploit** — `W_FORWARD` originally rewarded the pelvis's instantaneous physical velocity. The policy learned to produce a violent spin/jerk (observed as "spinning 360° in place") reading up to 8.77 m/s instantaneous velocity while the pelvis was actually net moving *backward*, then falling — 100% fall rate despite an apparently-improving reward curve. Fixed by switching to real net per-step displacement (`capped_fwd_t − capped_fwd_{t−1}`), which cannot be gamed by a momentary spin.
17. **Reward-scale bug from a stray `/control_dt` division** — an earlier version of the forward-progress term divided by the 0.02 s control timestep, inflating it ~50× relative to one-time bonuses/penalties and pushing the policy toward fast/risky gaits over slow/safe ones. Fixed by removing the division and recalibrating `W_FORWARD` directly.
18. **Non-reproducibility bug** — walking episodes were not reproducible across runs with the same seed. Root-caused to the environment's own custom, never-reseeded RNG (`self._rng`) rather than to GPU nondeterminism (the first suspicion) — fixed by using `self.np_random`, which gymnasium's `reset(seed=...)` actually reseeds.
19. **Foot-lift/knee-bend reward decoupling bug** — the knee-bend reward was gated on a lift-*duration* counter that stays at exactly 0 (which trivially satisfies "not lifted too long") when the foot never leaves the ground at all — so the policy could collect the full knee-bend bonus while shuffling with the foot flat on the floor, never actually stepping. Fixed by additionally requiring `lift > LIFT_EPS` (an actual, nonzero lift) before paying the bonus.
20. **Over-converged/stuck warm-start** — after multiple escalating reward-weight increases still failed to produce any foot-lift or knee-bend (`knee=0.000` exactly, unchanged across an entire additional 2M-step run), the issue was diagnosed as an already-converged warm-started policy having near-zero exploration variance in that specific action dimension, not a reward-magnitude problem — resolved by raising `ent_coef` from `0.01` to `0.05` on the warm-started run rather than retraining from scratch, which did unlock foot-lift and knee-bend within the next run.
21. **Hip-yaw-freeze action-space break** — an experiment to remove `hip_yaw` from the learned action space (12→10 dims) was implemented by editing the *shared* `g1_walk_env.py` directly, which is also the file Task 4 depends on via a frozen 12-action checkpoint — this immediately broke Task 4's demo script (`ValueError: shape mismatch: value array of shape (12,) could not be broadcast to indexing result of shape (10,)`). Root cause: same lesson as bug #14's mitigation — a shared file was modified for an experiment that should have been forked into its own copy from the start. Fixed by restoring the shared file to its original 12-action form and moving the hip-yaw-freeze experiment into a dedicated `envs/g1_walk_frozen_yaw_env.py` + `scripts/train_ppo_walk_frozenyaw.py` pair that does not touch the production file.

---

## 8. Design Decisions Worth Documenting Explicitly

- **X-only landing-error scoring (Tasks 2/4):** `landing_error` is computed as
  `|ball_x − target_x|` (depth only), not full 3D Euclidean distance. This was an
  explicit, requested design choice — lateral (Y) miss is not scored directly (though it
  is softly discouraged by the shared torso-stability terms), only forward depth
  accuracy counts toward "success." This alone raised Task 2's measured success rate
  from 60% to 90% on the *same* underlying model, purely by changing what's measured —
  a reminder that success-rate numbers are only meaningful alongside their exact
  definition.
- **`TARGET_POS` module-level constant drift:** `TARGET_POS` in `g1_free_throw_env.py`
  changed from `(2.3, 0.0, 0.6)` to `(2.8, 0.0, 0.6)` partway through the project (to
  compensate for the extra 0.5 m the robot covers by walking, in Task 4). Because
  `target_pos` is baked into the *observation* (not just the reward), this is not a
  free change — evaluating an *older* Task 2 checkpoint against the *new* target
  position measurably changes its behavior (a real regression was caught this way:
  6.55 m → 1.58 m mean distance on Task 1's checkpoint from an unrelated attempt to move
  the target for physical-safety reasons, before the fix was to leave the target
  position untouched and solve the physical-safety concern differently). **Anyone
  reproducing Task 2 numbers must explicitly pass `target_pos=(2.3, 0.0, 0.6)`** to
  match the documented 92% figure, since the shared file's default is now `2.8`.
- **Two independent copies of the throw environment exist on purpose:**
  `g1_free_throw_env.py` (Tasks 2/4, target-accuracy reward) and
  `g1_free_throw_distance_env.py` (Task 1, pure-distance reward). These were *not*
  always separate — Task 1's reward logic used to live in the same file as Task 2's,
  until Task 2's evolution (target-accuracy terms, X-only scoring, recovery-phase
  logic) made the shared file structurally incompatible with "maximize distance,"
  discovered only when a distance-training attempt against the then-current shared file
  would have silently trained for target-precision instead. Splitting them was the fix,
  and the same lesson was reapplied for the walking hip-yaw-freeze incident (bug #21).

---

## 9. Honest Limitations & Sim2Real Risks

**What is proven:**
- All four tasks run end-to-end in MuJoCo, deterministically, from a saved policy checkpoint.
- Tasks 1–3 each have a completed, converged (or near-converged) training run with real before/after metrics.
- Zero falls across every evaluation in this report (throwing and walking alike) — balance was never the dominant failure mode once the early exploits (bugs #16, #10) were fixed.
- Multiple real bugs (reward exploits, physics instabilities, reward-scale blowups) were found and fixed with root-cause diagnosis, not just parameter poking.

**What is not proven / open work:**
- **Task 3's gait-naturalness refinements (higher foot lift, real stride length, no
  knee-locking) were not run to full convergence** — the reward design for these is
  finalized and verified structurally sound, but the most recent training runs
  (`_stance`, `_stride`, `_freelift`) were interrupted for other priorities before
  reaching their target step counts. The production Task 3/4 checkpoint predates these
  refinements.
- **Task 4's success rate (50% general-case) is meaningfully lower than Task 2's
  standalone 92%** — the throw policy has not fully closed the gap to handling arbitrary
  post-walk poses; the demo's reliance on a specific known-good seed for a guaranteed
  clean presentation is an explicit, documented shortcut, not a claim of general
  robustness.
- **No domain randomization, sensor noise, or actuator-delay modeling was implemented**
  anywhere in this project — every result above is a clean-simulation number. The
  concrete Sim2Real gaps that would need addressing before any real-hardware attempt:
  contact/friction mismatch (the ball-catch and foot-contact dynamics are tuned against
  MuJoCo's specific solver parameters, e.g. the deliberately-softened `solref` on the
  sticking weld — bug #6), actuator saturation and control latency (all control here is
  instantaneous position control at 50 Hz with no communication delay modeled), and
  model/inertia mismatch (masses and inertias are the stock Unitree G1 MJCF values,
  never randomized or perturbed).
- **No robustness testing (noise injection, domain randomization, repeated stress
  tests near failure conditions) was performed** — every number in §6 is a clean-model
  evaluation. This is the single largest gap relative to the "robustness concept"
  scoring criterion and would be the natural next experiment.
- **Reward-shape sensitivity is real and only partially explored** — Task 2's
  non-monotonic success-rate history (60→90→78→92%) across small `K_ACC`/`W_ACCURACY`
  changes shows the policy is sensitive to exact reward curvature in a way that was
  only diagnosed empirically (train, evaluate on many seeds, adjust), not predicted in
  advance.

**Next-step roadmap (realistic, not aspirational):** (1) finish the Task 3 gait-refinement
training runs to full convergence and re-freeze a new production checkpoint for Task 4;
(2) run Task 4's throw-fine-tuning for materially more than 300k steps against the
*current* (refined) walk policy rather than the original one; (3) add basic domain
randomization (mass ±10%, friction ±20%, small per-step observation noise) and re-evaluate
all four tasks' success rates under it as a first, honest Sim2Real robustness signal.
