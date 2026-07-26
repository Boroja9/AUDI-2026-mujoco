#!/usr/bin/env python3
"""G1 pitcher throw u tri faze:

  1. RAISE  — nadlaktica se podize horizontalno, normalno na torzo
  2. COCK   — podlaktica sa sakom se zabacuje gore/nazad (rotacija ramena
              oko ose nadlaktice + savijen lakat)
  3. WHIP   — brza ekstenzija lakta + rame + wrist snap povlace saku napred,
              weld se deaktivira usred zamaha i lopta leti napred

Stavi fajl u scripts/ i pokreni iz root-a repoa:
    python scripts/pitcher_throw.py       (Linux/Windows)
    mjpython scripts/pitcher_throw.py     (macOS)

Redosled zglobova: shoulder pitch, shoulder roll, shoulder yaw, elbow,
wrist roll, wrist pitch, wrist yaw.
"""

from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv

# ----------------------------- tjuning ---------------------------------------
# Poza 1: nadlaktica horizontalno napred, lakat blago savijen.
RAISE_POSE_RAD = np.array([-2.0, -0.80, 0.00, 0, 0, 0.0, 0.0])

# Poza 2 (cocked): shoulder yaw 2.5 rotira ravan lakta tako da savijena
# podlaktica ide GORE/NAZAD umesto dole; lakat skoro maksimalno savijen.
COCK_POSE_RAD = np.array([-2, -0.8, 0, -0.5, 0.0, 0, 0.0])

# Poza 3 (whip target): namerno PREKO granica zglobova — env klipuje na
# sigurnosne limite, a PD kontroler dobija maksimalan drive pa je zamah brz.
THROW_POSE_RAD = np.array([-2, -0.8, 0, 1.5, 0.0, 0, 0.0])

RAISE_END = 0.45     # s
COCK_END = 1      # s
WHIP_END = 1.2      # s (COCK_END + 0.12 — kratko = brz zamah)
RELEASE_TIME = 1.1  # s, weld se gasi; kvantizovano na control_dt=0.02!

PITCHER_ACTION_SCALE = 3.2
# ------------------------------------------------------------------------------


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def make_actions(env):
    nominal = env.nominal_ctrl[env.arm_actuator_ids]
    def to_action(pose):
        return np.clip((pose - nominal) / env.action_scale, -1, 1)
    return (np.zeros(7), to_action(RAISE_POSE_RAD),
            to_action(COCK_POSE_RAD), to_action(THROW_POSE_RAD))


def act(t, n_arm, start, raise_a, cock_a, throw_a):
    action = np.zeros(n_arm + 1, dtype=np.float32)
    n = min(n_arm, 7)
    if t < RAISE_END:
        p = smoothstep(t / RAISE_END)
        action[:n] = start[:n] + (raise_a[:n] - start[:n]) * p
    elif t < COCK_END:
        p = smoothstep((t - RAISE_END) / (COCK_END - RAISE_END))
        action[:n] = raise_a[:n] + (cock_a[:n] - raise_a[:n]) * p
    elif t < WHIP_END:
        p = smoothstep((t - COCK_END) / (WHIP_END - COCK_END))
        action[:n] = cock_a[:n] + (throw_a[:n] - cock_a[:n]) * p
    else:
        action[:n] = throw_a[:n]
    action[-1] = 1.0 if t >= RELEASE_TIME else 0.0  # gasi weld hold_throw_ball
    return action


def main():
    env = G1FixedBodyThrowEnv(
        learned_release=True,
        action_scale=PITCHER_ACTION_SCALE,
        episode_time=3.0,
    )
    target_geom = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_GEOM, "throw_target_geom"
    )
    if target_geom >= 0:
        env.model.geom_rgba[target_geom, 3] = 0.0  # slobodno bacanje, bez mete

    start, raise_a, cock_a, throw_a = make_actions(env)
    env.reset(seed=42)

    print("Pitcher viewer. Zatvori prozor ili Ctrl+C za kraj.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            t = env.step_count * env.control_dt
            _, _, terminated, truncated, info = env.step(
                act(t, env.n_arm, start, raise_a, cock_a, throw_a)
            )
            viewer.sync()
            time.sleep(env.control_dt)
            if terminated or truncated:
                rel = info["release_time"]
                rel_txt = f"{rel:.2f} s" if rel is not None else "nikad"
                print(f"bacanje gotovo: release={rel_txt}, pao={info['robot_fell']}")
                time.sleep(0.8)
                env.reset(seed=42)


if __name__ == "__main__":
    main()