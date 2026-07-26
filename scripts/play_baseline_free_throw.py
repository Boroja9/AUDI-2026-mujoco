#!/usr/bin/env python3
"""Gledaj cist baseline (bez RL korekcije) za slobodno bacanje.

    python scripts/play_baseline_free_throw.py
"""
from pathlib import Path
import sys, time

import mujoco, mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.g1_free_throw_env import G1FreeThrowEnv


def main():
    env = G1FreeThrowEnv()
    obs, _ = env.reset(seed=0)
    zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            obs, _, term, trunc, info = env.step(zero_action)
            viewer.sync()
            time.sleep(env.control_dt)
            if term or trunc:
                print(f"ep kraj: ball_x={info['ball_x']:.2f} m  "
                      f"pao={info['robot_fell']}  release={info['release_time']}")
                time.sleep(0.8)
                obs, _ = env.reset(seed=0)


if __name__ == "__main__":
    main()
