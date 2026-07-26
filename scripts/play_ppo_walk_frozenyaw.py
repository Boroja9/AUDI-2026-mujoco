#!/usr/bin/env python3
"""Gledaj istreniranu PPO politiku za hodanje, hip_yaw zakljucan (10 akcija).

    python scripts/play_ppo_walk_frozenyaw.py --model policies/g1_walk_ppo_v2_frozenyaw/best_model.zip
"""
from pathlib import Path
import argparse, sys, time

import mujoco, mujoco.viewer
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.g1_walk_frozen_yaw_env import G1WalkFrozenYawEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path,
                    default=ROOT / "policies" / "g1_walk_ppo_v2_frozenyaw" / "best_model.zip")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    env = G1WalkFrozenYawEnv()
    model = PPO.load(str(args.model))
    obs, _ = env.reset(seed=0)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            viewer.sync()
            time.sleep(env.control_dt / args.speed)
            if term or trunc:
                print(f"ep kraj: fwd={info['fwd']:.2f} m  "
                      f"fell={info['fell']}  success={info['success']}")
                time.sleep(0.8)
                obs, _ = env.reset(seed=0)


if __name__ == "__main__":
    main()
