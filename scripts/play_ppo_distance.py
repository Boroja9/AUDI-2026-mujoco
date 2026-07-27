#!/usr/bin/env python3
"""Gledaj istreniranu PPO politiku za Task 1 (baci sto dalje, posebna kopija).

    python scripts/play_ppo_distance.py --model policies/g1_free_throw_ppo_distance_v2/best_model.zip
"""
from pathlib import Path
import argparse, sys, time

import mujoco, mujoco.viewer
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.g1_free_throw_distance_env import G1FreeThrowDistanceEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path,
                    default=ROOT / "policies" / "g1_free_throw_ppo_distance_only" / "best_model.zip")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()

    env = G1FreeThrowDistanceEnv()
    model = PPO.load(str(args.model), device="cpu")
    obs, _ = env.reset(seed=args.seed)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            viewer.sync()
            time.sleep(env.control_dt / args.speed)
            if term or trunc:
                print(f"ep kraj: ball_x={info['ball_x']:.2f} m  "
                      f"pao={info['robot_fell']}  release={info['release_time']}")
                time.sleep(0.8)
                obs, _ = env.reset(seed=args.seed)


if __name__ == "__main__":
    main()
