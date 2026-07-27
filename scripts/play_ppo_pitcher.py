#!/usr/bin/env python3
"""Gledaj istreniranu PPO politiku za slobodno bacanje.

    python scripts/play_ppo_pitcher.py --model policies/g1_free_throw_ppo/best_model.zip
"""
from pathlib import Path
import argparse, sys, time

import mujoco, mujoco.viewer
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.g1_free_throw_env import G1FreeThrowEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path,
                    default=ROOT / "policies" / "g1_free_throw_ppo_distance_only" / "best_model.zip")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--scene", type=Path, default=None,
                    help="Alternativna MJCF scena (npr. scene_throw_2m.xml za metu na 2m).")
    ap.add_argument("--target-x", type=float, default=None,
                    help="Override X pozicije mete (npr. 2.0). Meta uvek ostaje na y=0, z=0.6.")
    args = ap.parse_args()

    env_kwargs = {}
    if args.scene is not None:
        env_kwargs["xml_path"] = args.scene
    if args.target_x is not None:
        env_kwargs["target_pos"] = (args.target_x, 0.0, 0.6)
    env = G1FreeThrowEnv(**env_kwargs)
    # device="cpu" - CPU je deterministicno, GPU (CUDA) unosi sitan
    # nedeterminizam koji se kroz epizodu nagomila u primetno drugaciji ishod
    # i pored istog seed-a (potvrdjeno: promasaj koji se ne ponavlja headless).
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