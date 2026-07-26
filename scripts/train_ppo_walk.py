#!/usr/bin/env python3
"""PPO trening: G1 uci da napravi dva koraka napred bez padanja.

    python scripts/train_ppo_walk.py --timesteps 500000 --n-envs 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_walk_env import G1WalkEnv


def make_env():
    return G1WalkEnv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "policies" / "g1_walk_ppo"
    )
    parser.add_argument(
        "--init", type=Path, default=None,
        help="Postojeci .zip model od kog se nastavlja trening (warm-start) umesto od nule."
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    train_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)
    eval_env = make_vec_env(make_env, n_envs=1, seed=args.seed + 10_000)
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.output),
        log_path=str(args.output / "evaluations"),
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    if args.init:
        model = PPO.load(str(args.init), env=train_env, tensorboard_log=str(args.output / "tensorboard"))
        print(f"warm-started from {args.init}")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(args.output / "tensorboard"),
        )
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    model.save(str(args.output / "final_model"))
    print(f"Gotovo. Modeli u: {args.output}")


if __name__ == "__main__":
    main()
