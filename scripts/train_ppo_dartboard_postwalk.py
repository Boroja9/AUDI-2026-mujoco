#!/usr/bin/env python3
"""Fino podesavanje throw modela da baca dobro i iz stanja "posle hoda", ne
samo iz standardne mirne poze.

    python scripts/train_ppo_dartboard_postwalk.py --timesteps 300000 --n-envs 4 \
        --init policies/g1_dartboard_ppo/best_model.zip
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

from envs.g1_post_walk_throw_env import G1PostWalkThrowEnv


def make_env():
    return G1PostWalkThrowEnv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "policies" / "g1_dartboard_postwalk_ppo"
    )
    parser.add_argument(
        "--init", type=Path, default=ROOT / "policies" / "g1_dartboard_ppo" / "best_model.zip",
        help="Postojeci .zip throw model od kog se nastavlja (warm-start)."
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
    model = PPO.load(str(args.init), env=train_env, tensorboard_log=str(args.output / "tensorboard"))
    print(f"warm-started from {args.init}")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    model.save(str(args.output / "final_model"))
    print(f"Gotovo. Modeli u: {args.output}")


if __name__ == "__main__":
    main()
