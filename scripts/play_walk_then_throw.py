#!/usr/bin/env python3
"""Sekvenca: prvo istrenirani hod (0.5m), pa istrenirano bacanje na metu.

Koristi dva odvojena istrenirana modela (razlicite politike), ali isti
MuJoCo scene (scene_throw.xml) - stanje (pozicija robota) se prenosi sa
kraja hoda na pocetak bacanja direktnim kopiranjem qpos/qvel.

    python scripts/play_walk_then_throw.py
"""
from pathlib import Path
import argparse, sys, time

import mujoco, mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.g1_walk_env import G1WalkEnv
from envs.g1_free_throw_env import G1FreeThrowEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk-model", type=Path,
                    default=ROOT / "policies" / "g1_walk_ppo_v2" / "best_model_frozen_test.zip")
    ap.add_argument("--throw-model", type=Path,
                    default=ROOT / "policies" / "g1_dartboard_postwalk_ppo" / "best_model.zip")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    walk_env = G1WalkEnv()
    # device="cpu" - CPU je determinisitcan, GPU (CUDA) unosi sitan
    # nedeterminizam u redosledu paralelnih racunanja koji se kroz epizodu
    # nagomila u primetno drugaciji ishod i pored istog seed-a.
    walk_model = PPO.load(str(args.walk_model), device="cpu")
    throw_env = G1FreeThrowEnv()
    throw_model = PPO.load(str(args.throw_model), device="cpu")
    # postavi metu na njenu pravu poziciju ODMAH, pre bilo kakvog sync-a sa
    # viewer-om - inace ostaje na default XML poziciji tokom hoda i onda
    # "teleportuje" se tacno kad pocne bacanje. NE zovemo pun reset() ovde
    # (to menja MuJoCo solver warm-start stanje i kvari determinizam
    # kasnijeg bacanja) - samo direktno postavimo poziciju.
    throw_env.model.body_pos[throw_env.target_body_id] = throw_env.target_pos

    with mujoco.viewer.launch_passive(throw_env.model, throw_env.data) as viewer:
        while viewer.is_running():
            # --- faza 1: hodanje ---
            obs, _ = walk_env.reset(seed=247)
            term = trunc = False
            while not (term or trunc) and viewer.is_running():
                action, _ = walk_model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = walk_env.step(action)
                # crtaj hod na throw_env-ovom vieweru: privremeno prebaci
                # qpos/qvel u prikazani model
                throw_env.data.qpos[:] = walk_env.data.qpos
                throw_env.data.qvel[:] = walk_env.data.qvel
                mujoco.mj_forward(throw_env.model, throw_env.data)
                viewer.sync()
                time.sleep(walk_env.control_dt / args.speed)
            print(f"hod gotov: fwd={info['fwd']:.2f} fell={info['fell']} success={info['success']}")

            # --- prenesi stanje sa kraja hoda na pocetak bacanja ---
            obs, _ = throw_env.reset(seed=0)
            throw_env.data.qpos[:] = walk_env.data.qpos
            throw_env.data.qvel[:] = walk_env.data.qvel
            mujoco.mj_forward(throw_env.model, throw_env.data)
            obs = throw_env._get_obs()

            # --- faza 2: bacanje ---
            term = trunc = False
            while not (term or trunc) and viewer.is_running():
                action, _ = throw_model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = throw_env.step(action)
                viewer.sync()
                time.sleep(throw_env.control_dt / args.speed)
            print(f"bacanje gotovo: ball_x={info.get('ball_x'):.2f} "
                  f"landing_error={info.get('landing_error')} success={info.get('success')} "
                  f"pao={info['robot_fell']}")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
