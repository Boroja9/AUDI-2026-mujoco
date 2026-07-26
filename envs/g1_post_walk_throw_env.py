"""G1FreeThrowEnv varijanta koja pocinje iz stanja "upravo zavrsio hod"
umesto iz standardne mirne stojece poze - da throw model nauci da se nosi
sa zaostalom nestabilnoscu/brzinom posle hoda, ne samo sa mirnim startom."""

from __future__ import annotations

from pathlib import Path

import mujoco
from stable_baselines3 import PPO

from envs.g1_free_throw_env import G1FreeThrowEnv
from envs.g1_walk_env import G1WalkEnv

ROOT = Path(__file__).resolve().parents[1]
WALK_MODEL_PATH = ROOT / "policies" / "g1_walk_ppo_v2" / "best_model_frozen_test.zip"
MAX_WALK_SEED = 300


class G1PostWalkThrowEnv(G1FreeThrowEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._walk_env = G1WalkEnv()
        self._walk_model = PPO.load(str(WALK_MODEL_PATH), device="cpu")

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        walk_seed = int(self.np_random.integers(0, MAX_WALK_SEED))
        self.last_walk_seed = walk_seed
        wobs, _ = self._walk_env.reset(seed=walk_seed)
        term = trunc = False
        while not (term or trunc):
            waction, _ = self._walk_model.predict(wobs, deterministic=True)
            wobs, r, term, trunc, winfo = self._walk_env.step(waction)
        self.data.qpos[:] = self._walk_env.data.qpos
        self.data.qvel[:] = self._walk_env.data.qvel
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), info
