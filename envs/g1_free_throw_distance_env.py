"""Posebna kopija g1_free_throw_env.py SAMO za Task 1 (baci sto dalje, bez mete).

Napravljeno zato sto je g1_free_throw_env.py u medjuvremenu postao iskljucivo
target-accuracy env (nagrada je uvek "koliko si blizu target_pos", ne "koliko
si daleko baciо") - taj fajl sada koriste Task 2/3 i ne sme se dirati. Ovo je
nezavisna kopija sa stvarnom "baci sto dalje" nagradom, da ne pokvarimo ono sto
vec radi za Task 2/3.

    python scripts/train_ppo_distance.py --timesteps 500000 --n-envs 4
"""

from __future__ import annotations

import mujoco
import numpy as np
from gymnasium import spaces

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv

ACTIVE_JOINTS = (0, 1, 3)
RESIDUAL_SCALE = 0.6
WAIST_RESIDUAL_SCALE = 0.3

RAISE_POSE_RAD = np.array([-2.0, -0.80, 0.0, 0.0, 0.0, 0.0, 0.0])
COCK_POSE_RAD = np.array([-2.0, -0.80, 0.0, -0.5, 0.0, 0.0, 0.0])
THROW_POSE_RAD = np.array([-2.0, -0.80, 0.0, 1.5, 0.0, 0.0, 0.0])
RAISE_END = 0.45
COCK_END = 1.00
WHIP_END = 1.20
LOWER_END = 1.55  # posle bacanja, ruka se za ovoliko sekundi spusti u neutralnu
                  # (nominalnu) pozu i ostaje tu - ne vise zamrznuta u pozi bacanja
MIN_RELEASE_TIME = 0.4
RELEASE_DEADLINE = 1.20
BASELINE_SCALE = 3.2

# ISTA pozicija kao u deljenom g1_free_throw_env.py (Task 2/3) - target_pos
# ulazi u observation, pa menjanje njegove vrednosti menja sta mreza "vidi"
# i kvari ponasanje vec istreniranog modela (probano: pomeranje mete je
# smanjilo prosecan domet sa 2.33m na 1.58m na istim seed-ovima, iako je
# fizicko preklapanje sa metom u praksi bio nepostojeci problem).
TARGET_POS = (2.8, 0.0, 0.6)
SUCCESS_RADIUS = 0.28

# --- prava "baci sto dalje" nagrada (nema mete, nagradjuje se sam domet) ---
W_DISTANCE = 8.0        # nagrada srazmerna dometu, do DISTANCE_CAP
DISTANCE_CAP = 6.0      # m - dalje od ovoga se vise ne nagradjuje (da ne trci u beskraj)
W_SIDE_THROW = 3.0      # kazna za bocno (Y) skretanje lopte od prave linije
W_SHORT_THROW = 15.0    # kazna za bacanje unazad ili kratko (< SHORT_THROW_MIN)
SHORT_THROW_MIN = 0.1   # m

MAX_BALL_X = 12.0
W_RELEASE_VX = 1.0
W_RELEASE_SPEED_CAP = 8.0
W_TILT = 1.2
TILT_FREE = 0.12
W_WOBBLE = 0.05
W_SWAY = 0.03
W_HEIGHT = 1.0
W_RESIDUAL = 0
ALIVE_BONUS = 0.05
FALL_PENALTY = 20.0

FILTER_ALPHA = 0.25


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class G1FreeThrowDistanceEnv(G1FixedBodyThrowEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("learned_release", True)
        kwargs.setdefault("action_scale", BASELINE_SCALE)
        kwargs.setdefault("episode_time", 3.0)
        kwargs.setdefault("target_pos", TARGET_POS)
        kwargs.setdefault("success_radius", SUCCESS_RADIUS)
        super().__init__(**kwargs)
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        if self.torso_body_id < 0:
            self.torso_body_id = self.base_body_id
        self.active_joints = np.array(ACTIVE_JOINTS, dtype=np.int32)
        nom = self.nominal_ctrl[self.arm_actuator_ids]
        self._A_raise = np.clip((RAISE_POSE_RAD[: self.n_arm] - nom) / self.action_scale, -1, 1)
        self._A_cock = np.clip((COCK_POSE_RAD[: self.n_arm] - nom) / self.action_scale, -1, 1)
        self._A_throw = np.clip((THROW_POSE_RAD[: self.n_arm] - nom) / self.action_scale, -1, 1)
        self._A_neutral = np.zeros_like(self._A_throw)
        self._whip_direction = np.sign(
            self._A_throw[self.active_joints] - self._A_cock[self.active_joints]
        )

        waist_aid = None
        for aid in range(self.model.nu):
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) == "waist_yaw_joint":
                waist_aid = aid
                break
        if waist_aid is None:
            raise RuntimeError("Could not find waist_yaw_joint actuator.")
        waist_jid = int(self.model.actuator_trnid[waist_aid, 0])
        wlo, whi = self.model.jnt_range[waist_jid]
        self.extra_actuator_ids = np.array([waist_aid], dtype=np.int32)
        self.extra_joint_lower = np.array([wlo + self.joint_safety_margin])
        self.extra_joint_upper = np.array([whi - self.joint_safety_margin])
        self.waist_qpos_adr = int(self.model.jnt_qposadr[waist_jid])
        self.waist_qvel_adr = int(self.model.jnt_dofadr[waist_jid])

        self.action_space = spaces.Box(
            -1, 1, shape=(len(self.active_joints) + 2,), dtype=np.float32
        )
        self.prev_action = np.zeros(self.n_arm + len(self.extra_actuator_ids) + 1)
        base_dim = super()._get_obs().shape[0]
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(base_dim + 11,), dtype=np.float32
        )
        self._released_here = False
        self._filtered_action = np.zeros(len(self.active_joints) + 1, dtype=np.float32)

    def _baseline_action(self, t):
        if t < RAISE_END:
            return self._A_raise * _smoothstep(t / RAISE_END)
        if t < COCK_END:
            p = _smoothstep((t - RAISE_END) / (COCK_END - RAISE_END))
            return self._A_raise + (self._A_cock - self._A_raise) * p
        if t < WHIP_END:
            p = _smoothstep((t - COCK_END) / (WHIP_END - COCK_END))
            return self._A_cock + (self._A_throw - self._A_cock) * p
        if t < LOWER_END:
            p = _smoothstep((t - WHIP_END) / (LOWER_END - WHIP_END))
            return self._A_throw + (self._A_neutral - self._A_throw) * p
        return self._A_neutral

    def _base_obs_extra(self):
        pelvis_z = self.data.xpos[self.base_body_id, 2]
        R = self.data.xmat[self.torso_body_id].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id, vel, 0,
        )
        waist_qpos = self.data.qpos[self.waist_qpos_adr]
        waist_qvel = self.data.qvel[self.waist_qvel_adr]
        return np.concatenate(
            [[pelvis_z], grav[:2], vel[3:6], vel[:3], [waist_qpos, waist_qvel]]
        )

    def _get_obs(self):
        return np.concatenate(
            [super()._get_obs(), self._base_obs_extra()]
        ).astype(np.float32)

    def _tilt(self):
        R = self.data.xmat[self.torso_body_id].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        return float(np.linalg.norm(grav[:2]))

    def reset(self, seed=None, options=None):
        self._released_here = False
        self._filtered_action = np.zeros(len(self.active_joints) + 1, dtype=np.float32)
        self._whip_high_water = None
        return super().reset(seed=seed, options=options)

    def step(self, rl_action):
        t = self.step_count * self.control_dt
        rl_action = np.asarray(rl_action, dtype=np.float32).ravel()
        self._filtered_action += FILTER_ALPHA * (
            rl_action[: len(self._filtered_action)] - self._filtered_action
        )
        arm_residual = self._filtered_action[: len(self.active_joints)]
        waist_residual = self._filtered_action[len(self.active_joints)]
        full = self._baseline_action(t).copy()
        if t < WHIP_END:
            full[self.active_joints] = np.clip(
                full[self.active_joints] + RESIDUAL_SCALE * arm_residual,
                -1, 1,
            )
        if COCK_END <= t < WHIP_END:
            if self._whip_high_water is None:
                self._whip_high_water = full[self.active_joints].copy()
            delta = full[self.active_joints] - self._whip_high_water
            forward = np.maximum(delta * self._whip_direction, 0.0) * self._whip_direction
            full[self.active_joints] = self._whip_high_water + forward
            self._whip_high_water = full[self.active_joints].copy()
        release_decision = rl_action[len(self.active_joints) + 1]
        release_cmd = -1.0
        if not self.released:
            if t >= MIN_RELEASE_TIME and release_decision > 0.5:
                release_cmd = 1.0
                self._released_here = True
            elif t >= RELEASE_DEADLINE:
                release_cmd = 1.0
        full9 = np.zeros(self.n_arm + 1 + 1, dtype=np.float32)
        full9[: self.n_arm] = full
        full9[self.n_arm] = np.clip(WAIST_RESIDUAL_SCALE * waist_residual, -1, 1)
        full9[-1] = release_cmd
        obs, _, terminated, truncated, info = super().step(full9)

        r = ALIVE_BONUS
        r -= W_RESIDUAL * float(np.linalg.norm(rl_action))
        r -= W_TILT * max(0.0, self._tilt() - TILT_FREE)
        bv = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.base_body_id, bv, 0)
        r -= W_WOBBLE * float(np.linalg.norm(bv[:3]))
        r -= W_SWAY * float(np.linalg.norm(bv[3:6]))
        r -= W_HEIGHT * max(0.0, self.nominal_base_height - self.data.xpos[self.base_body_id, 2])
        if self.robot_fell:
            r -= FALL_PENALTY
        if self._released_here:
            self._released_here = False
            r += W_RELEASE_VX * min(max(0.0, self._ball_vel()[0]), W_RELEASE_SPEED_CAP)

        landed = terminated and not self.robot_fell and self._ball_landed()
        ball_x = float(self._ball_pos()[0])
        ball_y = float(self._ball_pos()[1])
        if landed:
            # prava "sto dalje" nagrada: srazmerna dometu (capped), kazna za
            # bocno skretanje od prave linije, kazna za kratko/nazad bacanje.
            r += W_DISTANCE * min(max(0.0, ball_x), DISTANCE_CAP)
            r -= W_SIDE_THROW * abs(ball_y)
            if ball_x < SHORT_THROW_MIN:
                r -= W_SHORT_THROW
            if abs(ball_x) > MAX_BALL_X or abs(ball_y) > MAX_BALL_X:
                terminated = True
        info["ball_x"] = ball_x
        info["tilt"] = self._tilt()
        return obs, float(r), terminated, truncated, info

    def _ball_landed(self):
        ball_pos = self._ball_pos()
        return bool(self.released and ball_pos[2] <= self.ball_radius + 0.015)
