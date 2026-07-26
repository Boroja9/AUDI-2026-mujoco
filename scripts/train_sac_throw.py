"""G1 stands in place and throws a welded ball past 2 m, using SAC.

The ball is held by a weld equality constraint. Add to the scene XML if missing:

    <equality>
      <weld name="ball_grip" body1="right_wrist_yaw_link" body2="ball"
            relpose="0.16 0 0 1 0 0 0"/>
    </equality>

    python scripts/train_sac_throw.py --check
    python scripts/train_sac_throw.py --stage balance --steps 400000
    python scripts/train_sac_throw.py --stage throw --steps 1500000 \
        --init runs/balance/final.zip
    python scripts/train_sac_throw.py --play runs/throw/final.zip
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE = Path("assets/unitree_g1/scene_throw.xml")
HAND_BODY = "right_wrist_yaw_link"
BALL_OFFSET = 0.16

# --- control ---------------------------------------------------------------
CTRL_HZ = 50
ACTION_SCALE = 0.6          # rad, size of the offset from the standing pose
KP, KD = 150.0, 10.0        # softer than before -> less twitchy, more inertial

# --- episode ---------------------------------------------------------------
EPISODE_SEC = 5.0
MIN_HOLD = 0.5              # cannot release before this
MAX_HOLD = 2.0              # forced release
SETTLE_AFTER_LANDING = 0.2

# --- task ------------------------------------------------------------------
GOAL_DISTANCE = 2.0         # m, the bar to clear
GOAL_BONUS = 50.0           # flat, does NOT grow with distance
DRIFT_TOL = 0.15            # m of free pelvis movement

# --- reward weights --------------------------------------------------------
W_ALIVE = 1.0
W_UPRIGHT = 1.5
W_HEIGHT = 0.8
W_DRIFT = 3.0
W_LEG_POSTURE = 0.5
W_ACTION_RATE = 0.05        # main smoothness knob
W_TORQUE = 2e-4
W_BALL_SPEED = 0.5          # early-training signal only
W_RAMP = 5.0                # linear 0 -> GOAL_DISTANCE
W_ARM_FORWARD = 0.3         # forward swing good, sideways bad
FALL_PENALTY = 25.0

LEG_JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
]
WAIST_JOINTS = ["waist_yaw", "waist_roll", "waist_pitch"]
ARM_JOINTS = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
CONTROLLED = LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS
N_LEG = len(LEG_JOINTS)


class G1ThrowEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, stage="throw", scene=SCENE, learned_release=True,
                 fixed_release=1.0, render_mode=None, seed=None):
        super().__init__()
        self.stage = stage
        self.learned_release = learned_release
        self.fixed_release = fixed_release
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.decim = max(1, int(round(1.0 / CTRL_HZ / self.model.opt.timestep)))
        self.dt = self.model.opt.timestep * self.decim

        self._resolve_actuators()
        self.pelvis = self._body("pelvis", "torso_link", "base_link")
        self.hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        if self.hand < 0:
            raise RuntimeError(f"body '{HAND_BODY}' not found")
        self.ball_bid, self.ball_q, self.ball_d = self._find_ball()
        self.weld_id = self._find_weld()
        self.foot_geoms, self.floor_geoms = self._contact_sets()

        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.q_default = self.data.qpos[self.qadr].copy()
        self.z_default = float(self.data.xpos[self.pelvis, 2])

        self.action_space = spaces.Box(-1.0, 1.0, (self.nj + 1,), np.float32)
        obs_dim = 3 + 3 + self.nj * 3 + 1 + 1 + 2 + 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

        self.prev_action = np.zeros(self.nj + 1, np.float32)
        self.viewer = None
        self._rng = np.random.default_rng(seed)

    # -------------------------------------------------------------- setup
    def _resolve_actuators(self):
        amap = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                for i in range(self.model.nu)
                if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}
        ids, qa, da, lo_, hi_ = [], [], [], [], []
        self.missing = []
        for label in CONTROLLED:
            aid = next((i for n, i in amap.items()
                        if all(f in n.lower() for f in label.split("_"))), None)
            if aid is None:
                self.missing.append(label)
                continue
            jid = int(self.model.actuator_trnid[aid, 0])
            ids.append(aid)
            qa.append(int(self.model.jnt_qposadr[jid]))
            da.append(int(self.model.jnt_dofadr[jid]))
            a, b = self.model.actuator_ctrlrange[aid]
            if self.model.actuator_ctrllimited[aid] == 0:
                a, b = -1e6, 1e6
            lo_.append(a)
            hi_.append(b)
        self.act_ids = np.array(ids)
        self.qadr = np.array(qa)
        self.dadr = np.array(da)
        self.ctrl_lo = np.array(lo_)
        self.ctrl_hi = np.array(hi_)
        self.nj = len(ids)
        # index of the right shoulder pitch inside the controlled list
        self.rsp = next((i for i, l in enumerate(
            [l for l in CONTROLLED if l not in self.missing])
            if l == "right_shoulder_pitch"), None)

    def _body(self, *names):
        for n in names:
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            if b >= 0:
                return b
        return 1

    def _find_ball(self):
        for bid in range(1, self.model.nbody):
            name = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            j = self.model.body_jntadr[bid]
            if j >= 0 and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE \
                    and ("ball" in name or "projectile" in name):
                return bid, int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])
        raise RuntimeError("no free-joint ball body found")

    def _find_weld(self):
        for i in range(self.model.neq):
            if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD \
                    and self.ball_bid in (int(self.model.eq_obj1id[i]),
                                          int(self.model.eq_obj2id[i])):
                return i
        return None

    def _contact_sets(self):
        feet, floor = set(), set()
        for g in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[g])
            bname = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            gname = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
            if "ankle" in bname or "foot" in bname:
                feet.add(g)
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE \
                    or "floor" in gname or "ground" in gname:
                floor.add(g)
        return feet, floor

    def _set_weld(self, active):
        if self.weld_id is None:
            return
        try:
            self.data.eq_active[self.weld_id] = int(active)
        except (AttributeError, ValueError, IndexError):
            self.model.eq_active0[self.weld_id] = int(active)

    def _carry_kinematic(self):
        rot = self.data.xmat[self.hand].reshape(3, 3)
        off = rot @ np.array([BALL_OFFSET, 0.0, 0.0])
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.hand, vel, 0)
        omega, v = vel[:3], vel[3:]
        self.data.qpos[self.ball_q:self.ball_q + 3] = self.data.xpos[self.hand] + off
        self.data.qpos[self.ball_q + 3:self.ball_q + 7] = self.data.xquat[self.hand]
        self.data.qvel[self.ball_d:self.ball_d + 3] = v + np.cross(omega, off)
        self.data.qvel[self.ball_d + 3:self.ball_d + 6] = omega

    # ------------------------------------------------------------- helpers
    def _gravity_in_base(self):
        return self.data.xmat[self.pelvis].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

    def _bad_contact(self):
        """True if anything other than a foot touches the floor."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = {g1, g2}
            if not (pair & self.floor_geoms):
                continue
            other = g2 if g1 in self.floor_geoms else g1
            if other in self.floor_geoms:
                continue
            if other in self.foot_geoms:
                continue
            if int(self.model.geom_bodyid[other]) == self.ball_bid:
                continue
            return True
        return False

    def _throw_distance(self):
        return float(np.linalg.norm(self.data.xpos[self.ball_bid, :2] - self.origin_xy))

    def _obs(self):
        return np.concatenate([
            self._gravity_in_base(),
            self.data.cvel[self.pelvis, :3],
            self.data.qpos[self.qadr] - self.q_default,
            self.data.qvel[self.dadr] * 0.1,
            self.prev_action[:self.nj],
            [float(self.held)],
            [self.t / EPISODE_SEC],
            self.data.xpos[self.pelvis, :2] - self.origin_xy,
            self.data.xpos[self.ball_bid] - self.data.xpos[self.pelvis],
            self.data.qvel[self.ball_d:self.ball_d + 3] * 0.1,
        ]).astype(np.float32)

    # ----------------------------------------------------------------- gym
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] += self._rng.normal(0, 0.02, self.nj)
        self.data.qvel[:] += self._rng.normal(0, 0.01, self.model.nv)
        self._set_weld(True)
        mujoco.mj_forward(self.model, self.data)

        self.t = 0.0
        self.held = True
        self.landed = None
        self.prev_action[:] = 0.0
        self.origin_xy = self.data.xpos[self.pelvis, :2].copy()
        if self.weld_id is None:
            self._carry_kinematic()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        q_des = self.q_default + ACTION_SCALE * action[:self.nj]

        if self.held:
            if self.learned_release:
                go = (action[self.nj] > 0.0 and self.t >= MIN_HOLD) or self.t >= MAX_HOLD
            else:
                go = self.t >= self.fixed_release
            if go:
                self.held = False
                self._set_weld(False)

        for _ in range(self.decim):
            tau = KP * (q_des - self.data.qpos[self.qadr]) - KD * self.data.qvel[self.dadr]
            self.data.ctrl[self.act_ids] = np.clip(tau, self.ctrl_lo, self.ctrl_hi)
            if self.held and self.weld_id is None:
                self._carry_kinematic()
            mujoco.mj_step(self.model, self.data)
        self.t += self.dt

        r, term, trunc, info = self._reward(action)
        self.prev_action = action
        return self._obs(), r, term, trunc, info

    # -------------------------------------------------------------- reward
    def _reward(self, action):
        info = {}
        z = float(self.data.xpos[self.pelvis, 2])
        tilt = math.acos(np.clip(-self._gravity_in_base()[2], -1.0, 1.0))

        r = W_ALIVE
        r += W_UPRIGHT * math.exp(-8.0 * tilt ** 2)
        r += W_HEIGHT * math.exp(-40.0 * (z - self.z_default) ** 2)

        drift = float(np.linalg.norm(self.data.xpos[self.pelvis, :2] - self.origin_xy))
        r -= W_DRIFT * max(0.0, drift - DRIFT_TOL) ** 2
        info["drift"] = drift

        leg_dev = self.data.qpos[self.qadr[:N_LEG]] - self.q_default[:N_LEG]
        r -= W_LEG_POSTURE * float(np.dot(leg_dev, leg_dev)) / N_LEG

        d = action - self.prev_action
        r -= W_ACTION_RATE * float(np.dot(d, d))
        r -= W_TORQUE * float(np.dot(self.data.ctrl, self.data.ctrl))

        if self.stage == "throw":
            hand_v = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                     self.hand, hand_v, 0)
            # forward swing rewarded, sideways punished
            r += W_ARM_FORWARD * (hand_v[3] - abs(hand_v[4])) * self.dt

            if self.held:
                v = float(np.linalg.norm(self.data.qvel[self.ball_d:self.ball_d + 3]))
                r += W_BALL_SPEED * min(v, 15.0) * self.dt
            elif self.landed is None:
                if self.data.xpos[self.ball_bid, 2] < 0.06 \
                        and self.data.qvel[self.ball_d + 2] > -0.05:
                    self.landed = self.t
                    dist = self._throw_distance()
                    r += W_RAMP * min(dist, GOAL_DISTANCE)
                    if dist >= GOAL_DISTANCE:
                        r += GOAL_BONUS
                    info["throw_distance"] = dist
                    info["success"] = float(dist >= GOAL_DISTANCE)

        term = False
        if z < 0.45 or tilt > 1.0 or self._bad_contact():
            term = True
            r -= FALL_PENALTY
            info["fall"] = 1.0

        trunc = self.t >= EPISODE_SEC
        if self.landed is not None and self.t >= self.landed + SETTLE_AFTER_LANDING:
            trunc = True
        return r, term, trunc, info

    def render(self):
        import mujoco.viewer
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# --------------------------------------------------------------------------
def make_env(stage, seed, rank, learned_release):
    def _f():
        return G1ThrowEnv(stage=stage, seed=seed + rank,
                          learned_release=learned_release)
    return _f


def check():
    env = G1ThrowEnv()
    print(f"controlled joints : {env.nj}")
    print(f"missing actuators : {env.missing or 'none'}")
    print(f"weld constraint   : {env.weld_id}")
    print(f"foot geoms        : {len(env.foot_geoms)}")
    print(f"floor geoms       : {len(env.floor_geoms)}")
    print(f"obs / act         : {env.observation_space.shape} / {env.action_space.shape}")
    print(f"control dt        : {env.dt:.4f} s  ({1/env.dt:.0f} Hz)")
    obs, _ = env.reset()
    total = 0.0
    for _ in range(60):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            break
    print(f"random rollout    : reward {total:.1f}, info {info}")
    if env.weld_id is None:
        print("\n!! no weld found - add the <equality> block to the scene XML")
    if env.missing:
        print("\n!! fix the joint names in CONTROLLED before training")


def train(args):
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback

    out = Path("runs") / args.stage
    out.mkdir(parents=True, exist_ok=True)
    venv = SubprocVecEnv([make_env(args.stage, args.seed, i, not args.fixed_release)
                          for i in range(args.n_envs)])
    # reward normalisation is off on purpose: the replay buffer would hold
    # rewards normalised by out-of-date statistics
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    kw = dict(
        buffer_size=500_000, learning_starts=10_000, batch_size=512,
        tau=0.005, gamma=0.98, train_freq=(1, "step"), gradient_steps=-1,
        ent_coef="auto", learning_rate=3e-4, verbose=1, seed=args.seed,
        tensorboard_log=str(out / "tb"),
        policy_kwargs=dict(net_arch=[512, 256, 128]),
    )
    if args.init:
        model = SAC.load(args.init, env=venv, **kw)
        print(f"warm-started from {args.init}")
    else:
        model = SAC("MlpPolicy", venv, **kw)

    cb = CheckpointCallback(save_freq=max(1, 100_000 // args.n_envs),
                            save_path=str(out), name_prefix="ckpt")
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=True)
    model.save(out / "final")
    venv.save(str(out / "vecnorm.pkl"))
    print(f"saved {out/'final.zip'}")


def play(path, n=20):
    import time
    from stable_baselines3 import SAC
    env = G1ThrowEnv(stage="throw", render_mode="human")
    model = SAC.load(path)
    obs, _ = env.reset()
    dists, ok, eps = [], 0, 0
    while eps < n:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        env.render()
        time.sleep(env.dt)
        if "throw_distance" in info:
            d = info["throw_distance"]
            dists.append(d)
            ok += int(d >= GOAL_DISTANCE)
            print(f"throw {d:5.2f} m   {'OK' if d >= GOAL_DISTANCE else 'short'}")
        if term or trunc:
            eps += 1
            obs, _ = env.reset()
    if dists:
        print(f"\nmean {np.mean(dists):.2f} m | best {max(dists):.2f} m | "
              f"success {ok}/{len(dists)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["balance", "throw"], default="throw")
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=str, default=None)
    ap.add_argument("--play", type=str, default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fixed-release", action="store_true",
                    help="release at a fixed time instead of letting SAC decide")
    args = ap.parse_args()
    if args.check:
        check()
    elif args.play:
        play(args.play)
    else:
        train(args)


if __name__ == "__main__":
    main()
    