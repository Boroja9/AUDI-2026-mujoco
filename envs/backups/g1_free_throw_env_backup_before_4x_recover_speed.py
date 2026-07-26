"""Residual RL pod starim imenom: baseline pitcher kao osnova, RL uci korekciju."""

from __future__ import annotations

import mujoco
import numpy as np
from gymnasium import spaces

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv

ACTIVE_JOINTS = (0, 1, 3)
RESIDUAL_SCALE = 0.6
WAIST_RESIDUAL_SCALE = 0.3  # struk je osetljiviji zglob, manji opseg korekcije

RAISE_POSE_RAD = np.array([-2.0, -0.80, 0.0, 0.0, 0.0, 0.0, 0.0])
COCK_POSE_RAD = np.array([-2.0, -0.80, 0.0, -0.5, 0.0, 0.0, 0.0])
THROW_POSE_RAD = np.array([-2.0, -0.80, 0.0, 1.5, 0.0, 0.0, 0.0])
RAISE_END = 0.45
COCK_END = 1.00
WHIP_END = 1.20
RECOVER_DURATION = 0.175  # s - duplo brze (bilo 0.35) - koliko traje GLATKI skriptovani prelaz iz STVARNE
                          # (ne idealizovane) pozicije ruke u trenutku kraja
                          # bacanja, ka neutralnoj (0,0,0). RL vise NE ucestvuje
                          # u ovoj fazi - garantovano bez mrdanja/trzaja/pogresnog
                          # smera, jer je to cista matematika (smoothstep), ne
                          # nesto sto mreza "izlaze".
MIN_RELEASE_TIME = 0.4   # RL ne sme da ispusti loptu pre ovog trenutka
RELEASE_DEADLINE = 1.20  # prinudni release ako RL nikad ne odluci
BASELINE_SCALE = 3.2

TARGET_POS = (2.8, 0.0, 0.6)   # pomereno 0.5m dalje da kompenzuje hodanje pre bacanja
SUCCESS_RADIUS = 0.28          # "bullseye" - unutar ovog radijusa = pogodak
W_MISS_PENALTY = 3.0           # linearna kazna srazmerna udaljenosti - jasan signal "prisi blize" i kad je promasaj veliki
W_ACCURACY = 20.0              # ublazeno (bilo 25.0, pre toga 15.0) - K_ACC=2.5 je
                                # previse ostro sruslo success rate sa 90% na 78%,
                                # cilj je da ostane blizu 90% uz malo vise nagrade za centar
K_ACC = 1.8                    # ublazeno (bilo 2.5, pre toga 1.5)
BULLSEYE_BONUS = 25.0          # ublazeno (bilo 30.0, pre toga 20.0)
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

# KLJUCNO: epizoda se inace gasi ISTOG koraka kad lopta sleti/pogodi metu -
# to je bilo pre WHIP_END kod uspesnih bacanja (target je blizu), pa recovery
# kod NIKAD nije stigao da se izvrsi. Sad se epizoda drzi ziva jos ovoliko
# sekundi posle sletanja, da recovery stigne da se desi i nagradi.
RECOVER_HOLD_SECONDS = 1.0

# nisko-propusni filter na RL korekciju: garantuje glatkoce bez obzira sta
# mreza nauci (umesto da se to uci kroz reward, sto je bilo nepouzdano).
FILTER_ALPHA = 0.25


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class G1FreeThrowEnv(G1FixedBodyThrowEnv):
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
        # neutralna (nominalna) poza - ruka se posle bacanja spusta ovde i ostaje
        self._A_neutral = np.zeros_like(self._A_throw)
        # "recet": tokom zamaha (cock->whip) zglobovi ne smeju da se vrate
        # nazad ka cock pozi - samo napred ka throw, da se izbegne mahanje
        # rukom na sve strane. Van tog prozora (raise/cock/recover) slobodno.
        self._whip_direction = np.sign(
            self._A_throw[self.active_joints] - self._A_cock[self.active_joints]
        )

        # struk (waist_yaw) kao dodatni kontrolisani zglob: RL time moze da
        # kompenzuje reakcioni obrtni moment tela izazvan zamahom ruke, koji
        # inace sistematski skrece bacanje u stranu.
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

        # +1 waist, +1 odluka o ispustanju lopte (RL bira trenutak, min MIN_RELEASE_TIME)
        self.action_space = spaces.Box(
            -1, 1, shape=(len(self.active_joints) + 2,), dtype=np.float32
        )
        # prev_action (koji ulazi u baznu _get_obs) sad mora da uracuna i
        # dodatni (waist) aktuator - probaj stvarnu duzinu umesto da je
        # racunamo analiticki, da izbegnemo neslaganje sa observation_space.
        self.prev_action = np.zeros(self.n_arm + len(self.extra_actuator_ids) + 1)
        base_dim = super()._get_obs().shape[0]
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(base_dim + 11,), dtype=np.float32
        )
        self._released_here = False
        self._filtered_action = np.zeros(len(self.active_joints) + 1, dtype=np.float32)
        self._recover_hold_steps = int(round(RECOVER_HOLD_SECONDS / self.control_dt))

    def _baseline_action(self, t):
        if t < RAISE_END:
            return self._A_raise * _smoothstep(t / RAISE_END)
        if t < COCK_END:
            p = _smoothstep((t - RAISE_END) / (COCK_END - RAISE_END))
            return self._A_raise + (self._A_cock - self._A_raise) * p
        if t < WHIP_END:
            p = _smoothstep((t - COCK_END) / (WHIP_END - COCK_END))
            return self._A_cock + (self._A_throw - self._A_cock) * p
        # posle WHIP_END: recovery prelaz se racuna u step() na osnovu STVARNE
        # pozicije u tom trenutku (ne odavde) - ovo se koristi samo za
        # zglobove van active_joints (npr. rucni zglobovi), koji ionako
        # ostaju na neutrali.
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
        self._whip_high_water = None  # postavlja se na STVARNU poziciju pri ulasku u whip
        self._recover_entry_pos = None  # STVARNA pozicija u trenutku kraja bacanja
        self._recover_entry_t = None
        self._landed_at_step = None
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
        else:
            # RECOVER faza: cist skriptovani prelaz, RL vise NE ucestvuje.
            # Polazna tacka je STVARNA pozicija ruke na kraju whip faze
            # (self._whip_high_water - azuriran svaki korak tokom cock->whip,
            # pa poslednja vrednost = tacno gde je ruka bila kad je WHIP_END
            # dostignut), ne idealizovana konstanta. Garantovano glatko
            # (smoothstep) i u ispravnom smeru - nema RL suma, nema mrdanja.
            if self._recover_entry_pos is None:
                self._recover_entry_pos = (
                    self._whip_high_water.copy() if self._whip_high_water is not None
                    else full[self.active_joints].copy()
                )
                self._recover_entry_t = t
            p = _smoothstep((t - self._recover_entry_t) / RECOVER_DURATION)
            neutral_active = self._A_neutral[self.active_joints]
            full[self.active_joints] = (
                self._recover_entry_pos + (neutral_active - self._recover_entry_pos) * p
            )
        if COCK_END <= t < WHIP_END:
            # recet: samo napred (ka throw), nikad nazad (ka cock).
            # startuje od STVARNE pozicije ruke pri ulasku u whip, ne od
            # idealizovane konstante - da izbegnemo trzaj/zaustavljanje.
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


        ball_x = float(self._ball_pos()[0])
        ball_y = float(self._ball_pos()[1])
        just_landed_now = (
            not self.robot_fell
            and info.get("landing_error") is not None
            and self._landed_at_step is None
        )
        if just_landed_now:
            err = float(info["landing_error"])
            # linearna kazna daje jasan signal "prisi blize" cak i kad je promasaj
            # veliki (eksponencijalni deo sam po sebi skoro nista ne razlikuje
            # izmedju "jako daleko" i "malo manje daleko").
            r -= W_MISS_PENALTY * err
            # dodatni bonus koji brzo raste sto si blizi centru - fina preciznost.
            r += W_ACCURACY * np.exp(-K_ACC * err)
            if info.get("success"):
                r += BULLSEYE_BONUS

        # epizoda se vise NE gasi istog koraka kad lopta sleti/pogodi - drzi se
        # ziva jos RECOVER_HOLD_SECONDS da recovery (povratak ruke) stigne da
        # se izvrsi i nagradi. Pad i dalje odmah gasi epizodu (sigurnosno).
        if info.get("landing_error") is not None and not self.robot_fell:
            if self._landed_at_step is None:
                self._landed_at_step = self.step_count
            terminated = bool(
                self.step_count - self._landed_at_step >= self._recover_hold_steps
            )
        # sigurnosni prekid uvek vazi, bez obzira na hold prozor iznad
        if abs(ball_x) > MAX_BALL_X or abs(ball_y) > MAX_BALL_X:
            terminated = True

        info["ball_x"] = ball_x
        info["tilt"] = self._tilt()
        return obs, float(r), terminated, truncated, info
