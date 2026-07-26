"""RL uci G1 da napravi dva koraka napred, bez padanja.

Odvojeno od bacanja - cist walking zadatak. RL sam otkriva balans/hod
(pokusaji sa rucno skriptovanim hodom su padali - jednonozni oslonac
zahteva feedback koji je tesko rucno podesiti)."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE = Path(__file__).resolve().parents[1] / "assets" / "unitree_g1" / "scene_throw.xml"

LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

CTRL_HZ = 50
ACTION_SCALE = 0.5
EPISODE_SEC = 3.0
TARGET_DISTANCE = 0.5   # ~dva koraka napred
FALL_HEIGHT_RATIO = 0.55

# indeksi hip_roll i hip_yaw u LEG_JOINTS - kaznimo preteranu rotaciju u
# stranu/oko ose da sprecimo unakrstanje nogu (smanjeno - bilo je prejako,
# gusilo svaki pokusaj koraka pa se politika zaglavila na "ne mrdaj")
HIP_LATERAL_IDX = [1, 2, 7, 8]  # L_hip_roll, L_hip_yaw, R_hip_roll, R_hip_yaw
W_HIP_LATERAL = 0.5

# levi/desni parovi (isti zglob, suprotna noga) - za kaznu asimetrije (smanjeno)
LEFT_IDX = [0, 1, 2, 3, 4, 5]
RIGHT_IDX = [6, 7, 8, 9, 10, 11]
W_ASYMMETRY = 0.1

# "stabilan na kraju": mora da ostane u ciljnoj zoni SETTLE_STEPS koraka
# (nizak nagib/brzina), ne samo da je trenutno prosao 0.5m usred pada
SETTLE_STEPS = 15
SETTLE_TILT_MAX = 0.15
SETTLE_VEL_MAX = 0.3

W_FORWARD = 40.0  # rekalibrisano posle uklanjanja /control_dt (bilo je 6.0 sa
                   # deljenjem = efektivno ~150 ukupno; sad 40*0.5m = 20 ukupno,
                   # razumno u odnosu na SUCCESS_BONUS=30)
W_ALIVE = 0.05
W_UPRIGHT = 1.0
W_HEIGHT = 2.0   # povecano - jaca kazna ako karlica padne ispod normalne visine
W_SIDE = 1.0
W_ACTION_RATE = 0.02  # smanjeno iz istog razloga (previse gusilo pokusaj koraka)
SUCCESS_BONUS = 30.0
FALL_PENALTY = 30.0   # povecano - jaca kazna za pad robota

# nagrada za podizanje stopala od zemlje - da noge ne vuce po podu nego da
# ih stvarno podize kao pri hodu. FOOT_REST_Z je visina stopala kad je
# ravno na podu (izmereno). Ograniceno na MAX_FOOT_LIFT da ne "farmi"
# nagradu pretreanim/neprirodnim mahsiranjem (marširanje previsoko).
FOOT_REST_Z = 0.033
MAX_FOOT_LIFT = 0.16  # povecano (bilo 0.10) - noga sme/treba da ide vise, prirodniji korak
W_FOOT_LIFT = 32.0    # povecano (bilo 18.0) - podizanje stopala vaznije od savijanja kolena,
                      # max doprinos (32*0.16=5.12) sada preteze nad kolenom (ispod)
                      # (vuci noge) je vec davala solidnu nagradu pa se mrezi
                      # nije isplatilo da rizikuje promenu; ovo mora da bude
                      # dovoljno veliko da podizanje postane ocigledno bolje
# nagrada za podignutu nogu vazi samo dok je podignuta unutar prirodnog
# trajanja zamaha (swing faze) - ne beskonacno, da ne "visi" u vazduhu
# radi farmljenja nagrade nego da stvarno koracka.
MAX_LIFT_DURATION = 0.4  # sekunde

# direktna nagrada za savijanje kolena (ne samo visinu stopala) - podstice
# da se koleno stvarno savija tokom koraka, ne da noga ide gore uglavnom
# iz kuka.
KNEE_IDX = [3, 9]  # L_knee, R_knee u LEG_JOINTS
MAX_KNEE_BEND = 0.6  # rad
W_KNEE_BEND = 6.0  # smanjeno (bilo 15.0) - koleno je sekundarno, samo pomaze da
                   # hod izgleda prirodnije; podizanje stopala (W_FOOT_LIFT) je
                   # glavni prioritet, max doprinos kolena (6*0.6=3.6) < stopala (5.12)

# kazna ako OBE noge napuste pod istovremeno (skok) - hod treba da zadrzi
# bar jedno stopalo u kontaktu sa podom u svakom trenutku
W_AIRBORNE = 4.0

# stvarna kazna (ne samo izostanak nagrade) za prelazak preko cilja, i
# kontinualan (ne samo jednokratan) podsticaj da MIRUJE kad stigne tamo
W_OVERSHOOT = 6.0
W_STAND_STILL = 2.0

# kazna ako krene unazad (fwd < 0) - smanjeno, ne treba da bude prestrogo
W_BACKWARD = 2.0

FILTER_ALPHA = 0.3  # nisko-propusni filter na akciju, isti trik kao kod bacanja


class G1WalkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scene=SCENE):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.control_dt = 1.0 / CTRL_HZ
        self.frame_skip = max(1, int(round(self.control_dt / self.model.opt.timestep)))

        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self.torso_id < 0:
            self.torso_id = self.pelvis_id
        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geom_ids = set(
            g for g in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[g]) or "")
            .startswith(("left_ankle", "right_ankle"))
        )

        self.leg_joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEG_JOINTS]
        )
        if np.any(self.leg_joint_ids < 0):
            missing = [n for n, j in zip(LEG_JOINTS, self.leg_joint_ids) if j < 0]
            raise RuntimeError(f"Missing leg joints: {missing}")
        self.leg_qpos_adr = self.model.jnt_qposadr[self.leg_joint_ids]
        self.leg_qvel_adr = self.model.jnt_dofadr[self.leg_joint_ids]

        amap = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                for i in range(self.model.nu)
                if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}
        self.leg_actuator_ids = np.array([amap[n] for n in LEG_JOINTS])
        self.n_leg = len(LEG_JOINTS)

        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.nominal_ctrl = self.data.ctrl.copy()
        self.nominal_qpos = self.data.qpos.copy()
        self.nominal_base_height = float(self.data.xpos[self.pelvis_id, 2])
        self.fall_height = FALL_HEIGHT_RATIO * self.nominal_base_height

        self.action_space = spaces.Box(-1, 1, shape=(self.n_leg,), dtype=np.float32)
        obs_dim = 1 + 2 + 3 + 3 + self.n_leg + self.n_leg + self.n_leg + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.origin_x = 0.0
        self._filtered_action = np.zeros(self.n_leg, dtype=np.float32)
        self._prev_action = np.zeros(self.n_leg, dtype=np.float32)
        self._settle_count = 0
        self._prev_fwd = 0.0
        self._prev_capped_fwd = 0.0
        self._left_lift_steps = 0
        self._right_lift_steps = 0

    def _gravity_in_base(self):
        R = self.data.xmat[self.torso_id].reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0])

    def _get_obs(self):
        grav = self._gravity_in_base()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                  self.pelvis_id, vel, 0)
        pelvis_z = self.data.xpos[self.pelvis_id, 2]
        leg_qpos = self.data.qpos[self.leg_qpos_adr] - self.nominal_qpos[self.leg_qpos_adr]
        leg_qvel = self.data.qvel[self.leg_qvel_adr] * 0.1
        return np.concatenate([
            [pelvis_z], grav[:2], vel[3:6], vel[:3],
            leg_qpos, leg_qvel, self._prev_action,
            [max(0.0, EPISODE_SEC - self.step_count * self.control_dt)],
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self.nominal_qpos
        self.data.ctrl[:] = self.nominal_ctrl
        self.data.qpos[self.leg_qpos_adr] += self.np_random.uniform(-0.02, 0.02, self.n_leg)
        mujoco.mj_forward(self.model, self.data)
        self.origin_x = float(self.data.xpos[self.pelvis_id, 0])
        self.step_count = 0
        self._filtered_action[:] = 0.0
        self._prev_action[:] = 0.0
        self._settle_count = 0
        self._prev_fwd = 0.0
        self._prev_capped_fwd = 0.0
        self._left_lift_steps = 0
        self._right_lift_steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).ravel(), -1, 1)
        self._filtered_action += FILTER_ALPHA * (action - self._filtered_action)
        targets = self.nominal_ctrl[self.leg_actuator_ids] + ACTION_SCALE * self._filtered_action
        self.data.ctrl[:] = self.nominal_ctrl
        self.data.ctrl[self.leg_actuator_ids] = targets
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        pelvis = self.data.xpos[self.pelvis_id]
        grav = self._gravity_in_base()
        tilt = float(np.linalg.norm(grav[:2]))
        fwd = float(pelvis[0] - self.origin_x)
        side = float(pelvis[1])
        fell = bool(pelvis[2] < self.fall_height or tilt > 0.7)

        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                  self.pelvis_id, vel, 0)
        # stvaran pomak po koraku, ne trenutna fizicka brzina - trenutna brzina
        # se moze "prevariti" naglim trzajem/rotacijom oko jedne noge koja da
        # veliku ocitanu brzinu bez stvarnog napretka (potvrdjeno u praksi).
        # Pomak se racuna do TARGET_DISTANCE - dalje od te linije se ne
        # nagradjuje, da robot nema razlog da nastavi da trci umesto da stane.
        capped_fwd = min(fwd, TARGET_DISTANCE)
        # BEZ deljenja sa control_dt - to je ranije vestacki naduvavalo ovu
        # nagradu ~50x (1/0.02s) u odnosu na jednokratne bonuse/kazne, sto je
        # guralo politiku da preferira brz/rizican hod nad sporim/bezbednim.
        progress = capped_fwd - self._prev_capped_fwd
        self._prev_capped_fwd = capped_fwd
        self._prev_fwd = fwd

        r = W_ALIVE
        r += W_FORWARD * progress
        r += W_UPRIGHT * max(0.0, 1.0 - tilt)
        r -= W_HEIGHT * max(0.0, self.nominal_base_height - pelvis[2])
        r -= W_SIDE * abs(side)
        d = action - self._prev_action
        r -= W_ACTION_RATE * float(np.dot(d, d))
        self._prev_action = action.copy()
        hip_lateral_dev = (self.data.qpos[self.leg_qpos_adr[HIP_LATERAL_IDX]]
                           - self.nominal_qpos[self.leg_qpos_adr[HIP_LATERAL_IDX]])
        r -= W_HIP_LATERAL * float(np.dot(hip_lateral_dev, hip_lateral_dev))

        LIFT_EPS = 0.005
        max_lift_steps = int(MAX_LIFT_DURATION / self.control_dt)
        left_lift = np.clip(self.data.xpos[self.left_foot_id, 2] - FOOT_REST_Z, 0.0, MAX_FOOT_LIFT)
        right_lift = np.clip(self.data.xpos[self.right_foot_id, 2] - FOOT_REST_Z, 0.0, MAX_FOOT_LIFT)
        self._left_lift_steps = self._left_lift_steps + 1 if left_lift > LIFT_EPS else 0
        self._right_lift_steps = self._right_lift_steps + 1 if right_lift > LIFT_EPS else 0
        # nagrada samo dok je noga podignuta unutar prirodnog trajanja zamaha -
        # predugo "visenje" u vazduhu se vise ne nagradjuje
        if self._left_lift_steps <= max_lift_steps:
            r += W_FOOT_LIFT * left_lift
        if self._right_lift_steps <= max_lift_steps:
            r += W_FOOT_LIFT * right_lift

        # direktna nagrada za savijanje kolena (uz istu vremensku granicu kao
        # i podizanje stopala - ne nagradjuje predugo drzanje savijenog kolena)
        knee_qpos = self.data.qpos[self.leg_qpos_adr[KNEE_IDX]] - self.nominal_qpos[self.leg_qpos_adr[KNEE_IDX]]
        knee_bend = np.clip(knee_qpos, 0.0, MAX_KNEE_BEND)
        if left_lift > LIFT_EPS and self._left_lift_steps <= max_lift_steps:
            r += W_KNEE_BEND * knee_bend[0]
        if right_lift > LIFT_EPS and self._right_lift_steps <= max_lift_steps:
            r += W_KNEE_BEND * knee_bend[1]

        left_dev = self.data.qpos[self.leg_qpos_adr[LEFT_IDX]] - self.nominal_qpos[self.leg_qpos_adr[LEFT_IDX]]
        right_dev = self.data.qpos[self.leg_qpos_adr[RIGHT_IDX]] - self.nominal_qpos[self.leg_qpos_adr[RIGHT_IDX]]
        asym = left_dev - right_dev
        r -= W_ASYMMETRY * float(np.dot(asym, asym))

        # bar jedno stopalo mora da dodiruje pod - ako su obe u vazduhu, to je
        # skok, ne hod
        any_foot_down = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = {g1, g2}
            if self.floor_geom_id in pair and (pair & self.foot_geom_ids):
                any_foot_down = True
                break
        if not any_foot_down:
            r -= W_AIRBORNE

        # stvarna kazna za prelazak preko cilja (ne samo izostanak nagrade)
        r -= W_OVERSHOOT * max(0.0, fwd - TARGET_DISTANCE)

        # kazna ako krene unazad
        if fwd < 0.0:
            r -= W_BACKWARD * abs(fwd)

        speed = float(np.linalg.norm(vel[:3]))
        in_zone = fwd >= TARGET_DISTANCE and not fell
        # kontinualan podsticaj da miruje cim je u ciljnoj zoni (ne samo
        # jednokratan bonus na kraju)
        if in_zone:
            r += W_STAND_STILL * max(0.0, 1.0 - speed / SETTLE_VEL_MAX)
        if in_zone and tilt <= SETTLE_TILT_MAX and speed <= SETTLE_VEL_MAX:
            self._settle_count += 1
        else:
            self._settle_count = 0
        success = in_zone and self._settle_count >= SETTLE_STEPS
        if success:
            r += SUCCESS_BONUS
        if fell:
            r -= FALL_PENALTY

        terminated = bool(fell or success)
        truncated = bool(self.step_count * self.control_dt >= EPISODE_SEC)
        info = {"fwd": fwd, "fell": fell, "success": success, "tilt": tilt}
        return self._get_obs(), float(r), terminated, truncated, info
