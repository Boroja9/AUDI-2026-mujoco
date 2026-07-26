#!/usr/bin/env python3
"""G1 baseline: bacanje lopte ispruzenom rukom napred.

Sve u jednom fajlu: kontroler + evaluacija + viewer.

Pokretanje (iz root foldera audi2026):
    python Boroja/throw.py            # evaluacija, 20 epizoda
    python Boroja/throw.py --play     # MuJoCo viewer
    python Boroja/throw.py --preset stable
    python Boroja/throw.py --episodes 50

Pokret:
    Ruka je ISPRUZENA (lakat ~0) celo vreme. Jedini zglob koji se krece je
    right_shoulder_pitch - ramenom se cela prava ruka rotira iz pocetnog
    polozaja unapred, i na kraju zamaha se lopta pusta.

Zasto originalni baselines/baseline_controller.py udara:
    Imao je right_shoulder_roll = +1.10 rad. Kod desne ruke G1 pozitivan
    roll je ADUKCIJA (ruka ide KA telu), pa je rame ulazilo u torso_link, a
    saka i lopta u right_hip_*_link. Ovde je roll drzan na -0.10 (blago
    odmaknuto od tela) i putanja je verifikovana MuJoCo kolizijama.
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv

# ---------------------------------------------------------------------------
# PODESAVANJA
# ---------------------------------------------------------------------------
# Env racuna:  ctrl = nominal_ctrl + ACTION_SCALE * action,  action u [-1, 1].
# Sa default 0.5 se poze zasicuju i ruka ne stigne do zadatog ugla.
ACTION_SCALE = 2.0

# Redosled zglobova desne ruke (env.arm_joint_names):
#   0 shoulder pitch  <- JEDINI koji se menja tokom zamaha
#   1 shoulder roll   <- negativno = ruka odmaknuta od tela
#   2 shoulder yaw
#   3 elbow           <- ~0 = ispruzena ruka
#   4 wrist roll
#   5 wrist pitch
#   6 wrist yaw
JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

PRESETS = {
    # Manji drift karlice (robot stabilniji), greska ~5 cm - i dalje duboko
    # unutar success radijusa od 18 cm.
    "stable": dict(
        pitch_start=0.5437,
        pitch_end=0.1312,
        roll=-0.10,
        elbow=-0.3525,
        swing_start=0.148,
        release_time=0.308,
    ),
    # Maksimalna preciznost (~0.5 cm), ali brzi zamah pa karlicu odgurne ~2 cm.
    "accurate": dict(
        pitch_start=0.2626,
        pitch_end=-0.4705,
        roll=-0.10,
        elbow=-0.111,
        swing_start=0.115,
        release_time=0.418,
    ),
}


# ---------------------------------------------------------------------------
# KONTROLER
# ---------------------------------------------------------------------------
def smoothstep(x):
    """Glatka S-kriva: nula brzine na oba kraja, bez trzaja."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class StraightArmThrow:
    """Open-loop: ispruzena ruka rotira ramenom unapred, pa pusti loptu."""

    def __init__(self, env, preset="stable", **overrides):
        cfg = dict(PRESETS[preset])
        cfg.update(overrides)
        self.cfg = cfg

        self.n_arm = env.n_arm
        self.t0 = float(cfg["swing_start"])
        self.trel = float(cfg["release_time"])
        if not 0.0 <= self.t0 < self.trel:
            raise ValueError("Mora vaziti 0 <= swing_start < release_time")

        roll, elbow = float(cfg["roll"]), float(cfg["elbow"])
        self.start_rad = np.array(
            [float(cfg["pitch_start"]), roll, 0.0, elbow, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.end_rad = np.array(
            [float(cfg["pitch_end"]), roll, 0.0, elbow, 0.0, 0.0, 0.0], dtype=np.float64
        )

        nominal = np.asarray(env.nominal_ctrl[env.arm_actuator_ids], dtype=np.float64)
        scale = float(env.action_scale)

        # Provera dostiznosti pre nego sto krene simulacija.
        self.warnings = []
        for i, name in enumerate(JOINT_NAMES[: self.n_arm]):
            lo_reach, hi_reach = nominal[i] - scale, nominal[i] + scale
            for tag, val in (("start", self.start_rad[i]), ("end", self.end_rad[i])):
                if not lo_reach - 1e-6 <= val <= hi_reach + 1e-6:
                    self.warnings.append(
                        f"{name} ({tag}) {val:+.4f} van dosega "
                        f"[{lo_reach:+.4f}, {hi_reach:+.4f}] - povecaj ACTION_SCALE"
                    )
                if not env.arm_joint_lower[i] <= val <= env.arm_joint_upper[i]:
                    self.warnings.append(
                        f"{name} ({tag}) {val:+.4f} van granica zgloba "
                        f"[{env.arm_joint_lower[i]:+.4f}, {env.arm_joint_upper[i]:+.4f}]"
                    )

        # radijani -> normalizovana akcija
        self.a_start = np.clip((self.start_rad - nominal) / scale, -1.0, 1.0)
        self.a_end = np.clip((self.end_rad - nominal) / scale, -1.0, 1.0)

    def act(self, t):
        """Akcija duzine n_arm + 1. Poslednji element gasi weld koji drzi loptu."""
        a = np.zeros(self.n_arm + 1, dtype=np.float32)
        n = min(self.n_arm, len(self.a_start))
        if t < self.t0:
            a[:n] = self.a_start[:n]
        else:
            p = smoothstep((float(t) - self.t0) / (self.trel - self.t0))
            a[:n] = self.a_start[:n] + (self.a_end[:n] - self.a_start[:n]) * p
        a[-1] = 1.0 if t >= self.trel else 0.0
        return a

    def summary(self):
        lines = [
            f"preset          : {self.cfg}",
            f"start (rad)     : {np.round(self.start_rad, 4).tolist()}",
            f"kraj  (rad)     : {np.round(self.end_rad, 4).tolist()}",
            f"zamah pocinje   : {self.t0:.3f} s",
            f"lopta se pusta  : {self.trel:.3f} s",
        ]
        lines += ["UPOZORENJE: " + w for w in self.warnings] or ["poze dostizne, bez klipovanja"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# POMOCNE
# ---------------------------------------------------------------------------
def make_env():
    return G1FixedBodyThrowEnv(learned_release=True, action_scale=ACTION_SCALE)


def self_contacts(env):
    """Kontakti robot-robot. Ignorise pod (body 0) i stopala."""
    m, d = env.model, env.data
    out = []
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
        if b1 == 0 or b2 == 0:
            continue
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
        if "ankle" in n1 or "ankle" in n2:
            continue
        out.append((n1, n2))
    return out


# ---------------------------------------------------------------------------
# EVALUACIJA
# ---------------------------------------------------------------------------
def evaluate(preset="stable", episodes=20, base_seed=42, verbose=True):
    env = make_env()
    ctrl = StraightArmThrow(env, preset)
    print(ctrl.summary())
    print()

    errors, rewards, releases, smooth, drifts = [], [], [], [], []
    successes = falls = 0
    collisions = {}

    for ep in range(episodes):
        env.reset(seed=base_seed + ep)
        pelvis0 = env.data.xpos[env.base_body_id].copy()
        prev = np.zeros(env.n_arm + 1)
        deltas, drift = [], 0.0
        done = False
        total = 0.0

        while not done:
            t = env.step_count * env.control_dt
            a = ctrl.act(t)
            _, r, term, trunc, info = env.step(a)
            total += r
            deltas.append(float(np.linalg.norm(a - prev)))
            prev = a.copy()
            drift = max(drift, float(np.linalg.norm(env.data.xpos[env.base_body_id][:2] - pelvis0[:2])))
            for pair in self_contacts(env):
                key = tuple(sorted(pair))
                collisions[key] = collisions.get(key, 0) + 1
            done = term or trunc

        rewards.append(total)
        smooth.append(float(np.mean(deltas)))
        drifts.append(drift)
        if info["landing_error"] is not None:
            errors.append(info["landing_error"])
        if info["release_time"] is not None:
            releases.append(info["release_time"])
        successes += int(info["success"])
        falls += int(info["robot_fell"])

        if verbose:
            le = info["landing_error"]
            txt = f"{le:.4f} m" if le is not None else "nije sleteo"
            print(f"ep {ep+1:3d}   greska = {txt:>14s}   reward = {total:7.2f}")

    print("\n" + "=" * 58)
    print(f"REZULTAT  (preset = {preset})")
    print("=" * 58)
    print(f"Epizoda                : {episodes}")
    print(f"Success rate           : {100*successes/episodes:.1f}%")
    print(f"Prosecna greska        : {np.mean(errors):.4f} m")
    print(f"Max greska             : {np.max(errors):.4f} m")
    print(f"Prosecan reward        : {np.mean(rewards):.2f}")
    print(f"Vreme pustanja         : {np.mean(releases):.3f} s")
    print(f"Padova robota          : {falls}")
    print(f"Max pomeraj karlice    : {np.max(drifts):.4f} m")
    print(f"Glatkoca (mean |da|)   : {np.mean(smooth):.4f}")
    if collisions:
        print("\nSELF-KOLIZIJE:")
        for (a, b), n in sorted(collisions.items(), key=lambda z: -z[1]):
            print(f"  {a} <-> {b}   ({n} koraka)")
    else:
        print("Self-kolizije          : nema")


# ---------------------------------------------------------------------------
# VIEWER
# ---------------------------------------------------------------------------
def play(preset="stable", seed=42):
    import time

    import mujoco.viewer

    env = make_env()
    ctrl = StraightArmThrow(env, preset)
    print(ctrl.summary())
    print("\nZatvori prozor ili Ctrl+C za izlaz.")

    env.reset(seed=seed)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            t = env.step_count * env.control_dt
            _, _, term, trunc, info = env.step(ctrl.act(t))
            viewer.sync()
            time.sleep(env.control_dt)
            if term or trunc:
                le = info["landing_error"]
                txt = f"{le:.4f} m" if le is not None else "nije sleteo"
                print(f"greska = {txt}   success = {info['success']}   pao = {info['robot_fell']}")
                time.sleep(0.6)
                env.reset(seed=seed)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="G1 baseline - bacanje ispruzenom rukom")
    ap.add_argument("--play", action="store_true", help="otvori MuJoCo viewer")
    ap.add_argument("--preset", default="stable", choices=sorted(PRESETS), help="izbor podesavanja")
    ap.add_argument("--episodes", type=int, default=20, help="broj epizoda za evaluaciju")
    ap.add_argument("--seed", type=int, default=42, help="pocetni seed")
    args = ap.parse_args()

    if args.play:
        play(args.preset, args.seed)
    else:
        evaluate(args.preset, args.episodes, args.seed)