"""
Baseline B — kontrolisano izbacivanje lopte u zonu ispred robota. Unitree G1, MuJoCo.
Radi samostalno: sam nalazi G1 model, sam napravi scenu, sam se kalibrise.

    python drop_baseline.py                  # 20 epizoda, bez prozora
    python drop_baseline.py --render         # sa prozorom
    python drop_baseline.py --episodes 50 --csv rezultati.csv --plot scatter.png

Ako model nije nadjen automatski:
    python drop_baseline.py --robot /putanja/do/g1_29dof.xml

Nema ucenja. Ruka izvede zamah (rame vodi, lakat kasni), lopta se pusti kada
rame prodje ugao otpustanja. Fizika je namestena za blisku zonu: iz visine
~1 m lopta leti 0.45 s, pa za 0.7 m treba samo ~1.5 m/s — zato je zamah SPOR.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

import numpy as np
import mujoco
import mujoco.viewer  # mora na vrhu fajla — unutar funkcije bi zasenio ime "mujoco"


# =====================================================================================
# 1. NADJI ROBOTA — trazi g1 MJCF po uobicajenim lokacijama
# =====================================================================================

def find_robot(explicit: str | None) -> str:
    if explicit:
        p = os.path.expanduser(explicit)
        if os.path.isfile(p):
            return os.path.abspath(p)
        sys.exit(f"[greska] --robot pokazuje na nepostojeci fajl: {p}")

    patterns = [
        "g1_29dof*.xml", "g1*.xml",
        "**/unitree_g1/g1_29dof*.xml", "**/unitree_g1/*.xml",
        "**/g1_29dof*.xml",
    ]
    roots = [
        os.getcwd(),
        os.path.join(os.getcwd(), ".."),
        os.path.expanduser("~/mujoco_menagerie"),
        os.path.expanduser("~"),
    ]
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for pat in patterns:
            for hit in glob.glob(os.path.join(root, pat), recursive=True):
                hit = os.path.abspath(hit)
                if hit in seen or "_scene_" in os.path.basename(hit):
                    continue
                seen.add(hit)
                # preskoci scene/keyframe varijante, hocemo goli model
                base = os.path.basename(hit).lower()
                if "scene" in base:
                    continue
                print(f"[info] koristim model: {hit}")
                return hit
    sys.exit(
        "[greska] G1 model nije nadjen. Skini menagerie:\n"
        "  git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie\n"
        "ili prosledi putanju:  python drop_baseline.py --robot /putanja/g1_29dof.xml"
    )


# =====================================================================================
# 2. SCENA — robot + lopta + zona, u temp fajlu pored modela (zbog relativnih putanja)
# =====================================================================================

def build_scene(robot_path: str, dt: float, ball_r: float, ball_m: float,
                tx: float, ty: float, th: float) -> str:
    xml = f"""
<mujoco model="g1_drop_baseline">
  <include file="{os.path.basename(robot_path)}"/>
  <option timestep="{dt}" integrator="implicitfast"/>
  <worldbody>
    <body name="bb_ball" pos="0.35 -0.25 1.2">
      <freejoint name="bb_ball_free"/>
      <geom name="bb_ball_geom" type="sphere" size="{ball_r}" mass="{ball_m}"
            rgba="0.78 0.06 0.18 1" friction="0.8 0.01 0.001"
            solref="0.005 1" condim="4"/>
    </body>
    <site name="bb_zone" type="box" pos="{tx} {ty} 0.001"
          size="{th} {th} 0.001" rgba="0.78 0.06 0.18 0.35"/>
  </worldbody>
</mujoco>
"""
    out = os.path.join(os.path.dirname(robot_path), "_scene_drop_baseline.xml")
    with open(out, "w") as f:
        f.write(xml)
    return out


# =====================================================================================
# 3. AUTODETEKCIJA ZGLOBOVA — bez rucnog upisivanja imena
# =====================================================================================

def _find_joint(model, keywords_all, prefer=("right", "r_", "_r")):
    """Vrati id zgloba cije ime sadrzi sve reci iz keywords_all, uz prednost desnoj strani."""
    cands = []
    for j in range(model.njnt):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").lower()
        if all(k in name for k in keywords_all):
            score = sum(p in name for p in prefer)
            cands.append((score, j, name))
    if not cands:
        return -1, ""
    cands.sort(reverse=True)
    return cands[0][1], cands[0][2]


def detect_arm(model):
    """Nadji rame-pitch, rame-roll i lakat desne ruke po imenu; ispisi sta je nadjeno."""
    picks = {}
    sp, spn = _find_joint(model, ["shoulder", "pitch"])
    sr, srn = _find_joint(model, ["shoulder", "roll"])
    el, eln = _find_joint(model, ["elbow"])
    if el < 0:
        el, eln = _find_joint(model, ["elbow", "pitch"])
    for key, (jid, nm) in dict(shoulder_pitch=(sp, spn),
                               shoulder_roll=(sr, srn),
                               elbow=(el, eln)).items():
        if jid < 0:
            print("\n[greska] nisam nasao zglob:", key)
            print("Lista zglobova u modelu:")
            for j in range(model.njnt):
                print("   ", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
            sys.exit(1)
        picks[key] = jid
        print(f"[info] {key:15s} -> {nm}")
    return picks


def detect_hand_body(model):
    """Zglob/dlan desne ruke — NE prst. Prsti su sitni linkovi koji numericki
    vibriraju, pa brzina ocitana sa njih bude djubre (npr. 15 m/s)."""
    best, best_score = -1, -1e9
    for b in range(model.nbody):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").lower()
        if not name:
            continue
        score = 0
        for kw, s in (("wrist", 6), ("palm", 6), ("hand", 3), ("elbow", 1)):
            if kw in name:
                score += s
        for kw in ("thumb", "finger", "index", "middle", "ring", "pinky", "little"):
            if kw in name:
                score -= 20
        if not any(p in name for p in ("right", "r_")):
            score -= 10
        if score > best_score:
            best, best_score = b, score
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, best)
    print(f"[info] saka           -> {name}")
    return best


def actuator_for(model, jid):
    for a in range(model.nu):
        if model.actuator_trnid[a, 0] == jid:
            return a
    return -1


# =====================================================================================
# 4. KONTROLER — cetiri faze, otpustanje po stanju
# =====================================================================================

def min_jerk(t, T):
    if T <= 0:
        return 1.0
    s = float(np.clip(t / T, 0.0, 1.0))
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class DropController:
    # poze ruke [rame_pitch, rame_roll, lakat] — PUN bejzbol zamah preko ramena.
    # Brza ruka + bliska zona se mire PREDIKTIVNIM otpustanjem: na svakom koraku
    # se racuna gde bi lopta pala ako se pusti sada, i pusta se u trenutku kada
    # predikcija padne na zonu. Tajming radi posao, ne usporavanje zamaha.
    POSE_READY = np.array([0.35, -0.15, 1.30])
    POSE_BACK = np.array([1.55, -0.25, 2.05])
    POSE_FRONT = np.array([-1.15, -0.10, 0.30])

    def __init__(self, model, data, arm, hand_body, cfg):
        self.m, self.d, self.c = model, data, cfg
        self.hand = hand_body

        self.jids = [arm["shoulder_pitch"], arm["shoulder_roll"], arm["elbow"]]
        self.qadr = [model.jnt_qposadr[j] for j in self.jids]
        self.dadr = [model.jnt_dofadr[j] for j in self.jids]
        self.acts = [actuator_for(model, j) for j in self.jids]
        if any(a < 0 for a in self.acts):
            sys.exit("[greska] zglob ruke nema aktuator — proveri model")

        self.hold = []
        for a in range(model.nu):
            j = model.actuator_trnid[a, 0]
            if j in self.jids:
                continue
            self.hold.append((a, model.jnt_qposadr[j], model.jnt_dofadr[j]))

        bj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bb_ball_free")
        self.bq = model.jnt_qposadr[bj]
        self.bv = model.jnt_dofadr[bj]
        self.ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bb_ball_geom")
        # dok je lopta u saci, iskljuci njene kontakte — inace se sudara sa
        # prstima mesh-a u koje je zakacena i sve brzine postanu djubre
        self._ball_ct = (model.geom_contype[self.ball_geom],
                         model.geom_conaffinity[self.ball_geom])
        model.geom_contype[self.ball_geom] = 0
        model.geom_conaffinity[self.ball_geom] = 0

        # torzo za merenje nagiba: prvo telo posle korena
        self.torso = min(2, model.nbody - 1)

        self.reset()

    def reset(self):
        self.t = 0.0
        self.released = False
        self.v_release = 0.0
        self.landing = None
        self.max_tilt = 0.0
        self.hold_targets = None
        self.prev_pitch = None
        self.peak_pitch = -1e9
        self.prev_pin = None      # prosla pozicija lopte u saci (za cistu brzinu)
        self.v_hand = np.zeros(3)
        self.best_pred = float("inf")
        self.pred_err = float("inf")

    # ---- lopta u saci ----------------------------------------------------------

    def pin_pos(self):
        pos = self.d.xpos[self.hand].copy()
        rot = self.d.xmat[self.hand].reshape(3, 3)
        return pos + rot @ np.array([0.05, 0.0, -0.03])

    def pin_ball(self):
        """Lopta prati saku. Brzina = konacna razlika pozicije koju MI zadajemo,
        pa je cista — mj_objectVelocity na mesh linku vraca kontaktni sum."""
        pos = self.pin_pos()
        if self.prev_pin is not None:
            self.v_hand = (pos - self.prev_pin) / self.m.opt.timestep
        self.prev_pin = pos
        self.d.qpos[self.bq:self.bq + 3] = pos
        self.d.qpos[self.bq + 3:self.bq + 7] = [1, 0, 0, 0]
        self.d.qvel[self.bv:self.bv + 3] = self.v_hand
        self.d.qvel[self.bv + 3:self.bv + 6] = 0.0

    def predict_landing(self):
        """Balisticka predikcija: gde pada lopta ako se pusti OVOG koraka."""
        p = self.d.qpos[self.bq:self.bq + 3]
        v = self.v_hand
        z = p[2] - self.c["ball_r"]
        if z <= 0:
            return None
        g = 9.81
        disc = v[2] * v[2] + 2 * g * z
        t_fly = (v[2] + math.sqrt(disc)) / g
        return np.array([p[0] + v[0] * t_fly, p[1] + v[1] * t_fly])

    # ---- trajektorija ----------------------------------------------------------

    def arm_target(self):
        c = self.c
        t, t0 = self.t, c["t_settle"]
        t1 = t0 + c["t_windup"]
        t2 = t1 + c["t_swing"]
        if t < t0:
            return self.POSE_READY.copy()
        if t < t1:
            s = min_jerk(t - t0, c["t_windup"])
            return self.POSE_READY + s * (self.POSE_BACK - self.POSE_READY)
        if t < t2:
            s = min_jerk(t - t1, c["t_swing"])
            se = min_jerk(t - t1 - 0.3 * c["t_swing"], 0.7 * c["t_swing"])  # lakat kasni
            tgt = self.POSE_BACK + s * (self.POSE_FRONT - self.POSE_BACK)
            tgt[2] = self.POSE_BACK[2] + se * (self.POSE_FRONT[2] - self.POSE_BACK[2])
            return tgt
        s = min_jerk(t - t2, c["t_follow"])
        return self.POSE_FRONT + s * (self.POSE_READY - self.POSE_FRONT)

    def check_release(self):
        """Prediktivno: pusti u koraku u kome predvidjeno mesto pada najbolje
        lezi na zonu. Prati se minimum greske; kad greska prodje minimum i
        krene da raste — otpustanje je bilo bas tu."""
        in_swing = self.t >= self.c["t_settle"] + self.c["t_windup"]
        if not in_swing or self.prev_pin is None:
            return False
        pred = self.predict_landing()
        if pred is None:
            return False
        err = math.hypot(pred[0] - self.c["tx"], pred[1] - self.c["ty"])
        self.pred_err = err
        if err < self.best_pred:
            self.best_pred = err
            return err < 0.02                    # dovoljno dobro — pusti odmah
        # greska raste, a minimum je bio unutar zone -> pusti sada
        return self.best_pred < self.c["th"]

    # ---- korak -----------------------------------------------------------------

    def step(self):
        c = self.c
        if self.hold_targets is None:
            self.hold_targets = {a: self.d.qpos[q] for a, q, _ in self.hold}

        for a, q, dv in self.hold:
            e = self.hold_targets[a] - self.d.qpos[q]
            self.d.ctrl[a] = c["kp_hold"] * e - c["kd_hold"] * self.d.qvel[dv]

        tgt = self.arm_target()
        for k in range(3):
            e = tgt[k] - self.d.qpos[self.qadr[k]]
            grav = self.d.qfrc_bias[self.dadr[k]]   # kompenzacija gravitacije/Coriolisa
            self.d.ctrl[self.acts[k]] = (c["kp_arm"] * e
                                         - c["kd_arm"] * self.d.qvel[self.dadr[k]]
                                         + grav)

        if not self.released:
            self.pin_ball()                       # prvo azuriraj v_hand za ovaj korak
            if self.check_release():
                self.released = True
                self.v_release = float(np.linalg.norm(self.v_hand))
                # vrati kontakte lopti da moze da padne na pod
                self.m.geom_contype[self.ball_geom] = self._ball_ct[0]
                self.m.geom_conaffinity[self.ball_geom] = self._ball_ct[1]
                self.d.qvel[self.bv:self.bv + 3] = self.v_hand

        mujoco.mj_step(self.m, self.d)
        self.t += self.m.opt.timestep

        z = self.d.xmat[self.torso].reshape(3, 3)[:, 2]
        self.max_tilt = max(self.max_tilt, math.acos(float(np.clip(z[2], -1, 1))))

        if self.released and self.landing is None:
            if self.d.qpos[self.bq + 2] <= c["ball_r"] * 1.05:
                self.landing = self.d.qpos[self.bq:self.bq + 2].copy()


# =====================================================================================
# 5. EPIZODE, KALIBRACIJA, METRIKE
# =====================================================================================

def run_episode(model, data, arm, hand, cfg, seed=0, jitter=0.0, viewer=None):
    mujoco.mj_resetData(model, data)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    if jitter > 0:
        rng = np.random.default_rng(seed)
        data.qpos[7:] += rng.normal(0, jitter, size=model.nq - 7)
    mujoco.mj_forward(model, data)

    ctl = DropController(model, data, arm, hand, cfg)
    horizon = cfg["t_settle"] + cfg["t_windup"] + cfg["t_swing"] + cfg["t_follow"] + 1.5
    for _ in range(int(horizon / model.opt.timestep)):
        ctl.step()
        if viewer is not None:
            if not viewer.is_running():
                sys.exit(0)
            viewer.sync()
        if ctl.landing is not None:
            break

    fell = ctl.max_tilt > cfg["max_tilt"]
    if ctl.landing is None:
        return dict(seed=seed, landed=False, x=np.nan, y=np.nan, err=np.nan,
                    success=False, fell=fell, tilt=ctl.max_tilt, v_rel=ctl.v_release)
    x, y = float(ctl.landing[0]), float(ctl.landing[1])
    err = math.hypot(x - cfg["tx"], y - cfg["ty"])
    hit = abs(x - cfg["tx"]) <= cfg["th"] and abs(y - cfg["ty"]) <= cfg["th"]
    return dict(seed=seed, landed=True, x=x, y=y, err=err,
                success=bool(hit and not fell), fell=fell,
                tilt=ctl.max_tilt, v_rel=ctl.v_release)


def calibrate(model, data, arm, hand, cfg):
    """Grid pretraga po t_swing — bira tacku sa najmanjom greskom dometa.

    Odnos zamah->domet nije glatko monoton (otpustanje zavisi od dinamike),
    pa je gruba mreza + izbor najboljeg pouzdanija od bisekcije.
    """
    print(f"\n[kalibracija] trazim t_swing za zonu na {cfg['tx']:.2f} m")
    grid = np.linspace(0.30, 1.60, 14)
    best_t, best_err = cfg["t_swing"], float("inf")
    for i, ts in enumerate(grid):
        c = dict(cfg, t_swing=float(ts))
        xs = []
        for s in range(2):
            r = run_episode(model, data, arm, hand, c, seed=s)
            if r["landed"]:
                xs.append(r["x"])
        if not xs:
            print(f"  [{i:2d}] t_swing={ts:.3f}  lopta nije sletela")
            continue
        mx = float(np.mean(xs))
        err = abs(mx - cfg["tx"])
        mark = " <-- najbolje" if err < best_err else ""
        print(f"  [{i:2d}] t_swing={ts:.3f}  domet={mx:.3f} m  |greska|={err:.3f}{mark}")
        if err < best_err:
            best_t, best_err = float(ts), err
    # fino oko najbolje tacke
    for ts in np.linspace(best_t - 0.06, best_t + 0.06, 5):
        if ts <= 0.1:
            continue
        c = dict(cfg, t_swing=float(ts))
        xs = [run_episode(model, data, arm, hand, c, seed=s)["x"]
              for s in range(2)
              if run_episode(model, data, arm, hand, c, seed=s)["landed"]]
        xs = [x for x in xs if not math.isnan(x)]
        if not xs:
            continue
        err = abs(float(np.mean(xs)) - cfg["tx"])
        if err < best_err:
            best_t, best_err = float(ts), err
    print(f"[kalibracija] t_swing = {best_t:.3f} s  (|greska dometa| = {best_err:.3f} m)\n")
    return best_t


def summarise(rows, cfg):
    n = len(rows)
    landed = [r for r in rows if r["landed"]]
    errs = np.array([r["err"] for r in landed]) if landed else np.array([np.nan])
    print("\n" + "=" * 60)
    print(f"BASELINE B — izbacivanje u zonu   ({n} epizoda)")
    print("=" * 60)
    print(f"  zona                  {cfg['tx']:.2f} m ± {cfg['th']:.2f} m")
    print(f"  uspesnost             {100*sum(r['success'] for r in rows)/n:5.1f} %")
    print(f"  sletela uopste        {100*len(landed)/n:5.1f} %")
    print(f"  srednja greska        {np.nanmean(errs):6.3f} m")
    print(f"  std greske            {np.nanstd(errs):6.3f} m")
    print(f"  padovi                {sum(r['fell'] for r in rows)} / {n}")
    print(f"  max nagib torza       {max(r['tilt'] for r in rows):6.3f} rad")
    print(f"  srednja v otpustanja  {np.nanmean([r['v_rel'] for r in rows]):5.2f} m/s")
    v_req = cfg["tx"] / math.sqrt(2 * 1.0 / 9.81)
    print(f"  potrebna v (~1m vis.) {v_req:5.2f} m/s")
    print("=" * 60 + "\n")


# =====================================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=None, help="putanja do G1 MJCF (inace trazim sam)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="grid podesavanje t_swing (obicno nepotrebno — otpustanje je prediktivno)")
    ap.add_argument("--jitter", type=float, default=0.005)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--target_x", type=float, default=0.70)
    args = ap.parse_args()

    cfg = dict(
        t_settle=0.6, t_windup=0.40, t_swing=0.30, t_follow=0.5,
        rel_drop=0.35,
        kp_arm=60.0, kd_arm=5.0, kp_hold=200.0, kd_hold=10.0,
        ball_r=0.02, tx=args.target_x, ty=0.0, th=0.15, max_tilt=0.5,
    )

    robot = find_robot(args.robot)
    scene = build_scene(robot, dt=0.002, ball_r=cfg["ball_r"], ball_m=0.05,
                        tx=cfg["tx"], ty=cfg["ty"], th=cfg["th"])
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)

    arm = detect_arm(model)
    hand = detect_hand_body(model)

    if args.calibrate:
        cfg["t_swing"] = calibrate(model, data, arm, hand, cfg)

    viewer = mujoco.viewer.launch_passive(model, data) if args.render else None

    rows = []
    for ep in range(args.episodes):
        r = run_episode(model, data, arm, hand, cfg, seed=ep,
                        jitter=args.jitter, viewer=viewer)
        rows.append(r)
        print(f"  ep {ep:3d}  {'POGODAK' if r['success'] else 'promasaj'}"
              f"  x={r['x']:6.3f}  y={r['y']:6.3f}  err={r['err']:6.3f}"
              f"  v={r['v_rel']:4.2f}  nagib={r['tilt']:.3f}")

    summarise(rows, cfg)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("[info] upisano:", args.csv)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        th = cfg["th"]
        ax.add_patch(plt.Rectangle((cfg["tx"] - th, cfg["ty"] - th), 2 * th, 2 * th,
                                   fill=False, lw=1.5, ec="#C8102E"))
        hit = [r for r in rows if r["success"]]
        mis = [r for r in rows if r["landed"] and not r["success"]]
        ax.scatter([r["x"] for r in mis], [r["y"] for r in mis], s=22, c="#98A2AC", label="promasaj")
        ax.scatter([r["x"] for r in hit], [r["y"] for r in hit], s=22, c="#C8102E", label="pogodak")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(f"Baseline B — n={len(rows)}")
        ax.set_aspect("equal"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(args.plot, dpi=150)
        print("[info] upisano:", args.plot)

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
