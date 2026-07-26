#!/usr/bin/env python3
"""
Baseline: bacanje ispruzenom rukom napred (klatno oko ramena).
Ruka je ispruzena (lakat = 0), rame rotira iz pocetnog polozaja ka napred/gore,
lopta se otpusta tokom zamaha. Robot ostaje stabilan (pelvis weld + spor pokret).

Koriscenje:
    python3 baseline_forward_throw.py --view              # jedna epizoda, prozor
    python3 baseline_forward_throw.py --runs 10           # benchmark, brojevi
    python3 baseline_forward_throw.py --view --pitch-end -1.6 --rel-frac 0.75
"""
import argparse, csv, time
import numpy as np
import mujoco, mujoco.viewer

XML = 'assets/unitree_g1/scene_throw.xml'   # scena SA loptom (ne scene.xml!)

ARM_ACT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
    "right_wrist_roll_joint",     "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
BALL_R, HIT_R = 0.04, 0.18   # poluprecnik lopte, tolerancija pogotka [m]

# ---------------- parametri pokreta (default, menjaju se i preko CLI) ----------------
DEF = dict(
    t_start   = 0.40,   # kad zamah krece [s]
    t_swing   = 0.80,   # trajanje zamaha [s] (sporije = stabilnije)
    rel_frac  = 0.70,   # otpustanje na X% zamaha
    pitch_end = -1.60,  # krajnji ugao ramena [rad]; -1.57 = ruka horizontalno napred
    roll_hold = -0.35,  # rame roll: ruka blago odmaknuta od tela (izbegava koliziju)
    t_total   = 3.50,
)

def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return 3*u**2 - 2*u**3

def build():
    model = mujoco.MjModel.from_xml_path(XML)
    data  = mujoco.MjData(model)
    def bid(n):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    def eqid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, n)
    ids = dict(ball=bid('throw_ball'), target=bid('throw_target'),
               pelvis=bid('pelvis'), torso=bid('torso_link'),
               ball_weld=eqid('hold_throw_ball'), pelvis_weld=eqid('pelvis_weld'),
               act=[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                    for n in ARM_ACT_NAMES])
    for k in ('ball', 'target', 'ball_weld'):
        if ids[k] < 0:
            raise RuntimeError(f"'{k}' ne postoji u {XML} - proveri scenu/imena.")
    if any(a < 0 for a in ids['act']):
        raise RuntimeError("Aktuator desne ruke nije nadjen - proveri imena.")
    return model, data, ids

def episode(model, data, ids, cfg, viewer=None, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.eq_active[ids['ball_weld']] = 1
    if ids['pelvis_weld'] >= 0:
        data.eq_active[ids['pelvis_weld']] = 1        # stabilnost: baza fiksirana
    if noise > 0:
        data.qpos[7:] += rng.normal(0, noise, size=data.qpos[7:].shape)
    mujoco.mj_forward(model, data)

    nominal = model.key_ctrl[0].copy()
    act = ids['act']
    a0 = nominal[act].copy()
    # ciljna poza: ispruzena ruka napred, blago odmaknuta od tela
    a1 = a0.copy()
    a1[0] = cfg['pitch_end']    # rame pitch: rotacija napred/gore
    a1[1] = cfg['roll_hold']    # rame roll: drzi ruku van tela
    a1[3] = 0.0                 # lakat ISPRUZEN celo vreme
    # i pocetna poza dobija roll odmak + ispruzen lakat (da ne struze o butinu)
    a0[1] = cfg['roll_hold']
    a0[3] = 0.0

    t_rel = cfg['t_start'] + cfg['rel_frac']*cfg['t_swing']
    released = False
    landed, land_xy = False, np.array([np.nan, np.nan])
    min_pelv, min_up = 10.0, 1.0

    while data.time < cfg['t_total'] and (viewer is None or viewer.is_running()):
        data.ctrl[:] = nominal
        if data.time < cfg['t_start']:
            s = smoothstep(data.time / cfg['t_start'])   # meki ulazak u pocetnu pozu
            data.ctrl[act] = nominal[act] + s*(a0 - nominal[act])
        else:
            s = smoothstep((data.time - cfg['t_start']) / cfg['t_swing'])
            data.ctrl[act] = a0 + s*(a1 - a0)
        if not released and data.time >= t_rel:
            data.eq_active[ids['ball_weld']] = 0
            released = True
            if viewer: print(f"[{data.time:.2f}s] Lopta pustena")

        mujoco.mj_step(model, data)

        bp = data.xpos[ids['ball']]
        if released and not landed and bp[2] <= BALL_R + 0.005:
            landed, land_xy = True, bp[:2].copy()
        min_pelv = min(min_pelv, data.xpos[ids['pelvis']][2])
        min_up   = min(min_up, data.xmat[ids['torso']].reshape(3,3)[2,2])

        if viewer:
            viewer.sync()
            time.sleep(model.opt.timestep)

    tgt = data.xpos[ids['target']][:2]
    ref = land_xy if landed else data.xpos[ids['ball']][:2]
    dist = float(np.linalg.norm(ref - tgt))
    tilt = float(np.degrees(np.arccos(np.clip(min_up, -1, 1))))
    return dict(dist_cm=100*dist, hit=int(dist < HIT_R),
                land_x=float(ref[0]), land_y=float(ref[1]),
                pelvis_min=round(min_pelv,3), tilt_deg=round(tilt,1),
                stable=int(min_pelv > 0.55 and tilt < 30.0), landed=int(landed))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--view', action='store_true')
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--noise', type=float, default=0.01)
    p.add_argument('--csv', default='baseline_results.csv')
    for k, v in DEF.items():
        p.add_argument(f"--{k.replace('_','-')}", type=float, default=v)
    args = p.parse_args()
    cfg = {k: getattr(args, k) for k in DEF}

    model, data, ids = build()

    if args.view:
        with mujoco.viewer.launch_passive(model, data) as v:
            v.cam.lookat[:] = [0.30, -0.15, 0.60]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.6, 135, -12
            r = episode(model, data, ids, cfg, viewer=v)
        print("\n=== EPIZODA ===")
        for k, val in r.items(): print(f"  {k}: {val}")
        return

    rows = [episode(model, data, ids, cfg, noise=args.noise, seed=i)
            for i in range(args.runs)]
    d = np.array([r['dist_cm'] for r in rows])
    print(f"\n=== BASELINE ({args.runs} epizoda) ===")
    print(f"Rastojanje: mean {d.mean():.1f} cm | min {d.min():.1f} | max {d.max():.1f} | std {d.std():.1f}")
    print(f"Hit rate (<{HIT_R*100:.0f} cm): {100*np.mean([r['hit'] for r in rows]):.0f} %")
    print(f"Stabilnost: {100*np.mean([r['stable'] for r in rows]):.0f} %")
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"CSV: {args.csv}")

if __name__ == '__main__':
    main()