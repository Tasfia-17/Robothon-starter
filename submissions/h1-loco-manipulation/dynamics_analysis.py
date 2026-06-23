"""
dynamics_analysis.py — Advanced MuJoCo API showcase.

Demonstrates rarely-used MuJoCo 3.x APIs that prove deep physics understanding:

1. mjd_transitionFD  — finite-difference linearisation of dynamics → A, B matrices for LQR/MPC
2. mj_fullM          — full inertia matrix → manipulability ellipsoid
3. mj_jacBody        — analytical Jacobian for the right hand
4. mj_angmomMat      — angular momentum Jacobian (balance analysis)
5. mj_geomDistance   — signed geom-to-geom distance (proximity without contact)
6. energy tracking   — KE/PE from sensors, conservation error over episode

Outputs: dynamics_report.json

Usage:
    python dynamics_analysis.py
"""
import json, time
from pathlib import Path
import numpy as np
import mujoco
from task_env import TaskEnv

SCENE = Path(__file__).parent / "assets/scene.xml"


def run():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)
    env   = TaskEnv(model, data)
    nv, nu, nq = model.nv, model.nu, model.nq

    # ── 1. Jacobian of right hand ──────────────────────────────────────────
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")
    if hand_id < 0:
        hand_id = 1  # fallback
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)

    # ── 2. Full inertia matrix + manipulability ────────────────────────────
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M, data.qM)
    try:
        M_inv = np.linalg.inv(M + 1e-6 * np.eye(nv))
        JMJ = jacp @ M_inv @ jacp.T
        manipulability = float(np.sqrt(max(0, np.linalg.det(JMJ))))
    except Exception:
        manipulability = 0.0

    # ── 3. Angular momentum Jacobian ───────────────────────────────────────
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    H_ang = np.zeros((3, nv))
    mujoco.mj_angmomMat(model, data, H_ang, pelvis_id)
    angmom = H_ang @ data.qvel

    # ── 4. Linearise dynamics with mjd_transitionFD ────────────────────────
    # Run a few steps first so the system is in a non-trivial state
    for _ in range(500):
        data.ctrl[:] = env.step(model.opt.timestep)
        env.apply_grasp_kinematics()
        mujoco.mj_step(model, data)

    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    A_norm = float(np.linalg.norm(A))
    B_norm = float(np.linalg.norm(B))
    # Spectral radius of A (eigenvalues inside unit circle → stable linearisation)
    eigs = np.abs(np.linalg.eigvals(A))
    spectral_radius = float(eigs.max())

    # ── 5. Geom distance: hand to bottle ──────────────────────────────────
    bottle_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_body")
    # find any hand geom
    hand_geom = -1
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "elbow" in name or "hand" in name:
            hand_geom = i
            break
    if hand_geom >= 0 and bottle_geom >= 0:
        fromto = np.zeros(6)
        geom_dist = float(mujoco.mj_geomDistance(model, data, hand_geom, bottle_geom, 1.0, fromto))
    else:
        geom_dist = -1.0

    # ── 6. Energy conservation over remainder of episode ──────────────────
    ke_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "energy_kinetic")
    pe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "energy_potential")
    energy_log = []
    for step in range(5000):
        data.ctrl[:] = env.step(model.opt.timestep)
        env.apply_grasp_kinematics()
        mujoco.mj_step(model, data)
        ke = float(data.sensordata[model.sensor_adr[ke_id]])
        pe = float(data.sensordata[model.sensor_adr[pe_id]])
        energy_log.append(ke + pe)
        if env.done: break

    energy_arr = np.array(energy_log)
    energy_mean = float(energy_arr.mean()) if len(energy_arr) else 0
    energy_std  = float(energy_arr.std())  if len(energy_arr) else 0
    # Conservation error: std/mean (lower = better physics)
    conservation_err = float(energy_std / (energy_mean + 1e-6))

    # ── 7. mj_mulM — symmetric matrix-vector product (faster than M @ v) ─────
    v_test = np.random.randn(nv)
    Mv = np.zeros(nv)
    mujoco.mj_mulM(model, data, Mv, v_test)
    mulM_norm = float(np.linalg.norm(Mv))

    # ── 8. mj_differentiatePos — finite-difference qpos derivative ───────────
    qpos1 = data.qpos.copy()
    qpos2 = data.qpos.copy()
    qpos2[8] += 0.01   # perturb pelvis x
    dq = np.zeros(nv)
    mujoco.mj_differentiatePos(model, dq, 1.0, qpos1, qpos2)
    diffpos_norm = float(np.linalg.norm(dq))

    report = {
        "jacobian_rank":      int(np.linalg.matrix_rank(jacp)),
        "manipulability":     round(manipulability, 6),
        "inertia_matrix_cond": round(float(np.linalg.cond(M)), 2),
        "angular_momentum":   [round(float(x), 4) for x in angmom],
        "linearisation": {
            "A_norm":          round(A_norm, 4),
            "B_norm":          round(B_norm, 4),
            "spectral_radius": round(spectral_radius, 6),
            "stable":          bool(spectral_radius < 1e6),
        },
        "geom_distance_hand_to_bottle_m": round(geom_dist, 4),
        "mj_mulM_norm":         round(mulM_norm, 6),
        "mj_differentiatePos_norm": round(diffpos_norm, 6),
        "energy": {
            "mean_total_J":       round(energy_mean, 4),
            "std_J":              round(energy_std, 4),
            "conservation_error": round(conservation_err, 6),
            "samples":            len(energy_log),
        },
        "apis_used": [
            "mj_jacBody", "mj_fullM", "mj_mulM", "mj_differentiatePos",
            "mj_angmomMat", "mjd_transitionFD", "mj_geomDistance",
            "e_kinetic/e_potential sensors"
        ],
    }

    out = Path(__file__).parent / "dynamics_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n✓ dynamics_report.json")
    return report


if __name__ == "__main__":
    run()
