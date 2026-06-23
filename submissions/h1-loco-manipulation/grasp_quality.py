"""
grasp_quality.py — Grasp quality metrics for H1 Loco-Manipulation.

Two metrics:
  1. epsilon_quality — Ferrari-Canny epsilon metric via convex hull of
     contact wrenches (standard in dexterous manipulation research).
     Returns the radius of the largest inscribed ball in wrench space.
     Higher = more robust grasp (force-closure measure).

  2. grasp_isotropy — Direction isotropy of contact normals around
     object centroid. Range [0,1]; 1 = perfectly symmetric enclosure.
     Fast proxy suitable for real-time reward computation.

Usage:
    from grasp_quality import epsilon_quality, grasp_isotropy
    eps = epsilon_quality(model, data, "bottle")
    iso = grasp_isotropy(model, data, "bottle")
"""
import numpy as np
import mujoco

try:
    from scipy.spatial import ConvexHull
    _SCIPY = True
except ImportError:
    _SCIPY = False


def epsilon_quality(model: mujoco.MjModel, data: mujoco.MjData,
                    object_body: str, mu: float = 1.0,
                    n_cone_edges: int = 8) -> float:
    """
    Ferrari-Canny epsilon grasp quality.

    Linearises the friction cone at each contact point into n_cone_edges
    force primitives, builds the 6-D Grasp Wrench Space (GWS) convex
    hull, and returns the distance from the origin to the nearest facet.

    Requires scipy. Returns 0.0 if scipy is unavailable or <4 contacts.
    """
    if not _SCIPY:
        return 0.0

    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if obj_id < 0:
        return 0.0

    obj_com = data.xipos[obj_id].copy()
    wrenches = []
    angles = np.linspace(0, 2 * np.pi, n_cone_edges, endpoint=False)

    for i in range(data.ncon):
        c = data.contact[i]
        if model.geom_bodyid[c.geom1] != obj_id and model.geom_bodyid[c.geom2] != obj_id:
            continue

        frame  = c.frame.reshape(3, 3)
        normal = frame[0].copy()
        t1, t2 = frame[1].copy(), frame[2].copy()

        # Normal should point into the object
        if model.geom_bodyid[c.geom2] == obj_id:
            normal = -normal

        r = c.pos - obj_com

        for angle in angles:
            f = normal + mu * (np.cos(angle) * t1 + np.sin(angle) * t2)
            f /= (np.linalg.norm(f) + 1e-12)
            wrench = np.concatenate([f, np.cross(r, f)])
            wrenches.append(wrench)

    if len(wrenches) < 4:
        return 0.0

    W = np.array(wrenches)
    try:
        hull = ConvexHull(W)
        # Distance from origin to each facet: |offset| / ||normal||
        dists = np.abs(hull.equations[:, -1]) / (
            np.linalg.norm(hull.equations[:, :-1], axis=1) + 1e-12)
        return float(np.min(dists))
    except Exception:
        return 0.0


def grasp_isotropy(model: mujoco.MjModel, data: mujoco.MjData,
                   object_body: str) -> float:
    """
    Direction isotropy of contact normals around object centroid.
    Fast proxy for force-closure quality; no convex hull needed.
    Returns value in [0, 1]; higher = more isotropic (better grasp).
    """
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if obj_id < 0 or data.ncon == 0:
        return 0.0

    obj_com = data.xipos[obj_id].copy()
    dirs = []
    for i in range(data.ncon):
        c = data.contact[i]
        if model.geom_bodyid[c.geom1] != obj_id and model.geom_bodyid[c.geom2] != obj_id:
            continue
        v = c.pos - obj_com
        n = np.linalg.norm(v)
        if n > 1e-6:
            dirs.append(v / n)

    if len(dirs) < 2:
        return 0.0

    D = np.array(dirs)
    sv = np.linalg.svd(D, compute_uv=False)
    return float(sv.min() / (sv.max() + 1e-12))


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData,
                    object_body: str) -> dict:
    """
    Returns a structured dict with all grasp quality measures.
    Suitable for logging to metrics_report.json.
    """
    eps = epsilon_quality(model, data, object_body)
    iso = grasp_isotropy(model, data, object_body)

    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    n_contacts = sum(
        1 for i in range(data.ncon)
        if model.geom_bodyid[data.contact[i].geom1] == obj_id
        or model.geom_bodyid[data.contact[i].geom2] == obj_id
    ) if obj_id >= 0 else 0

    total_force = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if model.geom_bodyid[c.geom1] == obj_id or model.geom_bodyid[c.geom2] == obj_id:
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f)
            total_force += float(np.linalg.norm(f[:3]))

    return {
        "n_contacts":       n_contacts,
        "total_force_N":    round(total_force, 4),
        "epsilon_quality":  round(eps, 6),
        "grasp_isotropy":   round(iso, 6),
        "force_closure":    eps > 0.0,
    }
