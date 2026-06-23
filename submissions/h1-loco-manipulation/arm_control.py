"""
Arm manipulation controller: analytical IK for H1's 4-DOF arm.

Geometry from h1_model.xml:
- Right arm shoulder at pos="0 -0.2 0.37" in torso_link frame.
- torso_link at pos="0 0 0.1" in pelvis frame.
- Arm extends in LOCAL -Y direction from shoulder (robot's right side).
- L_UPPER = 0.28 m (shoulder to elbow), L_LOWER = 0.26 m (elbow to hand).

IK is solved in ROBOT LOCAL FRAME then mapped to joint angles.
This ensures correctness regardless of pelvis orientation.
"""
import numpy as np
from loco_control import ACTUATOR, get_pelvis_pos, get_pelvis_quat

L_UPPER    = 0.28
L_LOWER    = 0.26
GRASP_DIST = 0.10   # metres — generous proximity threshold

# bottle freejoint: qposadr=1 (pos[1:4], quat[4:8]), dofadr=0
_BOTTLE_POS_ADR  = 1
_BOTTLE_QVEL_ADR = 0


def _rot_from_quat(q: np.ndarray) -> np.ndarray:
    """[w,x,y,z] → 3×3 rotation matrix (world ← local)."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def get_pelvis_R(data) -> np.ndarray:
    """3×3 rotation: world ← pelvis."""
    return _rot_from_quat(get_pelvis_quat(data))


def get_shoulder_world(data, side: str) -> np.ndarray:
    """
    World position of shoulder joint.
    pelvis_pos + R @ (torso_local + shoulder_local)
    torso: (0,0,0.1) in pelvis frame
    right shoulder: (0,-0.2,0.37) in torso frame → total (0,-0.2,0.47) in pelvis
    left shoulder:  (0,+0.2,0.37) in torso frame → total (0,+0.2,0.47) in pelvis
    """
    pos = get_pelvis_pos(data)
    R   = get_pelvis_R(data)
    if side == "right":
        offset_local = np.array([0.0, -0.2, 0.47])
    else:
        offset_local = np.array([0.0,  0.2, 0.47])
    return pos + R @ offset_local


def _ik_in_local_frame(target_world: np.ndarray, shoulder_world: np.ndarray,
                        pelvis_R: np.ndarray, side: str) -> dict:
    """
    Solve IK in the robot's LOCAL frame.
    rel_world = target - shoulder
    rel_local = R^T @ rel_world  (transform to pelvis frame)
    In local frame:
      - Forward = +Y axis
      - Right   = -Y axis (right arm extends in -Y)
      - Up      = +Z axis
    """
    rel_w = target_world - shoulder_world
    rel_l = pelvis_R.T @ rel_w          # target in pelvis/robot local frame

    dist = float(np.linalg.norm(rel_l))
    dist = np.clip(dist, 0.02, L_UPPER + L_LOWER - 0.02)

    # Elevation angle (shoulder pitch): negative = arm goes up
    sp = np.arctan2(-rel_l[2], np.sqrt(rel_l[0]**2 + rel_l[1]**2))

    # Azimuth (shoulder yaw) in horizontal plane, relative to arm's rest axis
    # Right arm rest axis = -Y in local frame → forward reach = +Y → sy > 0
    # Left  arm rest axis = +Y in local frame
    if side == "right":
        # atan2(x_local, -y_local): 0 = straight ahead (-Y), positive = left turn
        sy = np.arctan2(rel_l[0], -rel_l[1])
    else:
        sy = np.arctan2(-rel_l[0], rel_l[1])

    cos_e = (dist**2 - L_UPPER**2 - L_LOWER**2) / (2 * L_UPPER * L_LOWER)
    elbow = -np.arccos(np.clip(cos_e, -1.0, 1.0))
    sr    = 0.15 if side == "right" else -0.15

    p = side + "_"
    return {
        p+"shoulder_pitch": float(np.clip(sp,    -3.14, 3.14)),
        p+"shoulder_roll":  float(np.clip(sr,    -3.14, 3.14)),
        p+"shoulder_yaw":   float(np.clip(sy,    -1.57, 1.57)),
        p+"elbow":          float(np.clip(elbow, -1.57, 0.0)),
    }


def _apply(ctrl: np.ndarray, targets: dict) -> np.ndarray:
    for name, val in targets.items():
        if name in ACTUATOR:
            ctrl[ACTUATOR[name]] = val
    return ctrl


def get_hand_pos(data, side: str = "right") -> np.ndarray:
    try:
        sid = data.site(f"{side}_hand_site").id
        return data.site_xpos[sid].copy()
    except Exception:
        return get_pelvis_pos(data)


def get_bottle_pos(data) -> np.ndarray:
    try:
        sid = data.site("bottle_center").id
        return data.site_xpos[sid].copy()
    except Exception:
        return data.qpos[_BOTTLE_POS_ADR:_BOTTLE_POS_ADR+3].copy()


def is_near_target(data, target: np.ndarray, side: str = "right",
                   threshold: float = GRASP_DIST) -> bool:
    return float(np.linalg.norm(get_hand_pos(data, side) - target)) < threshold


def reach_toward(ctrl: np.ndarray, data, target: np.ndarray,
                 side: str = "right") -> np.ndarray:
    shoulder = get_shoulder_world(data, side)
    R        = get_pelvis_R(data)
    return _apply(ctrl, _ik_in_local_frame(target, shoulder, R, side))


def retract_arm(ctrl: np.ndarray, side: str = "right") -> np.ndarray:
    s = side + "_"
    ctrl[ACTUATOR[s+"shoulder_pitch"]] =  0.3
    ctrl[ACTUATOR[s+"shoulder_roll"]]  = -0.15 if side == "right" else 0.15
    ctrl[ACTUATOR[s+"shoulder_yaw"]]   =  0.0
    ctrl[ACTUATOR[s+"elbow"]]          = -0.8
    return ctrl
