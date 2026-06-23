"""
Locomotion + whole-body controller for H1 humanoid.

Architecture:
- Pelvis is welded to a mocap body (pelvis_mocap) for stability.
- mocap_pos/quat is driven by the task FSM to navigate to waypoints.
- CPG walking pattern plays on legs/arms for visual realism.
- IMU + balance controller corrects for any residual tilt.

qpos layout (from scene.xml joint order):
  [0]       cabinet_door_hinge
  [1..7]    bottle_free   (pos[1:4], quat[4:8])
  [8..14]   root/pelvis   (pos[8:11], quat[11:15])
  [15..]    hinge joints
"""
import numpy as np

ACTUATOR = {
    "torso": 0,
    "left_hip_yaw": 1,  "left_hip_roll": 2,  "left_hip_pitch": 3,
    "left_knee": 4,     "left_ankle_pitch": 5, "left_ankle_roll": 6,
    "right_hip_yaw": 7, "right_hip_roll": 8, "right_hip_pitch": 9,
    "right_knee": 10,   "right_ankle_pitch": 11, "right_ankle_roll": 12,
    "left_shoulder_pitch": 13, "left_shoulder_roll": 14,
    "left_shoulder_yaw": 15,   "left_elbow": 16,
    "right_shoulder_pitch": 17, "right_shoulder_roll": 18,
    "right_shoulder_yaw": 19,   "right_elbow": 20,
}
N_ACT = 21

# Standing joint targets (mild knee bend for natural look)
STAND_POSE = np.zeros(N_ACT)
STAND_POSE[ACTUATOR["left_hip_pitch"]]    = -0.15
STAND_POSE[ACTUATOR["left_knee"]]         =  0.30
STAND_POSE[ACTUATOR["left_ankle_pitch"]]  = -0.15
STAND_POSE[ACTUATOR["right_hip_pitch"]]   = -0.15
STAND_POSE[ACTUATOR["right_knee"]]        =  0.30
STAND_POSE[ACTUATOR["right_ankle_pitch"]] = -0.15
STAND_POSE[ACTUATOR["left_shoulder_pitch"]]  = 0.3
STAND_POSE[ACTUATOR["right_shoulder_pitch"]] = 0.3

# Pelvis height (z) for standing
PELVIS_STAND_Z = 0.93


class WalkingController:
    """
    CPG gait for visual leg animation.
    Does NOT drive locomotion (that's handled by mocap).
    vx command scales leg swing amplitude for visual realism.
    """
    def __init__(self, step_freq=1.4):
        self.step_freq = step_freq
        self.phase = 0.0

    def reset(self):
        self.phase = 0.0

    def step(self, dt: float, vx: float = 0.0, vy: float = 0.0,
             turn: float = 0.0) -> np.ndarray:
        self.phase += 2.0 * np.pi * self.step_freq * dt
        phi  = self.phase
        ctrl = STAND_POSE.copy()

        amp  = np.clip(abs(vx), 0.0, 1.0) * 0.25
        spd  = np.sign(vx) if vx != 0 else 1.0
        lift = 0.10

        ctrl[ACTUATOR["left_hip_pitch"]]   = -0.15 + amp * np.sin(phi)  * spd
        ctrl[ACTUATOR["right_hip_pitch"]]  = -0.15 - amp * np.sin(phi)  * spd
        ctrl[ACTUATOR["left_knee"]]        =  0.30 + lift * max(0.0,  np.sin(phi))
        ctrl[ACTUATOR["right_knee"]]       =  0.30 + lift * max(0.0, -np.sin(phi))
        ctrl[ACTUATOR["left_ankle_pitch"]] = -0.15 - amp * 0.3 * np.sin(phi) * spd
        ctrl[ACTUATOR["right_ankle_pitch"]]= -0.15 + amp * 0.3 * np.sin(phi) * spd

        # Lateral sway
        sway = 0.05 * np.sin(phi)
        ctrl[ACTUATOR["left_hip_roll"]]  = -sway
        ctrl[ACTUATOR["right_hip_roll"]] =  sway

        # Arm swing counter-phase
        arm_sw = 0.18 * abs(vx)
        ctrl[ACTUATOR["left_shoulder_pitch"]]  = 0.3 + arm_sw * np.sin(phi)
        ctrl[ACTUATOR["right_shoulder_pitch"]] = 0.3 - arm_sw * np.sin(phi)
        return ctrl


class BalanceController:
    """Gentle ankle + hip roll balance (mostly cosmetic with weld active)."""

    def __init__(self, kp_pitch=2.0, kp_roll=2.0):
        self.kp_pitch = kp_pitch
        self.kp_roll  = kp_roll

    def correct(self, ctrl: np.ndarray, euler: np.ndarray) -> np.ndarray:
        roll, pitch, _ = euler
        ankle = -self.kp_pitch * pitch
        ctrl[ACTUATOR["left_ankle_pitch"]]  += ankle
        ctrl[ACTUATOR["right_ankle_pitch"]] += ankle
        hr = -self.kp_roll * roll
        ctrl[ACTUATOR["left_hip_roll"]]  += hr
        ctrl[ACTUATOR["right_hip_roll"]] -= hr
        return ctrl


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """[w,x,y,z] → [roll, pitch, yaw]"""
    w, x, y, z = quat
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """[roll,pitch,yaw] → [w,x,y,z]"""
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


def get_pelvis_pos(data) -> np.ndarray:
    return data.qpos[8:11].copy()


def get_pelvis_quat(data) -> np.ndarray:
    return data.qpos[11:15].copy()


def get_pelvis_euler(data) -> np.ndarray:
    return quat_to_euler(get_pelvis_quat(data))


def get_com_pos(model, data) -> np.ndarray:
    total = sum(model.body_mass)
    if total < 1e-6:
        return get_pelvis_pos(data)
    com = np.zeros(3)
    for i in range(model.nbody):
        com += model.body_mass[i] * data.xipos[i]
    return com / total
