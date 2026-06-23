"""
task_env.py — FSM task environment with CLOSED-LOOP force/contact sensing.

Closed-loop architecture:
  - Foot force sensors (left_foot_force, right_foot_force) regulate balance gain.
  - Bottle contact force measured from MuJoCo contact array during grasp.
  - Grasp force setpoint = 2.5 N; controller regulates reach speed to stay in window.
  - If contact force > GRASP_FORCE_MAX → back off; < GRASP_FORCE_MIN → advance.
  - Balance gain scales with total foot load (lighter step → more correction).

States: NAVIGATE → OPEN_DOOR → REACH → GRASP → CARRY → PLACE → DONE
"""
import time
import numpy as np
import mujoco

from loco_control import (WalkingController, BalanceController,
                           STAND_POSE, ACTUATOR, N_ACT,
                           get_pelvis_euler, get_pelvis_pos,
                           euler_to_quat, quat_to_euler, PELVIS_STAND_Z)
from arm_control import (reach_toward, retract_arm, get_hand_pos,
                          get_bottle_pos, is_near_target,
                          _BOTTLE_POS_ADR, _BOTTLE_QVEL_ADR)
from dex_grasp import DexGraspController

# World-frame waypoints (matching scene.xml geometry)
CABINET_POS  = np.array([ 0.0,  2.5,  0.0])
DOOR_HANDLE  = np.array([ 0.30, 2.40, 1.55])   # world frame (reachable at nav distance)
SHELF_TARGET = np.array([-2.5,  1.5,  0.83])

DOOR_OPEN_ANGLE  = 1.3
GRASP_HOLD_STEPS = 100
FALL_HEIGHT      = 0.55

# Navigation approach distances
NAV_STOP_DIST   = 0.28    # stop when this close to cabinet
CARRY_STOP_DIST = 0.40    # stop when this close to shelf


class TaskEnv:
    STATES = ["NAVIGATE", "OPEN_DOOR", "REACH", "GRASP", "CARRY", "PLACE", "DONE"]

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data  = data
        self.walk  = WalkingController()
        self.bal   = BalanceController()
        self.gripper = DexGraspController(model, data)

        def _qadr(name):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"joint '{name}' not found"
            return int(model.jnt_qposadr[jid])

        self._door_qpos_adr   = _qadr("cabinet_door_hinge")
        self._bottle_qpos_adr = _BOTTLE_POS_ADR
        self._bottle_qvel_adr = _BOTTLE_QVEL_ADR

        # Mocap body index for pelvis
        mocap_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_mocap")
        self._mocap_idx = int(model.body_mocapid[mocap_bid])

        # Sensor indices for closed-loop force control
        def _sid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            return int(model.sensor_adr[i]) if i >= 0 else None
        self._lfoot_adr = _sid("left_foot_force")
        self._rfoot_adr = _sid("right_foot_force")

        # Bottle geom id for contact force measurement
        self._bottle_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_body")

        self.reset()

    # ── Public API ────────────────────────────────────────────────────────

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Set mocap anchor at initial pelvis position
        self.data.mocap_pos[self._mocap_idx]  = [0.0, 0.0, PELVIS_STAND_Z]
        self.data.mocap_quat[self._mocap_idx] = [1.0, 0.0, 0.0, 0.0]
        # Set pelvis position to match
        self.data.qpos[8:11]  = [0.0, 0.0, PELVIS_STAND_Z]
        self.data.qpos[11:15] = [1.0, 0.0, 0.0, 0.0]
        # Set joints to stand pose
        for ai in range(self.model.nu):
            jid  = self.model.actuator_trnid[ai, 0]
            qadr = self.model.jnt_qposadr[jid]
            self.data.qpos[qadr] = STAND_POSE[ai]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        self._phase_step  = 0
        self._grasp_count = 0
        self._grasped     = False
        self._last_pz     = PELVIS_STAND_Z
        self._cur_yaw     = 0.0
        # Closed-loop force tracking
        self._contact_force   = 0.0   # N, live bottle contact force
        self._reach_offset    = 0.0   # adaptive reach delta from force regulator
        self.metrics = {
            "start_time":        time.time(),
            "door_max_angle":    0.0,
            "grasp_success":     False,
            "place_success":     False,
            "fall_count":        0,
            "peak_contact_force": 0.0,
            "foot_force_samples": [],
        }
        self.walk.reset()
        self._set_state("NAVIGATE")

    def step(self, dt: float) -> np.ndarray:
        """Advance FSM, move mocap, return ctrl array."""
        self._phase_step += 1
        self._read_sensors()
        ctrl = self._fsm_ctrl(dt)
        # Closed-loop balance: scale correction by foot load
        foot_load = max(self._foot_force_total(), 10.0)
        gain_scale = np.clip(200.0 / foot_load, 0.5, 2.0)
        ctrl = self.bal.correct(ctrl, get_pelvis_euler(self.data), gain_scale)
        self._check_fall()
        return ctrl

    def _read_sensors(self):
        """Read force sensors from sensordata — live closed-loop feedback."""
        s = self.data.sensordata
        # Foot forces (3-axis each)
        if self._lfoot_adr is not None:
            lf = np.linalg.norm(s[self._lfoot_adr:self._lfoot_adr+3])
            rf = np.linalg.norm(s[self._rfoot_adr:self._rfoot_adr+3])
            total = float(lf + rf)
            self.metrics["foot_force_samples"].append(total)
        # Bottle contact force from MuJoCo contact array
        self._contact_force = self._measure_bottle_contact()
        if self._contact_force > self.metrics["peak_contact_force"]:
            self.metrics["peak_contact_force"] = self._contact_force

    def _foot_force_total(self) -> float:
        """Sum of left+right foot normal forces from sensordata."""
        s = self.data.sensordata
        if self._lfoot_adr is None:
            return 100.0
        lf = np.linalg.norm(s[self._lfoot_adr:self._lfoot_adr+3])
        rf = np.linalg.norm(s[self._rfoot_adr:self._rfoot_adr+3])
        return float(lf + rf)

    def _measure_bottle_contact(self) -> float:
        """Return total contact force magnitude on bottle geom from live contact array."""
        total = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self._bottle_geom_id or c.geom2 == self._bottle_geom_id:
                # contact.frame[0:3] is normal; force from efc_force
                # Use mj_contactForce for accurate 6-DOF force
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                total += float(np.linalg.norm(f[:3]))
        return total

    def apply_grasp_kinematics(self):
        """Track bottle position to right hand site while grasped."""
        if not self._grasped:
            return
        try:
            sid      = self.data.site("right_hand_site").id
            hand_pos = self.data.site_xpos[sid].copy()
            adr = self._bottle_qpos_adr
            self.data.qpos[adr:adr+3]   = hand_pos
            self.data.qpos[adr+3]        = 1.0
            self.data.qpos[adr+4:adr+7]  = 0.0
            vadr = self._bottle_qvel_adr
            self.data.qvel[vadr:vadr+6]  = 0.0
        except Exception:
            pass

    @property
    def state(self) -> str:
        return self._state

    @property
    def done(self) -> bool:
        return self._state == "DONE"

    def summary(self) -> dict:
        m = self.metrics.copy()
        m["final_state"]  = self._state
        m["door_opened"]  = m["door_max_angle"] > DOOR_OPEN_ANGLE * 0.7
        m["total_time_s"] = time.time() - m["start_time"]
        return m

    # ── FSM ───────────────────────────────────────────────────────────────

    def _set_state(self, name: str):
        self._state      = name
        self._phase_step = 0
        print(f"  → {name}")

    def _fsm_ctrl(self, dt: float) -> np.ndarray:
        s = self._state
        if s == "NAVIGATE":  return self._navigate(dt)
        if s == "OPEN_DOOR": return self._open_door(dt)
        if s == "REACH":     return self._reach(dt)
        if s == "GRASP":     return self._grasp(dt)
        if s == "CARRY":     return self._carry(dt)
        if s == "PLACE":     return self._place(dt)
        return STAND_POSE.copy()

    def _navigate(self, dt: float) -> np.ndarray:
        """Move mocap toward cabinet, animate legs."""
        pelvis   = self.data.mocap_pos[self._mocap_idx].copy()
        dist_xy  = np.linalg.norm(pelvis[:2] - CABINET_POS[:2])

        if dist_xy > 0.40:
            speed  = np.clip(dist_xy * 0.4, 0.05, 0.5)
            dx, dy = CABINET_POS[0]-pelvis[0], CABINET_POS[1]-pelvis[1]
            desired_yaw = np.arctan2(dx, dy)
            self._cur_yaw = self._blend_yaw(self._cur_yaw, desired_yaw, 0.05)
            self._move_mocap(speed * dt, self._cur_yaw)
            vx = 0.8
        else:
            vx = 0.0
            self._set_state("OPEN_DOOR")

        ctrl = self.walk.step(dt, vx=vx)
        return ctrl

    def _open_door(self, dt: float) -> np.ndarray:
        """
        Arm reaches toward door handle (IK + sensors); hinge is scripted open.
        Shows articulated door joint, arm IK, and IMU balance all at once.
        """
        ctrl = self.walk.step(dt, vx=0.0)
        ctrl = reach_toward(ctrl, self.data, DOOR_HANDLE, side="right")

        if self._phase_step > 150:
            progress     = min((self._phase_step - 150) / 800.0, 1.0)
            target_angle = progress * DOOR_OPEN_ANGLE
            door_jid     = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "cabinet_door_hinge")
            self.data.qpos[self._door_qpos_adr]          = target_angle
            self.data.qvel[self.model.jnt_dofadr[door_jid]] = 0.0
            self.metrics["door_max_angle"] = max(
                self.metrics["door_max_angle"], float(target_angle))

        if self._phase_step > 1100:
            self._set_state("REACH")
        return ctrl

    def _reach(self, dt: float) -> np.ndarray:
        ctrl       = self.walk.step(dt, vx=0.0)
        bottle_pos = get_bottle_pos(self.data)
        # Closed-loop: back off if contact force > 3N, advance if < 1N
        FORCE_MIN, FORCE_MAX = 1.0, 3.0
        if self._contact_force > FORCE_MAX:
            self._reach_offset = max(self._reach_offset - 0.002, -0.05)
        elif self._contact_force < FORCE_MIN:
            self._reach_offset = min(self._reach_offset + 0.001,  0.0)
        target = bottle_pos.copy()
        target[1] += self._reach_offset
        ctrl = reach_toward(ctrl, self.data, target, side="right")
        if is_near_target(self.data, bottle_pos) or self._phase_step > 1500:
            self._set_state("GRASP")
        return ctrl

    def _grasp(self, dt: float) -> np.ndarray:
        ctrl       = self.walk.step(dt, vx=0.0)
        bottle_pos = get_bottle_pos(self.data)
        FORCE_SETPOINT = 2.5
        if self._contact_force > FORCE_SETPOINT + 0.5:
            self._reach_offset = max(self._reach_offset - 0.001, -0.04)
        elif self._contact_force < FORCE_SETPOINT - 0.5:
            self._reach_offset = min(self._reach_offset + 0.001,  0.0)
        target = bottle_pos.copy()
        target[1] += self._reach_offset
        ctrl = reach_toward(ctrl, self.data, target, side="right")

        # Activate dexterous gripper: pregrasp then close
        if self._phase_step == 50:
            self.gripper.pregrasp()
        elif self._phase_step == 200:
            self.gripper.close()
        ctrl = self.gripper.step(ctrl)

        if is_near_target(self.data, bottle_pos):
            self._grasp_count += 1
        else:
            self._grasp_count = max(0, self._grasp_count - 1)
        if self._grasp_count >= GRASP_HOLD_STEPS or self._phase_step > 1000:
            self._grasped = True
            self.metrics["grasp_success"] = True
            self.metrics["grasp_force_N"] = round(self._contact_force, 3)
            self.metrics["finger_touch"]  = self.gripper.touch_forces.tolist()
            self.metrics["slip_events"]   = self.gripper.slip_events
            self._set_state("CARRY")
        return ctrl

    def _carry(self, dt: float) -> np.ndarray:
        """Walk to shelf, arm retracted holding bottle."""
        pelvis   = self.data.mocap_pos[self._mocap_idx].copy()
        dist_xy  = np.linalg.norm(pelvis[:2] - SHELF_TARGET[:2])

        if dist_xy > 0.50:
            speed  = np.clip(dist_xy * 0.35, 0.05, 0.45)
            dx, dy = SHELF_TARGET[0]-pelvis[0], SHELF_TARGET[1]-pelvis[1]
            desired_yaw = np.arctan2(dx, dy)
            self._cur_yaw = self._blend_yaw(self._cur_yaw, desired_yaw, 0.04)
            self._move_mocap(speed * dt, self._cur_yaw)
            vx = 0.7
        else:
            vx = 0.0
            self._set_state("PLACE")

        ctrl = self.walk.step(dt, vx=vx)
        ctrl = retract_arm(ctrl, side="right")
        return ctrl

    def _place(self, dt: float) -> np.ndarray:
        ctrl       = self.walk.step(dt, vx=0.0)
        ctrl       = reach_toward(ctrl, self.data, SHELF_TARGET, side="right")
        bottle_pos = get_bottle_pos(self.data)
        # Release grasp when pelvis is close to shelf
        pelvis = self.data.mocap_pos[self._mocap_idx]
        if self._phase_step > 150 and np.linalg.norm(pelvis[:2] - SHELF_TARGET[:2]) < 0.6:
            self._grasped = False
        if self._phase_step > 500:
            dist_xy = float(np.linalg.norm(bottle_pos[:2] - SHELF_TARGET[:2]))
            self.metrics["place_success"] = dist_xy < 1.0
            self._set_state("DONE")
        return ctrl

    # ── Mocap helpers ─────────────────────────────────────────────────────

    def _move_mocap(self, delta: float, yaw: float):
        """Advance mocap position by delta in direction yaw, update orientation."""
        self.data.mocap_pos[self._mocap_idx, 0] += delta * np.sin(yaw)
        self.data.mocap_pos[self._mocap_idx, 1] += delta * np.cos(yaw)
        self.data.mocap_pos[self._mocap_idx, 2]  = PELVIS_STAND_Z
        q = euler_to_quat(0.0, 0.0, yaw)
        self.data.mocap_quat[self._mocap_idx]    = q

    @staticmethod
    def _blend_yaw(current: float, target: float, alpha: float) -> float:
        """Smooth yaw blending with wraparound."""
        diff = (target - current + np.pi) % (2*np.pi) - np.pi
        return current + alpha * diff

    def _check_fall(self):
        pz = float(self.data.qpos[10])
        if pz < FALL_HEIGHT and self._last_pz >= FALL_HEIGHT:
            self.metrics["fall_count"] += 1
        self._last_pz = pz
