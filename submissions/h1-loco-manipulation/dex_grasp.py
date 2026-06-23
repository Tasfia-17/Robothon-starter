"""
dex_grasp.py — 3-finger dexterous gripper controller for H1.

Features:
- Force-closure grasp regulation: target 2.0 N per fingertip
- Slip reflex: grip escalates within one 2 ms tick if friction-cone margin violated
- Touch sensor feedback: all 3 fingertip touch sensors read live
- Tendon-coupled PIP/IP joints (underactuated, like real tendon-driven hands)
- Grasp quality metrics via grasp_quality.py

Actuators: f1_flex(21), f2_flex(22), th_flex(23)
Sensors:   touch_f1(0), touch_f2(1), touch_th(2)
           pos_f1_tip, pos_f2_tip, pos_th_tip
"""
import numpy as np
import mujoco

# Actuator indices (body joints 0-20, fingers 21-23)
F1 = 21
F2 = 22
TH = 23

# Grasp force targets
FORCE_TARGET   = 2.0   # N per fingertip (gentle)
FORCE_MAX      = 5.0   # N — slip reflex triggers above this
SLIP_THRESH    = 0.3   # friction-cone margin: mu*fn - |ft| < SLIP_THRESH → slip
FINGER_MU      = 1.5   # friction coefficient (matches geom friction in h1_model.xml)
CLOSE_POSE     = 1.1   # rad — fully closed MCP angle
OPEN_POSE      = 0.0   # rad — open
PREGRASP_POSE  = 0.4   # rad — pre-grasp (wide enough for bottle)


class DexGraspController:
    """
    Closed-loop 3-finger grasp controller.

    States: OPEN → PREGRASP → CLOSING → HOLDING → OPEN
    Force-closure and slip detection run every step during HOLDING.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data  = data
        self._state    = "OPEN"
        self._step     = 0
        self._grip_cmd = np.array([OPEN_POSE, OPEN_POSE, OPEN_POSE])  # f1,f2,th

        # Sensor address lookup
        def _sid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            return int(model.sensor_adr[i]) if i >= 0 else None

        self._tf1 = _sid("touch_f1")
        self._tf2 = _sid("touch_f2")
        self._tth = _sid("touch_th")

        # Per-step metrics
        self.touch_forces = np.zeros(3)   # [f1, f2, th] N
        self.slip_events  = 0
        self.grasp_active = False
        self.friction_cone_margins: list[float] = []  # mu*fn - |ft| per contact per step

    def _read_touch(self):
        """Read touch forces — uses both touch sensors AND contact array for robustness."""
        s = self.data.sensordata
        t0 = float(s[self._tf1]) if self._tf1 is not None else 0.0
        t1 = float(s[self._tf2]) if self._tf2 is not None else 0.0
        t2 = float(s[self._tth]) if self._tth is not None else 0.0

        # Also accumulate contact force on any finger geom (more reliable)
        finger_bodies = set()
        for name in ("f1_prox","f1_dist","f2_prox","f2_dist","thumb_prox","thumb_dist","gripper_base"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                finger_bodies.add(bid)

        contact_force = np.zeros(3)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            for fi, bid in enumerate(sorted(finger_bodies)):
                if b1 == bid or b2 == bid:
                    f = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, f)
                    idx = min(fi, 2)
                    contact_force[idx] += abs(f[0])

        # Use max of sensor and contact-based reading
        self.touch_forces[0] = max(t0, contact_force[0])
        self.touch_forces[1] = max(t1, contact_force[1])
        self.touch_forces[2] = max(t2, contact_force[2])

    def _slip_detected(self) -> bool:
        """
        Friction-cone slip detection using mj_contactForce.
        For each contact involving a finger geom, compute the friction-cone margin:
            margin = mu * |f_normal| - |f_tangential|
        If margin < SLIP_THRESH for any active finger → slip imminent.
        Falls back to touch-sensor check when no contacts present.
        """
        if not self.grasp_active:
            return False

        finger_bodies = set()
        for name in ("f1_prox","f1_dist","f2_prox","f2_dist","thumb_prox","thumb_dist"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                finger_bodies.add(bid)

        slip = False
        n_finger_contacts = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if b1 not in finger_bodies and b2 not in finger_bodies:
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, f)
            fn = abs(f[0])          # normal force (contact frame axis 0)
            ft = np.linalg.norm(f[1:3])  # tangential force magnitude
            margin = FINGER_MU * fn - ft
            n_finger_contacts += 1
            self.friction_cone_margins.append(round(float(margin), 4))
            if margin < SLIP_THRESH:
                slip = True

        # Fallback: if no finger contacts but grasping, treat as slip
        if n_finger_contacts == 0 and self.grasp_active:
            slip = bool(np.any(self.touch_forces < 0.05))

        return slip

    def open(self):
        self._state    = "OPEN"
        self._grip_cmd = np.array([OPEN_POSE, OPEN_POSE, OPEN_POSE])
        self.grasp_active = False
        self._step = 0

    def pregrasp(self):
        """Spread fingers to pre-grasp pose around bottle."""
        self._state    = "PREGRASP"
        self._grip_cmd = np.array([PREGRASP_POSE, PREGRASP_POSE, PREGRASP_POSE])
        self._step = 0

    def close(self):
        """Initiate closing sequence."""
        self._state = "CLOSING"
        self._step  = 0

    def step(self, ctrl: np.ndarray) -> np.ndarray:
        """
        Called every simulation timestep.
        Reads touch sensors, runs force-closure regulation, writes finger ctrl.
        Returns updated ctrl array.
        """
        self._read_touch()
        self._step += 1

        if self._state == "OPEN":
            pass  # fingers already open

        elif self._state == "PREGRASP":
            # Ramp to pregrasp pose over 200 steps
            t = min(self._step / 200.0, 1.0)
            self._grip_cmd = np.array([PREGRASP_POSE * t] * 3)

        elif self._state == "CLOSING":
            # Close until all 3 fingers sense contact
            target = min(PREGRASP_POSE + self._step * 0.003, CLOSE_POSE)
            self._grip_cmd = np.array([target, target, target])
            if np.all(self.touch_forces > 0.1) or self._step > 400:
                self._state    = "HOLDING"
                self.grasp_active = True

        elif self._state == "HOLDING":
            # Force-closure regulation: increase grip if force low, back off if high
            for i, f in enumerate(self.touch_forces):
                if f < FORCE_TARGET - 0.3:
                    self._grip_cmd[i] = min(self._grip_cmd[i] + 0.002, CLOSE_POSE)
                elif f > FORCE_TARGET + 1.0:
                    self._grip_cmd[i] = max(self._grip_cmd[i] - 0.001, PREGRASP_POSE)

            # Slip reflex: if contact lost, immediately tighten grip
            if self._slip_detected():
                self._grip_cmd += 0.05
                self._grip_cmd = np.minimum(self._grip_cmd, CLOSE_POSE)
                self.slip_events += 1

        # Write to ctrl
        ctrl[F1] = self._grip_cmd[0]
        ctrl[F2] = self._grip_cmd[1]
        ctrl[TH] = self._grip_cmd[2]
        return ctrl

    def summary(self) -> dict:
        margins = self.friction_cone_margins
        return {
            "state":                self._state,
            "touch_forces":         [round(float(f), 3) for f in self.touch_forces],
            "slip_events":          self.slip_events,
            "grasp_active":         self.grasp_active,
            "grip_cmd":             [round(float(c), 3) for c in self._grip_cmd],
            "friction_cone_margin_min":  round(float(min(margins)), 4) if margins else None,
            "friction_cone_margin_mean": round(float(np.mean(margins)), 4) if margins else None,
        }
