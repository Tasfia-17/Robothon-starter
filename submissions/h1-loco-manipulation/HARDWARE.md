# Sim-to-Real Mapping

H1 Loco-Manipulation is a **simulation** submission. No physical hardware was used; we report no hardware numbers we did not measure. This document maps each simulated quantity to its real-world equivalent so the control pipeline transfers without scripting.

The compiled MJCF scene is in `assets/scene.xml` and `assets/h1_model.xml`.

## Sensor → Hardware Equivalent

| Simulated sensor | MuJoCo source | Real hardware equivalent |
|---|---|---|
| `left_foot_force` / `right_foot_force` | `force` sensor on foot geom | Foot-sole force/torque sensor (e.g. ATI Mini45) |
| `right_wrist_force` / `right_wrist_torque` | `force`/`torque` on wrist site | Wrist F/T sensor (e.g. Robotiq FT 300) |
| `touch_palm`, `touch_f1`, `touch_f2`, `touch_th` | `touch` on fingertip geoms | Fingertip tactile array (e.g. BioTac, GelSight) |
| `imu_quat`, `imu_gyro`, `imu_accel` | `framequat`/`frameangvel`/`framelinvel` on pelvis | Onboard IMU (e.g. Xsens MTi) |
| `bottle_pos`, `shelf_pos` | `framepos` on object/shelf | Wrist-mounted RGB-D camera + pose estimator |
| `pos_f1_tip`, `pos_f2_tip`, `pos_th_tip` | `framepos` on fingertip bodies | Fingertip position from FK on real robot |
| `mj_contactForce` (control loop) | constraint solver read | Wrist F/T + tactile array combined |

## What transfers directly

- The 8-state FSM transitions on live sensor reads — on hardware, the same gates apply to the real sensor channels above.
- The friction-cone slip detection (`mu*fn-|ft|`) uses `mj_contactForce`; on hardware it uses the wrist F/T + tactile array.
- The IMU-based balance controller reads `imu_quat`/`imu_gyro` directly — these map 1:1 to any onboard IMU.
- The CPG gait pattern (loco_control.py) and analytical arm IK (arm_control.py) are hardware-ready code.

## What needs real calibration (not claimed here)

- Actuator latency, compliance, and backlash (we model ideal position servos).
- Contact stiffness (`solref`/`solimp`) tuned to the real gripper material and object surface.
- Camera-based object pose estimation latency (we use ground-truth `framepos` in sim).

We do not fabricate latency or noise figures. The control **structure** is hardware-ready; physical calibration is future work.
