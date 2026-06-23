#!/usr/bin/env python3
"""
collect_demos.py — Imitation-learning data collection for H1 Loco-Manipulation.

Runs N autonomous episodes and saves each as a structured demonstration:
  - observations: 9 sensor readings + pelvis pose + arm joint angles (per timestep)
  - actions: ctrl array (21-DOF joint targets, per timestep)
  - events: timestep of each FSM state transition
  - outcome: door_opened, grasp_success, place_success, fall_count

Output: demos/demo_{i:03d}.npz  + demos/manifest.json

Usage:
    python collect_demos.py --n 10            # collect 10 demos
    python collect_demos.py --n 1 --render    # single demo with viewer
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import mujoco

from task_env import TaskEnv
from loco_control import ACTUATOR

SCENE  = Path(__file__).parent / "assets/scene.xml"
OUTDIR = Path(__file__).parent / "demos"


def _obs(model, data, env) -> np.ndarray:
    """
    51-dim observation vector:
      [0:3]   pelvis position (world)
      [3:7]   pelvis quaternion
      [7:10]  pelvis euler (roll/pitch/yaw)
      [10:13] right hand position (world)
      [13:16] left  hand position (world)
      [17:19] bottle position (world)
      [19:22] shelf target position (world)
      [22:25] door hinge: [qpos, qvel, 0]
      [25:28] foot force L (3-axis, from sensordata)
      [28:31] foot force R (3-axis)
      [31:34] imu_quat [w,x,y,z] first 3 (gyro-like)
      [34:37] imu_accel (3-axis)
      [37:58] arm joint angles (21 actuators)
    """
    s = data.sensordata

    def _sid(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(model.sensor_adr[i]) if i >= 0 else None

    def _s3(adr):
        return s[adr:adr+3].copy() if adr is not None else np.zeros(3)

    pelvis_pos  = data.qpos[8:11].copy()
    pelvis_quat = data.qpos[11:15].copy()

    def quat_to_euler(q):
        w,x,y,z = q
        r = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        p = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
        y_ = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        return np.array([r, p, y_])

    rhand = _s3(_sid("right_hand_pos"))
    lhand = _s3(_sid("left_hand_pos"))
    bottle = _s3(_sid("bottle_pos"))
    shelf  = _s3(_sid("shelf_pos"))

    door_adr = _sid("cabinet_door_hinge") if False else None
    # door from qpos directly
    djid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cabinet_door_hinge")
    door_q = float(data.qpos[model.jnt_qposadr[djid]])
    door_v = float(data.qvel[model.jnt_dofadr[djid]])

    lf = _s3(_sid("left_foot_force"))
    rf = _s3(_sid("right_foot_force"))
    imu_q = s[_sid("imu_quat"):_sid("imu_quat")+4].copy() if _sid("imu_quat") is not None else np.zeros(4)
    imu_a = s[_sid("imu_accel"):_sid("imu_accel")+3].copy() if _sid("imu_accel") is not None else np.zeros(3)

    # arm joint angles (from qpos, 21 actuators)
    arm_q = np.array([
        data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
        for i in range(model.nu)
    ])

    return np.concatenate([
        pelvis_pos, pelvis_quat, quat_to_euler(pelvis_quat),
        rhand, lhand, bottle, shelf,
        [door_q, door_v, 0.0],
        lf, rf, imu_q[:3], imu_a,
        arm_q,
    ])


def collect_episode(n: int, render: bool = False):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)
    env   = TaskEnv(model, data)

    obs_list  = []
    act_list  = []
    state_list = []   # FSM state index per step
    events    = {}    # state name → first step it appeared
    STATE_IDX = {s: i for i, s in enumerate(TaskEnv.STATES)}

    prev_state = env.state

    if render:
        import mujoco.viewer as mjv
        with mjv.launch_passive(model, data) as viewer:
            step = 0
            while not env.done and step < 100_000:
                ctrl = env.step(model.opt.timestep)
                env.apply_grasp_kinematics()
                obs_list.append(_obs(model, data, env))
                act_list.append(ctrl.copy())
                state_list.append(STATE_IDX[env.state])
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)
                if env.state != prev_state:
                    events[env.state] = step
                    prev_state = env.state
                viewer.sync()
                step += 1
    else:
        step = 0
        while not env.done and step < 100_000:
            ctrl = env.step(model.opt.timestep)
            env.apply_grasp_kinematics()
            obs_list.append(_obs(model, data, env))
            act_list.append(ctrl.copy())
            state_list.append(STATE_IDX[env.state])
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            if env.state != prev_state:
                events[env.state] = step
                prev_state = env.state
            step += 1

    s = env.summary()
    return {
        "obs":    np.array(obs_list,   dtype=np.float32),
        "acts":   np.array(act_list,   dtype=np.float32),
        "states": np.array(state_list, dtype=np.uint8),
        "events": events,
        "outcome": {
            "door_opened":   bool(s["door_opened"]),
            "grasp_success": bool(s["grasp_success"]),
            "place_success": bool(s["place_success"]),
            "fall_count":    int(s["fall_count"]),
            "steps":         step,
            "sim_time_s":    round(step * 0.002, 2),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",      type=int,  default=5,     help="Number of demos")
    ap.add_argument("--render", action="store_true",      help="Show viewer for first demo")
    args = ap.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    manifest = []
    success  = 0

    for i in range(args.n):
        t0 = time.time()
        print(f"\n[demo {i+1}/{args.n}]")
        ep = collect_episode(i, render=(args.render and i == 0))
        out = OUTDIR / f"demo_{i:03d}.npz"
        np.savez_compressed(str(out),
                            obs=ep["obs"], acts=ep["acts"], states=ep["states"])
        ok = ep["outcome"]["door_opened"] and ep["outcome"]["grasp_success"] and ep["outcome"]["place_success"]
        if ok: success += 1
        entry = {
            "file":     out.name,
            "steps":    ep["outcome"]["steps"],
            "sim_time": ep["outcome"]["sim_time_s"],
            "obs_dim":  ep["obs"].shape[1],
            "act_dim":  ep["acts"].shape[1],
            "outcome":  ep["outcome"],
            "events":   ep["events"],
            "wall_time_s": round(time.time() - t0, 2),
        }
        manifest.append(entry)
        print(f"  steps={ep['outcome']['steps']}  "
              f"door={ep['outcome']['door_opened']}  "
              f"grasp={ep['outcome']['grasp_success']}  "
              f"place={ep['outcome']['place_success']}  "
              f"ok={ok}  ({out.stat().st_size//1024} KB)")

    manifest_data = {
        "n_demos":      args.n,
        "n_success":    success,
        "success_rate": round(success / args.n, 3),
        "obs_dim":      manifest[0]["obs_dim"],
        "act_dim":      manifest[0]["act_dim"],
        "fsm_states":   TaskEnv.STATES,
        "obs_description": [
            "pelvis_pos[0:3]", "pelvis_quat[3:7]", "pelvis_euler[7:10]",
            "right_hand_pos[10:13]", "left_hand_pos[13:16]",
            "bottle_pos[16:19]", "shelf_pos[19:22]",
            "door_qpos[22]", "door_qvel[23]", "pad[24]",
            "left_foot_force[25:28]", "right_foot_force[28:31]",
            "imu_gyro[31:34]", "imu_accel[34:37]",
            "arm_joints[37:58]",
        ],
        "demos": manifest,
    }
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest_data, indent=2))

    print(f"\n✓ {args.n} demos → {OUTDIR}/")
    print(f"  success_rate={success}/{args.n}  obs_dim={manifest[0]['obs_dim']}  act_dim={manifest[0]['act_dim']}")
    print(f"  manifest → {OUTDIR}/manifest.json")


if __name__ == "__main__":
    main()
