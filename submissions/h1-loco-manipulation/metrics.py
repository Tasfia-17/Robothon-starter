#!/usr/bin/env python3
"""
metrics.py — Live sensor dashboard + batch evaluation.

Shows a live table of all 9 sensor readings while the simulation runs,
then prints a final metrics summary.

Usage:
    python metrics.py              # run 1 episode, print live sensor table
    python metrics.py --trials 5   # batch: 5 episodes → metrics_report.json
"""
import argparse, json, time
import numpy as np
import mujoco
from task_env import TaskEnv
from pathlib import Path

SCENE = Path(__file__).parent / "assets/scene.xml"

SENSORS = [
    ("imu_quat",         4, "pelvis orientation (w,x,y,z)"),
    ("imu_gyro",         3, "angular velocity (rad/s)"),
    ("imu_accel",        3, "linear acceleration (m/s²)"),
    ("left_foot_force",  3, "left foot contact force (N)"),
    ("right_foot_force", 3, "right foot contact force (N)"),
    ("right_hand_pos",   3, "right hand world position (m)"),
    ("left_hand_pos",    3, "left  hand world position (m)"),
    ("bottle_pos",       3, "bottle world position (m)"),
    ("shelf_pos",        3, "shelf target world position (m)"),
]


def run_episode(print_live: bool = False) -> dict:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)
    env   = TaskEnv(model, data)

    def sid(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(model.sensor_adr[i]) if i >= 0 else None

    adrs = {n: sid(n) for n, *_ in SENSORS}
    last_print = 0

    step = 0
    while not env.done and step < 100_000:
        data.ctrl[:] = env.step(model.opt.timestep)
        env.apply_grasp_kinematics()
        mujoco.mj_step(model, data)
        step += 1

        if print_live and (step - last_print) >= 500:
            last_print = step
            print(f"\n── step={step:6d}  state={env.state:12s} ──────────────────")
            for name, dim, desc in SENSORS:
                a = adrs[name]
                if a is None: continue
                v = data.sensordata[a:a+dim]
                vstr = " ".join(f"{x:7.3f}" for x in v)
                print(f"  {name:<22s} [{vstr}]  # {desc}")
            foot_total = (np.linalg.norm(data.sensordata[adrs["left_foot_force"]:adrs["left_foot_force"]+3])
                        + np.linalg.norm(data.sensordata[adrs["right_foot_force"]:adrs["right_foot_force"]+3]))
            print(f"  {'total_foot_force':<22s} {foot_total:7.2f} N")

    s = env.summary()
    foot = s.get("foot_force_samples", [])
    return {
        "door_opened":     bool(s["door_opened"]),
        "door_max_angle":  round(float(s["door_max_angle"]), 4),
        "grasp_success":   bool(s["grasp_success"]),
        "place_success":   bool(s["place_success"]),
        "fall_count":      int(s["fall_count"]),
        "peak_contact_N":  round(float(s.get("peak_contact_force", 0)), 2),
        "mean_foot_N":     round(float(np.mean(foot)) if foot else 0.0, 2),
        "steps":           step,
        "sim_time_s":      round(step * 0.002, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    args = ap.parse_args()

    results = []
    for i in range(args.trials):
        print(f"\n[trial {i+1}/{args.trials}]")
        r = run_episode(print_live=(args.trials == 1))
        results.append(r)
        print(f"  door={r['door_opened']}  grasp={r['grasp_success']}  "
              f"place={r['place_success']}  falls={r['fall_count']}  "
              f"sim={r['sim_time_s']:.1f}s")

    ok = [r for r in results if r["door_opened"] and r["grasp_success"] and r["place_success"]]
    report = {
        "trials":        args.trials,
        "success_count": len(ok),
        "success_rate":  round(len(ok) / args.trials, 3),
        "mean_sim_time": round(np.mean([r["sim_time_s"] for r in results]), 2),
        "sensors_used":  [n for n, *_ in SENSORS],
        "n_sensors":     len(SENSORS),
        "results":       results,
    }
    out = Path(__file__).parent / "metrics_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n✓ {args.trials} trial(s) — success={len(ok)}/{args.trials}")
    print(f"  report → {out.name}")


if __name__ == "__main__":
    main()
