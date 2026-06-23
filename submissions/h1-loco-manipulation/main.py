#!/usr/bin/env python3
"""
main.py — H1 Humanoid Loco-Manipulation Simulator
Robothon 2026

Usage:
    python main.py                     # scripted autonomous demo (viewer)
    python main.py --teleop            # keyboard teleoperation
    python main.py --trials 10 --no-render  # headless + metrics.json

Teleop controls:
    W/S   forward / backward
    A/D   turn left / right
    Q/E   strafe left / right
    I/K   right shoulder pitch up / down
    J/L   right shoulder yaw left / right
    R     reset episode
    ESC   quit
"""
import argparse, sys, time, json, pathlib
import numpy as np
import mujoco
import mujoco.viewer

from loco_control import STAND_POSE, N_ACT, ACTUATOR
from task_env import TaskEnv

SCENE_XML = pathlib.Path(__file__).parent / "assets" / "scene.xml"

# ── Keyboard state ─────────────────────────────────────────────────────────────
_keys: dict[int, bool] = {}

def _key_cb(key: int, action: int):
    _keys[key] = (action != 0)   # 0 = release


def _teleop_ctrl(walk_ctrl: np.ndarray,
                 arm_state: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Read _keys → velocity commands and arm deltas."""
    vx   = float(_keys.get(ord('W'), False)) - float(_keys.get(ord('S'), False))
    turn = float(_keys.get(ord('A'), False)) - float(_keys.get(ord('D'), False))
    vy   = float(_keys.get(ord('Q'), False)) - float(_keys.get(ord('E'), False))

    d_sp = (float(_keys.get(ord('I'), False)) - float(_keys.get(ord('K'), False))) * 0.03
    d_sy = (float(_keys.get(ord('J'), False)) - float(_keys.get(ord('L'), False))) * 0.03
    arm_state[ACTUATOR["right_shoulder_pitch"]] = np.clip(
        arm_state[ACTUATOR["right_shoulder_pitch"]] + d_sp, -3.14, 3.14)
    arm_state[ACTUATOR["right_shoulder_yaw"]] = np.clip(
        arm_state[ACTUATOR["right_shoulder_yaw"]] + d_sy, -1.57, 1.57)
    return arm_state, float(vx), float(vy), float(turn)


# ── Main loop ──────────────────────────────────────────────────────────────────
def run(teleop: bool = False, n_trials: int = 1, render: bool = True):
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)
    env   = TaskEnv(model, data)

    all_metrics: list[dict] = []
    arm_state = STAND_POSE.copy()   # persistent arm targets for teleop

    def run_trial(idx: int, viewer=None) -> dict:
        nonlocal arm_state
        env.reset()
        arm_state = STAND_POSE.copy()
        step = 0
        dt   = model.opt.timestep

        while not env.done and step < 60_000:
            step += 1

            # Reset key
            if _keys.get(ord('R'), False):
                env.reset()
                arm_state = STAND_POSE.copy()
                _keys[ord('R')] = False
                continue

            if teleop:
                from loco_control import WalkingController, BalanceController, get_pelvis_euler
                arm_state, vx, vy, turn = _teleop_ctrl(arm_state, arm_state)
                ctrl = env.walk.step(dt, vx=vx, vy=vy, turn=turn)
                ctrl = env.bal.correct(ctrl, get_pelvis_euler(data))
                ctrl[ACTUATOR["right_shoulder_pitch"]] = arm_state[ACTUATOR["right_shoulder_pitch"]]
                ctrl[ACTUATOR["right_shoulder_yaw"]]   = arm_state[ACTUATOR["right_shoulder_yaw"]]
            else:
                ctrl = env.step(dt)

            env.apply_grasp_kinematics()
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()

        m = env.summary()
        m["trial"] = idx
        all_metrics.append(m)
        return m

    if render:
        with mujoco.viewer.launch_passive(
            model, data,
            key_callback=_key_cb,
            show_left_ui=False,
        ) as viewer:
            viewer.cam.distance  = 5.0
            viewer.cam.elevation = -18
            viewer.cam.azimuth   = 150
            for i in range(n_trials):
                print(f"\n▶ Trial {i+1}/{n_trials}")
                m = run_trial(i + 1, viewer)
                _print_metrics(m)
                if n_trials > 1 and i < n_trials - 1:
                    time.sleep(0.5)
    else:
        for i in range(n_trials):
            m = run_trial(i + 1)
            _print_metrics(m)

    if n_trials > 1:
        _print_aggregate(all_metrics)

    return all_metrics


def _print_metrics(m: dict):
    print(
        f"  Trial {m['trial']:>2} | state={m['final_state']:<10} | "
        f"door={'✓' if m['door_opened'] else '✗'} | "
        f"grasp={'✓' if m['grasp_success'] else '✗'} | "
        f"place={'✓' if m['place_success'] else '✗'} | "
        f"time={m.get('total_time_s', 0):.1f}s | "
        f"falls={m.get('fall_count', 0)}"
    )


def _print_aggregate(metrics: list[dict]):
    n = len(metrics)
    times = [m.get("total_time_s", 0) for m in metrics]
    print(f"\n{'─'*55}")
    print(f" {n} trials — "
          f"door {sum(m['door_opened'] for m in metrics)/n*100:.0f}% | "
          f"grasp {sum(m['grasp_success'] for m in metrics)/n*100:.0f}% | "
          f"place {sum(m['place_success'] for m in metrics)/n*100:.0f}% | "
          f"avg {np.mean(times):.1f}s")
    out = pathlib.Path("metrics.json")
    out.write_text(json.dumps({
        "trials": metrics,
        "summary": {
            "n": n,
            "door_rate":  sum(m["door_opened"]   for m in metrics) / n,
            "grasp_rate": sum(m["grasp_success"]  for m in metrics) / n,
            "place_rate": sum(m["place_success"]  for m in metrics) / n,
            "mean_time_s": float(np.mean(times)),
        }
    }, indent=2))
    print(f" Metrics → {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teleop",    action="store_true")
    ap.add_argument("--trials",    type=int, default=1)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    run(teleop=args.teleop, n_trials=args.trials, render=not args.no_render)
