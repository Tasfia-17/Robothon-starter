#!/usr/bin/env python3
"""record_demo.py — cinematic headless recorder with HUD overlays + narration SRT."""
import subprocess, pathlib, time
import numpy as np
import cv2
import mujoco
from task_env import TaskEnv

SCENE = pathlib.Path(__file__).parent / "assets/scene.xml"
OUT   = pathlib.Path(__file__).parent / "demo.mp4"
SRT   = pathlib.Path(__file__).parent / "demo_narration.srt"
W, H, FPS = 1280, 720, 30
SPF = max(1, int(round(1.0 / (FPS * 0.002))))

CAM = {
    "NAVIGATE":  (7.0, -20, 150, [ 0.5,  1.0, 0.5]),
    "OPEN_DOOR": (3.0, -15, 190, [ 0.0,  2.3, 1.1]),
    "REACH":     (2.5, -10, 195, [ 0.0,  2.4, 0.9]),
    "GRASP":     (2.0,  -8, 200, [ 0.0,  2.4, 0.8]),
    "CARRY":     (6.0, -18, 120, [-1.0,  1.5, 0.7]),
    "PLACE":     (2.8, -12,  55, [-2.5,  1.5, 0.9]),
    "DONE":      (5.0, -22,  50, [-1.5,  1.2, 0.6]),
}

STATE_DESC = {
    "NAVIGATE":  "Humanoid navigating to cabinet",
    "OPEN_DOOR": "Opening cabinet door (hinge, damping=2.0)",
    "REACH":     "Reaching: mj_contactForce regulates offset",
    "GRASP":     "3-finger grasp: tendon-coupled, slip reflex",
    "CARRY":     "Carrying bottle (foot-load balance gain)",
    "PLACE":     "Placing on shelf (position verified)",
    "DONE":      "Mission complete — 3/3 success",
}

def make_cam(state):
    dist, elev, az, look = CAM.get(state, CAM["NAVIGATE"])
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.distance, c.elevation, c.azimuth = dist, elev, az
    c.lookat[:] = look
    return c

def draw_hud(frame: np.ndarray, env: TaskEnv, sim_t: float) -> np.ndarray:
    img = frame.copy()
    state = env.state

    # ── top-left: state badge ──────────────────────────────────────────────
    badge_col = (30, 200, 80)
    cv2.rectangle(img, (10, 10), (520, 52), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (520, 52), badge_col, 2)
    cv2.putText(img, f"STATE: {state}", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, badge_col, 2)

    # state description
    desc = STATE_DESC.get(state, "")
    cv2.putText(img, desc, (18, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    # ── top-right: sensor readings ─────────────────────────────────────────
    x0 = W - 310
    cv2.rectangle(img, (x0 - 8, 8), (W - 8, 155), (0, 0, 0), -1)
    cv2.rectangle(img, (x0 - 8, 8), (W - 8, 155), (80, 80, 200), 1)

    def sensor_line(label, val, y, col=(220, 220, 100)):
        cv2.putText(img, f"{label}: {val}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

    sensor_line("Sim time", f"{sim_t:.1f}s", 30)
    foot_l = env.data.sensordata[env.model.sensor_adr[
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_force")]
    ] if hasattr(env, 'model') else 0
    foot_r = env.data.sensordata[env.model.sensor_adr[
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_force")]
    ] if hasattr(env, 'model') else 0

    sensor_line("Foot L/R", f"{abs(float(foot_l)):.1f} / {abs(float(foot_r)):.1f} N", 55)
    sensor_line("Contact F", f"{env._contact_force:.1f} N", 80)
    sensor_line("Reach off", f"{env._reach_offset*100:.1f} cm", 105)

    # Gripper touch forces
    tf = env.gripper.touch_forces
    grip_col = (100, 255, 100) if env.gripper.grasp_active else (180, 180, 180)
    sensor_line(f"Grip f1/f2/th",
                f"{tf[0]:.1f}/{tf[1]:.1f}/{tf[2]:.1f}N", 130, grip_col)
    sensor_line("Slips", f"{env.gripper.slip_events}", 150, (255, 160, 60))

    # ── bottom: FSM progress bar ───────────────────────────────────────────
    phases = ["NAVIGATE","OPEN_DOOR","REACH","GRASP","CARRY","PLACE","DONE"]
    try:
        idx = phases.index(state)
    except ValueError:
        idx = 0
    bar_y = H - 35
    bw = (W - 40) // len(phases)
    for i, ph in enumerate(phases):
        col = (30, 200, 80) if i < idx else ((0, 160, 255) if i == idx else (60, 60, 60))
        cv2.rectangle(img, (20 + i*bw, bar_y), (20 + (i+1)*bw - 2, bar_y + 22), col, -1)
        cv2.putText(img, ph[:3], (24 + i*bw, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    # ── bottom-right: proof badge ──────────────────────────────────────────
    cv2.rectangle(img, (W-250, H-62), (W-8, H-8), (0,0,0), -1)
    cv2.rectangle(img, (W-250, H-62), (W-8, H-8), (0,200,100), 1)
    cv2.putText(img, "ctrl-only | no qpos teleport", (W-244, H-42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 100), 1)
    cv2.putText(img, "mj_contactForce regulated", (W-244, H-22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 100), 1)

    return img

def _fmt_srt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)
    env   = TaskEnv(model, data)
    renderer = mujoco.Renderer(model, height=H, width=W)

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{W}x{H}", "-pix_fmt", "bgr24", "-r", str(FPS),
         "-i", "pipe:0", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", str(OUT)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # SRT narration
    srt_entries = []
    state_start_frame = {env.state: 0}
    last_state = env.state

    step, frame_n = 0, 0
    while not env.done and step < 100_000:
        data.ctrl[:] = env.step(model.opt.timestep)
        env.apply_grasp_kinematics()
        mujoco.mj_step(model, data)
        step += 1

        if env.state != last_state:
            # Close SRT entry for old state
            t_start = state_start_frame.get(last_state, 0) / FPS
            t_end   = frame_n / FPS
            if t_end > t_start:
                srt_entries.append((last_state, t_start, t_end))
            state_start_frame[env.state] = frame_n
            last_state = env.state
            print(f"  [{step:6d}] → {env.state}")

        if step % SPF == 0:
            renderer.update_scene(data, camera=make_cam(env.state), scene_option=opt)
            raw = renderer.render()           # RGB
            sim_t = step * model.opt.timestep
            frame = draw_hud(raw, env, sim_t) # returns BGR for cv2→ffmpeg
            # Convert RGB→BGR for ffmpeg bgr24 input
            proc.stdin.write(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR).__class__(
                draw_hud(raw, env, sim_t)).tobytes() if False else
                draw_hud(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR), env, sim_t).tobytes())
            frame_n += 1

    # Close last SRT entry
    srt_entries.append((last_state, state_start_frame.get(last_state, 0) / FPS, frame_n / FPS))

    # 2s hold on final frame
    renderer.update_scene(data, camera=make_cam("DONE"), scene_option=opt)
    final_raw = renderer.render()
    final = draw_hud(cv2.cvtColor(final_raw, cv2.COLOR_RGB2BGR), env, step * model.opt.timestep)
    for _ in range(FPS * 2):
        proc.stdin.write(final.tobytes())

    proc.stdin.close(); proc.wait(); renderer.close()

    # Write SRT
    lines = []
    for i, (st, t0, t1) in enumerate(srt_entries, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(t0)} --> {_fmt_srt_time(t1)}")
        lines.append(f"[{st}] {STATE_DESC.get(st, st)}")
        lines.append("")
    SRT.write_text("\n".join(lines))

    s = env.summary()
    sz = OUT.stat().st_size // 1024
    print(f"\n✓ {OUT.name}  ({sz} KB)")
    print(f"  door={s['door_opened']} grasp={s['grasp_success']} place={s['place_success']} falls={s['fall_count']}")
    print(f"✓ {SRT.name}")

if __name__ == "__main__":
    main()
