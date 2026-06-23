#!/usr/bin/env python3
"""Headless demo recorder — outputs demo.mp4 via MuJoCo offscreen + ffmpeg pipe."""
import subprocess, pathlib, numpy as np, mujoco
from task_env import TaskEnv

SCENE  = pathlib.Path(__file__).parent / "assets/scene.xml"
OUT    = pathlib.Path(__file__).parent / "demo.mp4"
W, H, FPS = 1280, 720, 30
SPF = max(1, int(round(1.0 / (FPS * 0.002))))  # sim steps per frame

CAM = {
    "NAVIGATE":  (6.0, -18, 155, [0.0,  1.2, 0.8]),
    "OPEN_DOOR": (3.2, -14, 185, [0.0,  2.4, 1.0]),
    "REACH":     (2.8, -12, 185, [0.0,  2.4, 0.9]),
    "GRASP":     (2.5, -10, 185, [0.0,  2.4, 0.9]),
    "CARRY":     (5.5, -16, 130, [-1.2, 1.8, 0.8]),
    "PLACE":     (3.0, -12,  60, [-2.5, 1.5, 0.9]),
    "DONE":      (4.0, -18,  60, [-2.5, 1.5, 0.8]),
}

def make_cam(state):
    dist, elev, az, lookat = CAM.get(state, CAM["NAVIGATE"])
    c = mujoco.MjvCamera()
    c.type, c.distance, c.elevation, c.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, dist, elev, az
    c.lookat[:] = lookat
    return c

def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)
    env   = TaskEnv(model, data)
    renderer = mujoco.Renderer(model, height=H, width=W)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    # ffmpeg pipe — write raw RGB frames directly, no temp files
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(FPS),
         "-i", "pipe:0", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", str(OUT)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    step, last = 0, ""
    while not env.done and step < 100_000:
        data.ctrl[:] = env.step(model.opt.timestep)
        env.apply_grasp_kinematics()
        mujoco.mj_step(model, data)
        step += 1
        if env.state != last:
            last = env.state
            print(f"  [{step:6d}] → {env.state}")
        if step % SPF == 0:
            renderer.update_scene(data, camera=make_cam(env.state), scene_option=opt)
            proc.stdin.write(renderer.render().tobytes())

    # 2s hold on final frame
    renderer.update_scene(data, camera=make_cam("DONE"), scene_option=opt)
    final = renderer.render().tobytes()
    for _ in range(FPS * 2):
        proc.stdin.write(final)

    proc.stdin.close()
    proc.wait()
    renderer.close()
    print(f"\n✓ {OUT}  ({OUT.stat().st_size//1024} KB)")
    s = env.summary()
    print(f"  door={s['door_opened']} grasp={s['grasp_success']} place={s['place_success']} falls={s['fall_count']}")

if __name__ == "__main__":
    main()
