#!/usr/bin/env python3
"""audit.py — proves closed-loop sensor integration is real. Prints ALL CHECKS PASS."""
import numpy as np, mujoco, sys
from task_env import TaskEnv

SCENE = "assets/scene.xml"
fails = []

def check(name, cond, detail=""):
    s = "  ✓" if cond else "  ✗"
    print(f"{s}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond: fails.append(name)

# 1. sensordata changes each step (not static/echoed)
print("\n[1] Sensor liveness")
m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
mujoco.mj_resetData(m,d); mujoco.mj_forward(m,d)
mujoco.mj_step(m,d); s1 = d.sensordata.copy()
mujoco.mj_step(m,d); s2 = d.sensordata.copy()
check("sensordata differs between steps", not np.allclose(s1,s2))
check("sensordata != ctrl array", not np.allclose(s1[:m.nu], d.ctrl))

# 2. closed-loop reach_offset varies with contact force
print("\n[2] Closed-loop force regulation")
m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
env = TaskEnv(m,d)
offsets = []
for _ in range(100000):
    d.ctrl[:] = env.step(m.opt.timestep)
    env.apply_grasp_kinematics(); mujoco.mj_step(m,d)
    if env.state in ("REACH","GRASP"):
        offsets.append(env._reach_offset)
    if env.state == "CARRY": break
# offset should move from 0 toward negative as force builds
moved = offsets and (min(offsets) < -1e-6 or max(offsets) > 1e-6)
check("reach_offset regulated by force (moved from zero)", moved,
      f"min={min(offsets) if offsets else 0:.5f} max={max(offsets) if offsets else 0:.5f}")

# 3. ablation — blinding force sensor changes behavior
print("\n[3] Ablation: sensor-blind vs closed-loop")
def run(blind=False):
    m2=mujoco.MjModel.from_xml_path(SCENE); d2=mujoco.MjData(m2)
    e=TaskEnv(m2,d2)
    if blind: e._measure_bottle_contact = lambda: 0.0
    reach_offsets = []
    for _ in range(100000):
        d2.ctrl[:]=e.step(m2.opt.timestep); e.apply_grasp_kinematics(); mujoco.mj_step(m2,d2)
        if e._state in ("REACH","GRASP"):
            reach_offsets.append(e._reach_offset)
        if e.done: break
    s = e.summary()
    s["reach_offset_final"] = round(reach_offsets[-1], 5) if reach_offsets else 0.0
    s["reach_offset_min"]   = round(min(reach_offsets), 5) if reach_offsets else 0.0
    return s

print("  running closed-loop..."); cl=run(blind=False)
print("  running open-loop (sensor blinded)..."); ol=run(blind=True)
check("closed-loop succeeds", cl["grasp_success"] and cl["place_success"],
      f"grasp={cl['grasp_success']} place={cl['place_success']}")
check("closed-loop reach_offset moved (force regulation active)",
      cl["reach_offset_final"] < -1e-4,
      f"offset={cl['reach_offset_final']:.4f} (negative = backed off when contact force exceeded setpoint)")
check("open-loop reach_offset stays zero (no force sensor = no regulation)",
      abs(ol["reach_offset_final"]) < 1e-4,
      f"offset={ol['reach_offset_final']:.4f}")

# 4. all 7 states visited
print("\n[4] All 7 FSM states reachable")
m=mujoco.MjModel.from_xml_path(SCENE); d=mujoco.MjData(m)
env2=TaskEnv(m,d); visited=set()
for _ in range(100000):
    d.ctrl[:]=env2.step(m.opt.timestep); env2.apply_grasp_kinematics(); mujoco.mj_step(m,d)
    visited.add(env2.state)
    if env2.done: break
for st in ["NAVIGATE","OPEN_DOOR","REACH","GRASP","CARRY","PLACE","DONE"]:
    check(f"state {st} visited", st in visited)

print("\n"+"─"*45)
if fails:
    print(f"FAILED: {fails}"); sys.exit(1)
else:
    print("ALL CHECKS PASS")
