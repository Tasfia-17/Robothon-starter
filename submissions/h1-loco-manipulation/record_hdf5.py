"""
record_hdf5.py — robomimic-compatible HDF5 dataset recorder.

Records (obs, action, reward, done, state) per timestep.
Compatible with robomimic/LeRobot dataset schema.

Usage:
    pip install h5py
    python record_hdf5.py --n 5 --out demos/dataset.hdf5
"""
import argparse, json
from pathlib import Path
import numpy as np
import mujoco

try:
    import h5py
    _H5 = True
except ImportError:
    _H5 = False

from task_env import TaskEnv
from reward import compute as reward_compute
from collect_demos import _obs

SCENE = Path(__file__).parent / "assets/scene.xml"


def run_episode(model, data):
    env = TaskEnv(model, data)
    door_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cabinet_door_hinge")
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    prev_ctrl = np.zeros(model.nu)
    obs_l, act_l, rew_l, don_l, sta_l = [], [], [], [], []

    for _ in range(100_000):
        ctrl = env.step(model.opt.timestep)
        env.apply_grasp_kinematics()
        door_q = float(data.qpos[model.jnt_qposadr[door_jid]])
        fell = data.qpos[10] < 0.55
        r = reward_compute(model, data, prev_ctrl, env.state, door_q, fell)
        obs_l.append(_obs(model, data, env))
        act_l.append(ctrl.copy())
        rew_l.append(float(r["total"]))
        don_l.append(bool(env.done))
        st = np.empty(nstate); mujoco.mj_getState(model, data, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        sta_l.append(st)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        prev_ctrl = ctrl
        if env.done: break

    s = env.summary()
    return {
        "obs":     np.array(obs_l, dtype=np.float32),
        "actions": np.array(act_l, dtype=np.float32),
        "rewards": np.array(rew_l, dtype=np.float32),
        "dones":   np.array(don_l, dtype=np.uint8),
        "states":  np.array(sta_l, dtype=np.float32),
        "success": bool(s["door_opened"] and s["grasp_success"] and s["place_success"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",   type=int, default=5)
    ap.add_argument("--out", default="demos/dataset.hdf5")
    args = ap.parse_args()

    if not _H5:
        print("pip install h5py"); return

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)

    with h5py.File(str(out), "a") as f:
        grp = f.require_group("data")
        start = sum(1 for k in grp if k.startswith("demo_"))
        n_ok = 0
        for i in range(args.n):
            print(f"\n[demo {start+i+1}/{start+args.n}]")
            ep = run_episode(model, data)
            dg = grp.create_group(f"demo_{start+i:03d}")
            dg.attrs["num_samples"] = len(ep["obs"])
            dg.attrs["success"]     = ep["success"]
            for k in ("obs","actions","rewards","dones","states"):
                dg.create_dataset(k, data=ep[k])
            if ep["success"]: n_ok += 1
            print(f"  T={len(ep['obs'])}  success={ep['success']}")

        grp.attrs.update({
            "n_demos": start + args.n, "n_success": n_ok,
            "obs_dim": 58, "act_dim": model.nu,
            "obs_keys": json.dumps(["pelvis_pos[0:3]","pelvis_quat[3:7]",
                "right_hand_pos[10:13]","bottle_pos[16:19]","arm_joints[37:58]"]),
        })
    print(f"\n✓ {out}  ({start+args.n} demos, {n_ok} success)")


if __name__ == "__main__":
    main()
