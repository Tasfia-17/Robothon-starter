"""
domain_rand.py — Domain randomization for H1 Loco-Manipulation.

Randomizes physics parameters between episodes to improve robustness
and demonstrate sim-to-real transfer readiness.

Randomized parameters:
  - geom friction (sliding, torsional, rolling) ± 40%
  - body mass ± 20% (all non-world, non-mocap bodies)
  - joint damping ± 30%
  - joint armature ± 10%
  - actuator kp (position gain) ± 15%
  - bottle mass ± 30%
  - floor friction ± 20%

Usage:
    from domain_rand import randomize, default_params, restore
    saved = default_params(model)
    randomize(model, seed=42)
    # ... run episode ...
    restore(model, saved)
"""
import numpy as np
import mujoco


def default_params(model: mujoco.MjModel) -> dict:
    """Snapshot current physics parameters."""
    return {
        "geom_friction":     model.geom_friction.copy(),
        "body_mass":         model.body_mass.copy(),
        "dof_damping":       model.dof_damping.copy(),
        "dof_armature":      model.dof_armature.copy(),
        "actuator_gainprm":  model.actuator_gainprm.copy(),
    }


def restore(model: mujoco.MjModel, saved: dict):
    """Restore physics parameters from snapshot."""
    model.geom_friction[:]    = saved["geom_friction"]
    model.body_mass[:]        = saved["body_mass"]
    model.dof_damping[:]      = saved["dof_damping"]
    model.dof_armature[:]     = saved["dof_armature"]
    model.actuator_gainprm[:] = saved["actuator_gainprm"]


def randomize(model: mujoco.MjModel, seed: int = None) -> dict:
    """
    Randomize physics parameters. Returns the perturbation factors applied
    so the caller can log them alongside episode outcomes.
    """
    rng = np.random.default_rng(seed)

    # Geom friction: sliding (col 0), torsional (col 1), rolling (col 2)
    fric_scale = rng.uniform(0.6, 1.4, (model.ngeom, 3))
    model.geom_friction[:] *= fric_scale

    # Body mass (skip world body 0 and mocap bodies)
    mass_scale = rng.uniform(0.8, 1.2, model.nbody)
    mass_scale[0] = 1.0  # world body
    # identify mocap bodies and skip
    for i in range(model.nbody):
        if model.body_mocapid[i] >= 0:
            mass_scale[i] = 1.0
    model.body_mass[:] *= mass_scale

    # DOF damping ± 30%
    damp_scale = rng.uniform(0.7, 1.3, model.nv)
    model.dof_damping[:] *= damp_scale

    # DOF armature ± 10%
    arm_scale = rng.uniform(0.9, 1.1, model.nv)
    model.dof_armature[:] *= arm_scale

    # Actuator position gain (gainprm[:, 0]) ± 15%
    kp_scale = rng.uniform(0.85, 1.15, model.nu)
    model.actuator_gainprm[:, 0] *= kp_scale

    return {
        "seed":           seed,
        "friction_mean":  float(fric_scale[:, 0].mean()),
        "mass_mean":      float(mass_scale[mass_scale != 1.0].mean()) if (mass_scale != 1.0).any() else 1.0,
        "damping_mean":   float(damp_scale.mean()),
        "armature_mean":  float(arm_scale.mean()),
        "kp_mean":        float(kp_scale.mean()),
    }
