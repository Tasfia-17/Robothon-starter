# H1 Loco-Manipulation — Judge Brief

**One line:** the contest's only entry combining CPG-gait bipedal locomotion with in-hand object reorientation in a single closed-loop pipeline — a 21-DOF humanoid walks to a locked cabinet, opens it, grasps a free-body bottle with a 3-finger gripper, reorients the object 90° in-hand while walking, and places it on a shelf, all sensor-gated, all closed-loop, never time-driven.

Registration UUID: d389a440-2396-4705-8bb1-465a5e99fca8

---

## The thesis

The bottle is a free-body in `scene.xml` — it moves only under the gripper's `mj_contactForce`, never teleported or welded. Every FSM transition fires on a live sensordata read. The sensor-cut ablation proves this: blinding the contact sensor leaves `reach_offset` at 0.0 m; the closed-loop controller backs it off to **−0.04 m across 6/6 independent seeds**.

No fixed-base arm, single-hand, or quadruped entry can combine bipedal locomotion with in-hand reorientation. No other entry uses `mjd_transitionFD` to produce live LQR/MPC-ready A/B matrices. No other entry has a friction-cone slip margin (Coulomb `mu×|fn|−|ft|`) firing in 4 ms via `mj_contactForce`.

---

## Headline numbers

| Metric | Value | Source |
|---|---|---|
| Validate checks | **28 / 28 ALL PASS** | `validate_submission.py` |
| Task benchmark | **20 / 20 PASS** | `results/evidence/benchmark.json` |
| Mission success | **3 / 3 trials** | `metrics_report.json` |
| Domain-rand seeds | **10 / 10** | `dataset/stress_eval.json` |
| Dex benchmark | **6 / 6 PASS** | `dex_report.json` |
| Sensor-cut ablation Δ | **0.04 m (6/6 seeds)** | `ablation.json` |
| Advanced MuJoCo APIs | **8** | `dynamics_report.json` |
| Sensors declared | **21** | `assets/scene.xml` |
| Slip reflex | **≤ 4 ms** (2 steps @ 500 Hz) | `dex_grasp.py` |
| In-hand reorientation | **90° wrist yaw**, F/T nonzero 100% | `dataset/contact_timeline.json` |
| Energy conservation | **0.27%** over 5000 steps | `dynamics_report.json` |
| Sensor cross-validation | bottle_pos agrees to < 1 mm; touch_palm agrees with mj_contactForce | `audit.py` check [6] |

---

## Reproduce

```bash
pip install mujoco numpy
python run.py --audit   # 28/28 ALL CHECKS PASS
python run.py --demo    # 3-trial batch
python main.py --teleop # keyboard teleoperation
pytest tests/           # regression suite
```

---

## Cooperation / closed-loop verification (re-run: `python audit.py`)

- **sensor_liveness** — PASS. sensordata changes between steps; ctrl ≠ sensordata
- **closed_loop_force_regulation** — PASS. reach_offset moves from zero under contact force
- **sensor_cut_ablation** — PASS. closed-loop −0.04 m vs open-loop 0.0 m, Δ=0.04 m (6/6 seeds)
- **all_fsm_states_reachable** — PASS. all 8 states visited (NAVIGATE→OPEN_DOOR→REACH→GRASP→REORIENT→CARRY→PLACE→DONE)
- **no_time_driven_outputs** — PASS. zero wall-clock time calls in any control path
- **sensor_cross_validation** — PASS. bottle_pos sensor agrees with body xpos; touch_palm agrees with mj_contactForce
- **dex_gripper_sensor_dependent** — PASS. grip_cmd ≥ 1.0 rad in HOLDING state

---

## Rubric mapping

- **Runnability:** `run.py --audit` is one command. No GPU, no mesh files, 2 dependencies. `validate_submission.py` 28/28.
- **MuJoCo depth:** `cone=elliptic`, `impratio=10`, `noslip_iterations=3`, `integrator=implicitfast`, `energy flag`. 21 declared sensors. 8 advanced APIs: `mjd_transitionFD` · `mj_fullM` · `mj_mulM` · `mj_differentiatePos` · `mj_jacBody` · `mj_angmomMat` · `mj_geomDistance` · `mj_contactForce`.
- **Task design:** 8-state sensor-gated FSM; 3/3 trials; 10/10 domain-rand seeds; zero time-driven transitions.
- **Control:** Autonomous FSM + teleop + IL data collection. Closed-loop proven by ablation. LQR/MPC-ready A/B from `mjd_transitionFD`.
- **Dexterous manipulation:** 3-finger gripper, condim=4, tendon-coupled. Friction-cone slip detection ≤ 4 ms. In-hand reorientation 90° with wrist F/T verification. 6/6 dex benchmark.
- **Engineering quality:** 28/28 validate · 20/20 task suite · 6/6 dex · 10/10 domain-rand · CI workflow · pytest suite · complete evidence + dataset pack.
- **Presentation:** 58 s · 1280×720 · live HUD · slow-motion on key phases · narration SRT · GIF preview.
- **Innovation:** Only bipedal locomotion + in-hand reorientation entry. Friction-cone slip margin. 8 advanced APIs. Ablation-proven sensor-gated FSM. 10-seed domain randomization.
