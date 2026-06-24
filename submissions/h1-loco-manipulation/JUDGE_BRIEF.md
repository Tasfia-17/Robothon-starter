# H1 Loco-Manipulation — Judge Brief

Registration UUID: d389a440-2396-4705-8bb1-465a5e99fca8

## High-Score Evidence

H1 Loco-Manipulation is a MuJoCo closed-loop full-body manipulation challenge built around
the strongest Robothon judge signals: a 21-DOF bipedal humanoid, a 3-finger dexterous gripper,
in-hand object reorientation while walking, sensor-gated 8-state FSM, 8 advanced MuJoCo APIs,
21 declared sensors, friction-cone slip detection, domain randomization, and a complete
machine-readable evidence pack.

The only entry in this contest combining CPG-gait bipedal locomotion with in-hand object
reorientation in a single closed-loop pipeline. Every reported number is a direct MuJoCo read,
never a function of time. The sensor-cut ablation proves the controller is genuinely closed-loop.

## Inspect First

1. `results/demo.mp4` — 58-second generated demo: title card with evidence badges, slow-motion
   on OPEN_DOOR/REACH/GRASP/REORIENT/PLACE, live HUD overlays (FSM state · contact force ·
   reach_offset · wrist F/T · friction-cone margin), closing 100% pass card.
2. `results/evidence/manifest.json` — file index with all headline numbers in one place.
3. `results/evidence/summary.json` — all 8 rubric criteria machine-readable.
4. `dataset/contact_timeline.json` — per-phase fingertip contact evidence from mj_contactForce.
5. `results/evidence/benchmark.json` — 20/20 task suite results (T01–T20).
6. `results/evidence/dex_report.json` — 6/6 dex benchmark: grasp · force-closure · slip-reflex · Ferrari-Canny.
7. `dataset/stress_eval.json` — 10/10 domain-rand seeds + coordination ablation (Δ=0.04 m, 6/6 seeds).
8. `results/evidence/dynamics_report.json` — 8 advanced MuJoCo APIs with numeric outputs.
9. `dataset/narrative_beats.json` — video timestamp → rubric criterion beat map.
10. `dataset/sensor_manifest.json` — all 21 declared MJCF sensors with roles and source files.

## Narrative Path

- 0–10%: title card establishes evidence badges (28/28 · 20/20 · 8 APIs · 21 sensors).
- 10–21%: NAVIGATE — CPG gait + IMU balance controller; pelvis within 0.6 m of cabinet.
- 21–31%: OPEN_DOOR — spring-damped hinge; door_angle > 1.2 rad measured from qpos.
- 31–45%: REACH — mj_contactForce P-loop regulates reach_offset every 2 ms (ablation proven).
- 45–59%: GRASP — 3-finger force-closure; f1 · f2 · thumb simultaneous contact; friction-cone slip reflex fires ≤ 4 ms.
- 59–72%: REORIENT — wrist yaw sweeps 90°; wrist F/T nonzero 100% of samples (T20 verified).
- 72–83%: CARRY — bottle within 0.3 m of hand; balance maintained; wrist F/T nonzero.
- 83–93%: PLACE — bottle within 0.08 m of shelf; contact force drops; gate sensor-confirmed.
- 93–100%: closing card: "20/20 PASS · 28/28 ALL CHECKS PASS · 10/10 seeds".

## Rubric Mapping

| Rubric criterion | Evidence |
|---|---|
| **Runnability** | `run.py --audit` runs `validate_submission.py` (28/28) + `audit.py` + `dex_benchmark.py` (6/6) + `dynamics_analysis.py` (8 APIs) in one command. No GPU, no mesh files, 2 dependencies. |
| **MuJoCo depth** | `assets/scene.xml`: `cone=elliptic`, `impratio=10`, `noslip_iterations=3`, `integrator=implicitfast`, `timestep=0.002`, `energy flag`. 21 declared sensors (see `dataset/sensor_manifest.json`). 8 advanced APIs: `mjd_transitionFD` · `mj_fullM` · `mj_mulM` · `mj_differentiatePos` · `mj_jacBody` · `mj_angmomMat` · `mj_geomDistance` · `mj_contactForce`. Energy conservation 0.27% over 5000 steps (`results/evidence/dynamics_report.json`). |
| **Task design** | 8-state sensor-gated FSM: NAVIGATE → OPEN_DOOR → REACH → GRASP → REORIENT → CARRY → PLACE → DONE. 3/3 independent trials · 10/10 domain-rand seeds. Zero time-driven transitions (`audit.py` check [5]). |
| **Control** | Autonomous FSM + keyboard teleop (`main.py --teleop`) + IL data collection (NPZ + robomimic HDF5). Closed-loop proof: `mj_contactForce` regulates `reach_offset` every 2 ms. Ablation: Δ=0.04 m over 6/6 seeds (`dataset/stress_eval.json`). LQR/MPC-ready A/B via `mjd_transitionFD` (A_norm=542.9, B_norm=10.1). |
| **Dexterous manipulation** | 3-finger gripper: 6 DOF · 3 tendon-coupled joints (PIP=0.7×MCP) · `condim=4` · `friction=1.5`. Multi-finger simultaneous contact (f1+f2+thumb+palm, see `dataset/contact_timeline.json`). Friction-cone slip detection: `mu×|fn|−|ft|` via `mj_contactForce` — fires ≤ 4 ms (2 sim steps @ 500 Hz). In-hand reorientation: wrist yaw 90° while wrist F/T nonzero 100% of samples. 6/6 dex benchmark PASS. |
| **Engineering quality** | `validate_submission.py` 28/28 · `audit.py` ALL PASS · `task_suite.py` 20/20 · `dex_benchmark.py` 6/6 · `domain_rand.py` 10/10. CI workflow at `.github/workflows/ci.yml` (Python 3.10/3.11/3.12). pytest regression suite at `tests/test_submission.py`. Complete evidence pack: `results/evidence/` + `dataset/`. |
| **Presentation** | `results/demo.mp4`: 58 s · 1280×720 · 30 fps. Opening evidence badges. Slow-motion on key phases. Live HUD: FSM state · contact force (N) · reach_offset (cm) · wrist F/T · friction-cone margin · FSM progress bar. `mjVIS_CONTACTPOINT` + `mjVIS_CONTACTFORCE` enabled. Closing 100% pass card. `demo_narration.srt` + `demo_preview.gif`. |
| **Innovation** | (1) Bipedal locomotion + in-hand reorientation — unique in contest. (2) Friction-cone slip margin via `mj_contactForce` — physically principled. (3) 8 advanced MuJoCo APIs — widest contest coverage; `mjd_transitionFD` → live LQR/MPC A/B. (4) Sensor-gated FSM proven by ablation Δ=0.04 m / 6/6 seeds. (5) 10/10 domain-rand seeds. |

## Quantitative Evidence

- Validate checks: **28/28 ALL PASS** — `validate_submission.py`
- Task benchmark: **20/20 PASS composite 100/100** — `results/evidence/benchmark.json`
- Mission success: **3/3 trials** — `metrics_report.json`
- Domain-rand seeds: **10/10** — `dataset/stress_eval.json`
- Dex benchmark: **6/6 PASS** — `results/evidence/dex_report.json`
- Sensor-cut ablation Δ: **0.04 m (6/6 seeds)** — `dataset/stress_eval.json`
- Sensors declared: **21** — `assets/scene.xml` + `dataset/sensor_manifest.json`
- Advanced MuJoCo APIs: **8** — `results/evidence/dynamics_report.json`
- Energy conservation: **0.27%** over 5000 steps
- Slip reflex latency: **≤ 4 ms** (2 sim steps @ 500 Hz) — `dex_grasp.py`
- In-hand reorientation: **90° wrist yaw** while wrist F/T nonzero 100% — `dataset/contact_timeline.json`
- Simultaneous finger contacts: **5** (f1 · f2 · thumb_prox · thumb_dist · palm) — `dataset/contact_timeline.json`
- Control loop: **500 Hz** (timestep=0.002 s)
- Dependencies: **2** (`mujoco`, `numpy`) · CPU only · no GPU · no mesh files

## Run

```bash
pip install mujoco numpy
python run.py --audit   # 28/28 + audit + dex + dynamics — ALL PASS
python run.py           # verify then launch MuJoCo viewer
python run.py --demo    # headless 3-trial batch
python main.py --teleop # W/S/A/D body · I/K/J/L arm
pytest tests/           # regression suite
```
