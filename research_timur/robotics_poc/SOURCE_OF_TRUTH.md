# PSIVG → Robot Manipulation PoC — Source of Truth

> Goal of this document: a single, citable reference for the PoC idea so we can
> build with **as little new code as possible**. It fixes the framework, names
> the exact reused repos/methods for each pipeline step (inverse-dynamics /
> action extraction, and the simulator), maps PSIVG's existing outputs onto what
> those methods need, lists the real risks, and gives a staged plan.
>
> Scope decision (locked): **simulator = ManiSkill3**; action extraction =
> **AVDC closed-form solver** (no training); baseline generator =
> `research_timur/gen_template_video.py`. Rationale in §4–§5.

---

## 1. The framework (UniPi-style, corrected)

Planning-as-video-generation: *generate a video of the task being done, then
read the low-level actions out of it.* PSIVG is the **generator**; an
inverse-dynamics / action-extraction step turns the video into robot actions.

```
first frame + task prompt
   → VIDEO GENERATOR
        baseline:  gen_template_video.py        (CogVideoX-5B-I2V, plain noise — physics-FREE)
        ours:      full PSIVG pipeline (Steps 1–4, physics-warped noise)
   → ACTION EXTRACTION  (AVDC closed-form: flow + depth + mask + K → SE(3) subgoals)
   → position controller executes subgoals in the simulator
   → (loop per subtask — short clips only; PSIVG works best on short videos)
```

### Two corrections vs. earlier notes
1. **PSIVG *is* image+text → video.** Its front door is
   `research_timur/gen_template_video.py` = CogVideoX-5B-I2V (+ Go-with-the-Flow
   LoRA). With **plain Gaussian noise** it is the physics-free draft (= our
   baseline / UniPi-style generator); with **physics-warped noise** from Steps
   1–3 it is the physics-faithful generator. **Same backbone, only the noise
   differs** → the cleanest possible ablation.
2. **We keep ALL 4 steps (do NOT drop Step 4).** The action extractor reads RGB
   frames, so the final generated video is required. (The earlier "Steps 1–3
   only" idea was a *different*, abandoned plan that fed raw physics numbers to a
   learner.)

### Where PSIVG earns its keep
The generator's video must obey real physics, or the extracted actions are
infeasible — the **"executability gap"** (see EVA, §2.3). Physics-free models
hallucinate (object floats / teleports / wrong bounce) → wrong actions. PSIVG's
MPM-guided generation produces faithful object motion → correct actions.

### Task regime that shows the advantage
Pick a task whose **success depends on the physical outcome** and that lives in
PSIVG's sweet spot: **brief contact, then passive ballistic dynamics** (gravity,
bounce, roll) — the same regime as the tennis-ball demo. Good: *"toss/drop the
apple into the bowl"*, *"knock the cup off into the bin"*. Bad: *"drop on the
floor"* (success insensitive to physics → baseline "wins" too) and *continuously
actuated pushes* (the arm forces the object the whole time — PSIVG's MPM
simulates the **object**, not the arm; see §6 actuation gap).

---

## 2. Pipeline Step A — Inverse Dynamics / Action Extraction

Three reference points, ordered by how much we'd have to code.

### 2.1 AVDC — closed-form, NO training ⭐ (this is what we build on)
*Ko, Mao, Du, Sun, Tenenbaum — "Learning to Act from Actionless Videos through
Dense Correspondences", ICLR 2024. arXiv:2310.08576.*

Action extraction is a **closed-form geometry solve** — no inverse-dynamics
network, no action labels:

1. **GMFlow** → dense optical flow between consecutive frames.
2. Backproject the initial frame's pixels to 3D using **depth + intrinsics K**.
3. Sample ~500 points from the **object mask**; centroid = grasp/contact point.
4. Track those 3D points via flow; per step solve the rigid transform
   **Tₜ ∈ SE(3)** minimizing 2D reprojection error:
   `L = Σ_i ‖ uₜⁱ − π(K · Tₜ · xᵢ) ‖²`  (least squares).
5. Classify **grasp vs. push** by vertical-displacement magnitude; a **position
   controller** drives the arm to follow the SE(3) subgoals.

**Required inputs:** initial RGBD, K, **object mask**, **optical flow**.
**Action space:** sequence of SE(3) subgoals on the grasp point + gripper
open/close.

**Reused code:** `flow-diffusion/AVDC_experiments`
- `flowdiffusion/` — the flow→action solver (the part we lift; sim-agnostic).
- `experiment/` — eval/execute loop (`benchmark_mw.sh` runs the full
  generate→act→execute loop end-to-end on Meta-World as a reference).
- `metaworld/` — Meta-World env glue (we replace this with ManiSkill).
- Main repo `flow-diffusion/AVDC` — the (Meta-World/iTHOR/Bridge) video models;
  **we do NOT use their generator — PSIVG is our generator.**

> **Why this fits "code as little as possible":** AVDC's extractor needs exactly
> {flow, depth, object mask, K}. **PSIVG already produces all of them** (see §3).
> So we skip GMFlow, skip any depth model, and train nothing — we feed PSIVG's
> native outputs into the closed-form solver. The only sim-specific work is the
> task scene + the controller that follows the SE(3) subgoals.

### 2.2 UniPi — learned IDM (cite, do NOT build)
*Du et al., "Learning Universal Policies via Text-Guided Video Generation",
NeurIPS 2023. arXiv:2302.00111.*
Small CNN: 3×3 conv → 3 residual conv blocks → mean-pool → MLP(128, 7) → 7-dim
control. Trained Adam, lr 1e-4, ~2M steps, 10k warmup, on **action-labeled**
(frameₜ, frameₜ₊₁) pairs. No official public code found. This is our
**conceptual baseline framework** and the motivating quote ("hallucinations …
unfaithful … not in the physical world") — but as an implementation it needs
labeled data + training, so we cite it rather than build it.

### 2.3 EVA — reference / motivation only (2026)
*"EVA: Aligning Video World Models with Executable Robot Actions via Inverse
Dynamics Rewards", arXiv:2603.17808.*
Names the **"executability gap"**: visually-coherent rollouts violate
kinematics → infeasible actions. Trains an IDM, freezes it, uses it as an RL
reward to align the generator (SFT → GRPO). Benchmarks: RoboTwin 2.0 + real
dual-arm robot (**not** ManiSkill). Result: physics alignment 52.6% vs 46.2%;
**physics-free baselines fail hardest on contact-rich tasks.** → strongest recent
citation for *why PSIVG helps*. Related Work / motivation only.

---

## 3. PSIVG outputs ↔ AVDC inputs (the key synergy)

| AVDC needs | PSIVG already produces | Pipeline location |
|---|---|---|
| Optical flow | flow field (`*_flow_visual.mp4` + raw flow) | Step 2 render / Step 3 |
| Depth (initial / per-frame) | ViPE monocular depth (EXR, channel `Z`) | Step 2 ViPE (`OUT_ViPE_Raw/depth`) |
| Object mask | per-frame dynamic foreground mask (`obj_mask.mp4`) | Step 2/3 |
| Camera intrinsics K + poses | ViPE camera poses (`all_c2w.npz`) + intrinsics | Step 2 ViPE export |
| 3D object geometry | reconstructed mesh (`meshes/*.obj`) | Step 1 |
| RGB frames | final generated video | Step 4 |

**Net new perception code ≈ 0** — we route existing files into the AVDC solver.
(For the *baseline* generator `gen_template_video.py`, which has no ViPE/flow, we
fall back to GMFlow + sim-depth, exactly as AVDC does — that's the only place we
need AVDC's own flow/depth front-end.)

---

## 4. Pipeline Step B — Simulator: ManiSkill3 (locked)

*Tao et al., "ManiSkill3: GPU Parallelized Robotics Simulation and Rendering",
arXiv:2410.00425. Docs: https://maniskill.readthedocs.io (current 3.0.0bxx).*

Current release supersedes ManiSkill2; GPU-parallel; **ray-traced rendering**
(important — see §6 generation-domain risk); env-exposed ground-truth pose +
success. Built-in tooling we reuse instead of writing:

- **Demo generation (motion planning):**
  `python -m mani_skill.examples.motionplanning.panda.run -e "PickCube-v1"`
  (headless, saves video; `--vis` for GUI). Gives demo clips **and** ground-truth
  trajectories for free.
- **RGB(+depth) render to video:**
  `python -m mani_skill.examples.demo_random_action -e <ENV> --render-mode="rgb_array" --record-dir=videos`
  and `... benchmarking.gpu_sim -e <ENV> --save-video --render-mode="sensors"`.
- **Teleop / record demos:**
  `python -m mani_skill.examples.teleoperation.interactive_panda -e "StackCube-v1"`.
- **YCB objects (incl. apple):** `PickSingleYCB-v1`.
- Ground-truth object pose + success/failure: exposed per-env (info dict).

**Why ManiSkill3 over Meta-World** (which would reuse AVDC's loop with *less*
code): Meta-World renders are flat/cartoonish and its tasks are quasi-static.
ManiSkill3's realism gives CogVideoX a fighting chance to generate usable video
(the #1 risk, §6), and we can pick/build a passive-physics task that actually
exercises PSIVG's advantage. The AVDC solver is sim-agnostic, so the porting cost
is the task scene + controller, not the math.

---

## 5. Staged plan (minimal-code-first, de-risked)

- **P0 — Source of truth.** *(this document — done)*
- **P1 — Risk spike (M0):** feed a ManiSkill3 first-frame + prompt to BOTH
  `gen_template_video.py` (baseline) and the full PSIVG pipeline; confirm the
  generator produces a coherent sim-domain video. **Retire the generation-domain
  risk before any integration.** Reuse the pipeline already standing on the pod.
- **P2 — Lift the solver:** extract AVDC's closed-form flow+depth→SE(3) module
  from `flowdiffusion/`; wire PSIVG's native flow/depth/mask/K into it (no
  GMFlow, no training). For the baseline branch, use AVDC's GMFlow + sim-depth.
- **P3 — Compare:** same task, same extractor — physics-free baseline vs PSIVG.
  Metrics: task success (e.g. landed-in-bowl yes/no) + SE(3)/centroid trajectory
  error vs. ground truth + a physical-plausibility proxy (penetration, energy).
- **P4 — (optional, strongest result):** custom passive-physics task
  ("toss apple → bowl") to maximize the physics gap.

Planned repo layout (created lazily, only as each phase needs it):
```
research_timur/robotics_poc/
  SOURCE_OF_TRUTH.md      # this file
  env/                    # ManiSkill3 + AVDC env notes
  gen_demos.py            # ManiSkill3 motion-planning demos → clips + GT
  to_psivg_input.py       # clip → PSIVG contract (49f, 480×720, .mp4 + .json)
  run_psivg.sh            # wrap the existing 4-step pipeline on the pod
  action_extract/         # AVDC solver (lifted) fed by PSIVG outputs
  execute.py              # follow SE(3) subgoals in ManiSkill3, log success
  eval_compare.py         # PSIVG vs baseline table
```

---

## 6. Risks (ranked)

1. **Generation-domain gap (highest).** CogVideoX is trained on *real* video; can
   it generate a coherent clip from a *sim* first-frame? Mitigation: ManiSkill3
   ray-traced render (realistic) + the P1 spike before anything else. If it
   fails even on realistic renders, fall back to evaluating on *real* videos
   (loses GT/execution — weaker PoC).
2. **Actuation gap.** PSIVG's MPM simulates the **object**, not the robot arm
   forcing it. → choose **brief-contact / passive-dynamics** tasks; AVDC's
   controller chases *object* SE(3) subgoals, so it tolerates this as long as the
   *object* motion is right (which is exactly what PSIVG gets right).
3. **PSIVG perception on sim renders.** ViPE/DINO/SAM/InstantMesh on sim frames —
   validated implicitly in P1.
4. **Sim mismatch.** AVDC's grasp/push controller is tuned for Meta-World; porting
   the execution controller to ManiSkill3 is the main genuine code cost.

---

## 7. References
- UniPi — arXiv:2302.00111 — https://arxiv.org/pdf/2302.00111
- AVDC — arXiv:2310.08576 — paper https://flow-diffusion.github.io/AVDC.pdf ;
  code https://github.com/flow-diffusion/AVDC and
  https://github.com/flow-diffusion/AVDC_experiments
- EVA — arXiv:2603.17808 — https://arxiv.org/html/2603.17808v1
- ManiSkill3 — arXiv:2410.00425 — docs https://maniskill.readthedocs.io
- PSIVG generator entry point — `research_timur/gen_template_video.py`
