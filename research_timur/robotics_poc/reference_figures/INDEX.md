# Reference pipeline figures (for the robotics-extension draft)

Pipeline / architecture figures pulled from the relevant papers, for the
"preliminary" draft of the PSIVG-for-robot-manipulation section. All are the
authors' own figures — cite, do not claim as ours. Downloaded from the arXiv
HTML (ar5iv / arxiv.org/html) versions.

| file | paper | fig | what it shows | use it for |
|------|-------|-----|---------------|------------|
| `unipi_fig1_overview.png` | UniPi (2302.00111) | Fig 1 | text-conditional video gen as universal policies, trained on sim/real/YouTube | motivation: video-gen as a general policy substrate |
| `unipi_fig2_pipeline.png` | UniPi (2302.00111) | Fig 2 | **obs + text → Video Diffusion → Temporal Super-Res → Inverse Dynamics → robot actions** | the canonical "video → IDM → action" framework our method follows |
| `avdc_fig2_framework.png` | AVDC (2310.08576) | Fig 2 | **RGBD + goal → video pred → optical flow → SE(3) (flow+depth) → object pose & arm command** | the closed-form, training-free action-extraction path we reuse |
| `avdc_fig3_architecture.png` | AVDC (2310.08576) | Fig 3 | factorized spatial-temporal U-Net video diffusion backbone | only if we discuss their generator internals (optional) |
| `eva_fig1_executability_gap.png` | EVA (2603.17808) | Fig 1 | the **executability gap**: kinematic artifacts in generated video → bad IDM actions; reward-aligned model fixes it | motivation: why physically-faithful video matters for actions (our core argument) |
| `eva_fig2_artifacts_to_jitter.png` | EVA (2603.17808) | Fig 2 | how visual artifacts become 7-DOF joint jitter (reward 7.94 vs 3.04) | concrete evidence that video quality → action quality |
| `psivg_fig1_teaser.png` | PSIVG (2603.06408) | Fig 1 | baseline (chaotic motion) vs PSIVG (physically plausible) | the parent method's value prop |
| `psivg_fig2_pipeline.png` | PSIVG (2603.06408) | Fig 2 | **full 4-stage PSIVG pipeline**: prompt → template video → perception/4D → physics sim → guided generation (+TTCO) | the base pipeline our robotics extension plugs into |
| `psivg_fig3_perception.png` | PSIVG (2603.06408) | Fig 3 | perception sub-steps (detect → mesh → background geometry → dynamics) | detail on Step 1 if needed |
| `psivg_fig4_ttco.png` | PSIVG (2603.06408) | Fig 4 | TTCO: learnable embeddings optimized against simulator outputs | detail on Step 4 / texture consistency |

## Recommended for the preliminary draft (minimal set)

The story is: *video-gen-as-policy needs physically-faithful video, and PSIVG
provides exactly that.* Strongest 3-figure spine:

1. **`unipi_fig2_pipeline.png`** — establish the framework (video → IDM → action).
2. **`eva_fig1_executability_gap.png`** — establish the problem (un-physical video → unexecutable actions). This is our motivation in one picture.
3. **`avdc_fig2_framework.png`** — establish our action-extraction route (flow + depth → SE(3) → arm command), the training-free path PSIVG's outputs feed directly.

Then **`psivg_fig2_pipeline.png`** to show what we extend.

## Citations (bibtex keys to add)

- UniPi — Du et al., "Learning Universal Policies via Text-Guided Video Generation," NeurIPS 2023. arXiv:2302.00111
- AVDC — Ko et al., "Learning to Act from Actionless Videos through Dense Correspondences," ICLR 2024. arXiv:2310.08576
- EVA — "Executable Video world modeling / closing the executability gap," 2026. arXiv:2603.17808
- PSIVG — Foo et al., "Physical Simulator In-the-Loop Video Generation," CVPR 2026. arXiv:2603.06408

NOTE: verify EVA's exact title/authors and AVDC venue before final submission.
