# sim/

Post #1: can demonstrations made in simulation replace, or add to, the ones I record by hand? Started Sept 14, 2026.

**Question.** SmolVLA fine-tuned on 100 real demonstrations scored 9 / 20 on the real arm (`../experiments/`). Same
model, same recipe, same task and real-arm protocol; only the source of the demonstrations changes.

| condition | trained on | evaluated | status |
|---|---|---|---|
| A · real | 100 real teleop demos | real arm, 20 trials: 9 / 20 | done (`../experiments/`) |
| C · sim only | 100 scripted sim demos with domain randomization | in sim, then real arm, 20 trials | |
| D · sim + real | the 100 sim demos + the 100 real demos, co-trained | real arm, 20 trials | |

One figure carries the result: a bar per evaluation (in sim, on the real arm) per condition, with 95% intervals. The
gap inside a condition is the sim-to-real gap; the gap across conditions is the answer. Plus the failure maps, stage
scoring, a reel of the sim demonstrations, a reel of the sim policy on the real arm, and a cost-per-demonstration table.

**Tooling.** Isaac Sim / Isaac Lab on a rented Lambda A6000 (Modal cannot run Isaac Sim: its sandbox exposes CUDA but not
the GPU's graphics path, tested Sept 14, `tools/modal_isaac_smoke.py`). NVIDIA's
[Sim-to-Real SO-101 workshop](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/) supplies the arm
model and the scene, randomization and recording tools; its vial-to-rack task and GR00T are not used. Training stays on
Modal; evaluation on the real arm uses the `experiments/tools/eval.py` protocol.

**Steps.** (1) Isaac Sim headless on the VM, the workshop's SO-101 in an empty scene, one frame (`tools/vm_setup.sh` +
`tools/isaac_smoke_so101.py`). (2) The workshop's sim image (Isaac Lab 2.3 + LeRobot 0.4.3, headless) and our scene: mat, two
bowls, red and blue block, cameras named `camera_top` / `camera_wrist` at the real placements and 640×480; rendered frames
next to real ones, and a real episode's joint trajectory replayed on the sim arm as the first comparison. (3) Our own
scripted pick-and-place agent (the workshop records only by leader-arm teleop, which needs the arm on the VM and a window):
sample block and bowl poses, drive the arm through waypoints with Isaac Lab's IK, record through the workshop's LeRobot
recorder, with domain randomization; check per-joint `observation.state` ranges against the real dataset before training;
the sim reel. (4) Fine-tune C and D on Modal. (5) Evaluate C in sim, then C and D on the real arm. (6) Write-up.

Results go in numbered folders here, each with a `README.md` and `methods.md`; `tools/` holds the launchers and the
VM scripts; `so101_blocks/` is the scene package (Isaac Lab environment, rig geometry, unit mapping, asset generator,
headless scripts). Done: `01_pipeline/` (step 1), `02_scene/` (step 2: scene, replay, map fitted to tape measurements),
`03_sim_demos/` (step 3: 100 scripted demos, `timzwu/so101_blocks_sim`).
