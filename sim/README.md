# Simulation experiments

[Article figures, tables, and videos](article-media/)

Can simulated demonstrations replace or improve real robot training data? I recreated the SO-101
pick-and-place task in Isaac Sim and compared SmolVLA fine-tuned on real, simulated, and combined data.

[The Simulation Gap in Robotics](https://timzwu.substack.com/p/the-simulation-gap-in-robotics) · Post #1

| Training data | Real arm | Simulator |
|---|---|---|
| Real only · 100 real | 9/20 | 0/20 |
| Sim v1 only · 100 sim | 0/20 | 0/20 |
| Co-trained v1 · 100 real + 100 sim v1 | 4/20 | 1/20 |
| Sim v2 only · 100 sim | 0/20 | 2/20 |
| Co-trained v2 · 100 real + 100 sim v2 | 7/20 | 4/20 |
| 75/25 real/sim · 100 real + 33 sim v1 | 3/20 | Not run |

Sim v2 added retries, varied grasps and timing, broader block placement, and a photo-textured mat.
Co-training improved from 4/20 to 7/20 on the physical arm, below the 9/20 real-only baseline.
The sim-trained policies also struggled inside the simulator: transfer alone does not explain the failures.

All runs used the same pretrained model, 20,000 training steps, and batch size 64. Evaluations used
20 marked positions and 30 seconds of arm motion per trial. See [evaluation methods](04_real_evals/methods.md)
for data reuse, scoring, and the change in simulated mat appearance between rounds.

| Folder | Contents |
|---|---|
| [Pipeline check](01_pipeline/) | Headless physics and rendering on a cloud GPU |
| [Scene calibration](02_scene/) | Tabletop geometry, camera matching, and real trajectory replays |
| [Sim v1 demonstrations](03_sim_demos/) | Scripted demonstrations and comparison with teleoperation |
| [Policy evaluations](04_real_evals/) | Both rounds of real-arm and simulator results, CSVs, charts, and videos |
| [Sim v2 demonstrations](05_sim_demos_v2/) | Changes to appearance, approaches, retries, and placement |

[`so101_blocks/`](so101_blocks/) contains the Isaac Lab environment and demonstration scripts;
[`tools/`](tools/) contains cloud launchers, policy serving, plotting, and video tools.
Simulation ran on Lambda GPUs; fine-tuning ran on Modal.
