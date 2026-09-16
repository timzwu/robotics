# Sim v2 demonstrations

Sim v1 produced uniform movements but weak learned policies. Sim v2 tested whether more varied
approaches, recovery attempts, and a closer visual match would improve learning.

| Feature | Sim v1 | Sim v2 |
|---|---|---|
| Recovery attempts | None | Deliberate miss followed by a retry, targeting 15% of episodes |
| Approach | Straight down, 10 cm hover | Tilt up to 25°, hover 6–14 cm |
| Jaw heading | Nearest valid heading | Nearest 60%, second nearest 40% |
| Gripper opening | 20 | 22–35 on the arm's control scale |
| Grasp and timing | Consistent grasp point, segment timing ±20% | Grasp adjustment ±4 mm, varied closing command, timing ×0.6–1.4 |
| Block placement | 5 cm margin inside the zone | Broader coverage; 30% of starts near a marked evaluation position |
| Appearance | Drawn mat, randomized robot and lighting | Photo-textured mat, added block/bowl color variation, wider overhead-camera pose variation |

The fine-tunes used 100 sim v2 demonstrations, alone or co-trained with the same 100 real demonstrations.
Recording averaged about 135 attempts and 76 minutes per 100 accepted episodes on an A10.

Co-trained v2 scored 7/20 on the physical arm and 4/20 in simulation, up from 4/20 and 1/20 for v1.
Sim-only v2 scored 0/20 on the arm and 2/20 in simulation. The revised data improved co-training,
but the real-only policy still led on the arm at 9/20.

[Methods and generation logs](methods.md) · [Evaluation results and videos](../04_real_evals/) ·
[Training log](../../experiments/results/training_runs.md)
