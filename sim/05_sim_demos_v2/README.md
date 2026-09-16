# 05 · 200 simulated demonstrations, second recipe

Step 3 again, with the scripted agent changed along the axes the first round pointed at: the first 100 sim demos
gave 0 / 20 alone and 4 / 20 with the real data (`04_real_evals/`), and in the simulator the sim-trained policy
reached the block 13 times in 20 without ever closing the gripper. The data was too uniform to teach the grasp
decision. Sept 16, 2026.

**Result.** 200 episodes, 100 red → left and 100 blue → right, from 269 attempts (69 re-rolled). 29 of the 200 are
recovery episodes: a deliberate miss, the jaws close on nothing, reopen, and the second grasp succeeds. 2.5 A10 hours
of recording. Dataset: `timzwu/so101_blocks_sim_v2` on the Hub (private); merged with the 150 real episodes as
`timzwu/so101_blocks_simreal_v2`.

| | v1 (`03_sim_demos/`) | v2 (this folder) | real demos (post #0) |
|---|---|---|---|
| episodes, trained pair | 100 | 200 | 100 (of 150) |
| recovery episodes | 0 | 29 (15%) | a handful, by accident |
| length per episode | 11–14 s (median 12.2) | 10–20 s (median 13.2) | 18–30 s (median ~18) |
| frames | 36,771 | 82,126 | 54,000 |
| grasp: jaw tilt from vertical | median 3.6°, max 8.6° | median 7.1°, max 23° | not measured |
| grasp: jaw heading | nearest of 4 | nearest (60%) or second (40%) of 4, ±90° on retry | operator's choice |
| gripper opening on approach | 20 | 22–35 | up to 37 |
| hover height | 10 cm | 6–14 cm | varies |
| wrist roll used | ±54° | ±180° | −139° to +45° |
| block positions | zone minus a 5 cm margin | up to the tape's inner edge; 30% on a sticker | zone, random |
| mat | drawn | photo of the real mat, tape zone and stickers from the real overhead frames | the real mat |
| appearance | robot colour, lights, camera pose ±2 cm | + block and bowl colour jitter, camera pose ±3 cm, mat tint | |
| first-try success of the agent | 93% | 74% | 100 of 101 kept |

Kept episodes are unaffected by the re-rolls (a re-rolled attempt is discarded whole); the lower first-try rate is
the price of the tilted, off-axis approaches, see `methods.md`.

Reels, 10x: `sim_demos_v2_10x.mp4` (overhead | wrist, the dataset's own cameras, first ten episodes) and
`sim_demos_v2_side_10x.mp4` (a third-person view of every tenth episode, not in the dataset).

Fine-tunes on this data: C2 (200 sim episodes, first 100 used), D2 (100 real + 100 sim v2), and a 75 / 25 real : sim
split (100 real + 33 sim v1); rows in `../../experiments/results/training_runs.md`. In the simulator C2 scored 2 / 20
against C's 0 / 20 and closed on the block where C never did; on the real arm it scored 0 / 20, like C
and co-trained with the real demos 7 / 20 against D's 4 / 20; the 75 / 25 split on v1 data scored 3 / 20 (`../04_real_evals/`).

`methods.md` has the changed knobs, the commands, the re-roll analysis and the joint-range check against the real data.
`so101_blocks_sim_v2_extras/record_log.json` has every attempt.
